"""T3: exact (brute-force cosine) dense retrieval eval with MaxP judgment aggregation.

For every model with a `meta.json` and each of its chunkings, all chunk vectors are scored
against all queries, each judgment is scored as the max over its chunks (MaxP) and judgments
are ranked. Each query has exactly one relevant judgment. MRL variants are made by slicing
the first d dims and re-normalising.

Iterative scaling: `EVAL_ITER=k` evaluates the corpus formed by the deltas 0..k (iteration 0 is
the original set). Two views per iteration: `cumulative` (queries 0..k) and `fixed` (iteration-0
queries only, against the growing corpus). Models whose deltas 0..k are not all embedded are
skipped with a warning. Variants pruned at an earlier iteration (see `decisions.json`) are not
evaluated again.

Outputs under `<data_dir>/results/iter_k/`: `results_exact.csv`, `results_exact.md`,
`decisions.json`, `decisions.md`, `manifest.json`, `throughput.json`.
Re-runnable: evaluates whatever models are finished at the time.

Run: python -m scripts.embedding_eval.evaluate_exact   (configure with EVAL_* env vars,
`EVAL_DEVICE=cuda` to use the GPU; CPU by default)
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

from scripts.embedding_eval.common import DEFAULT_OUT_DIR, env_override
from scripts.embedding_eval.iterations import (
    data_dir,
    emb_dir,
    git_sha,
    read_done_meta,
    results_dir,
)

console = Console(width=max(Console().width, 200))

CHUNKINGS = ("c512", "c2048")
KINDS = ("verbatim", "abstractive")
# MRL slices per model (full dim is always evaluated); bgem3 is the non-MRL control
MRL_DIMS: dict[str, tuple[int, ...]] = {
    "qwen06": (128, 256, 512),
    "qwen4b": (1024, 512, 256),
    "bgem3": (512,),
}
MODEL_ORDER = ("qwen06", "bgem3", "mmlw_e5", "mmlw_rob2", "qwen4b", "gemma")


@dataclass
class ExactConfig:
    data_dir: str = DEFAULT_OUT_DIR
    device: str = "cpu"  # "cuda" only when a GPU is free
    block_size: int = 32768  # chunk rows per matmul block
    n_boot: int = 1000
    seed: int = 42
    iter: int = 0  # EVAL_ITER: evaluate deltas 0..iter
    baseline: str = "bgem3_c512"
    # pre-registered pruning rule: prune a variant whose nDCG@10 delta vs the current best has a
    # 95% CI upper bound below this; reference configs are never pruned
    prune_threshold: float = -0.03
    reference_configs: tuple[str, ...] = ("bgem3_c512", "qwen06_d128_c2048")
    store_vectors: int = 33_000_000  # vector count for the storage extrapolation
    corpus_judgments: int = 3_000_000
    models: tuple[str, ...] = field(default_factory=lambda: MODEL_ORDER)


# ---------------------------------------------------------------- metrics


def maxp_update(scores: np.ndarray, sims: np.ndarray, jidx: np.ndarray) -> None:
    """In place: scores[j, q] = max(scores[j, q], max over chunks c of judgment j of sims[c, q]).

    scores [n_judg, n_q]; sims [n_chunks, n_q]; jidx [n_chunks] judgment index per chunk.
    """
    order = np.argsort(jidx, kind="stable")
    sj = jidx[order]
    starts = np.flatnonzero(np.r_[True, sj[1:] != sj[:-1]])
    red = np.maximum.reduceat(sims[order], starts, axis=0)
    uniq = sj[starts]
    scores[uniq] = np.maximum(scores[uniq], red)


def target_ranks(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    """1-based rank of each query's relevant judgment (ties count against it). scores [n_judg, n_q]."""
    t = scores[target, np.arange(scores.shape[1])]
    return 1 + (scores > t[None, :]).sum(axis=0)


def per_query_metrics(ranks: np.ndarray) -> dict[str, np.ndarray]:
    """Per-query metrics for a single relevant item at the given 1-based ranks."""
    r = np.asarray(ranks)
    in10 = r <= 10
    return {
        "mrr10": np.where(in10, 1.0 / r, 0.0),
        "ndcg10": np.where(in10, 1.0 / np.log2(r + 1.0), 0.0),
        "recall1": (r <= 1).astype(float),
        "recall10": in10.astype(float),
        "recall100": (r <= 100).astype(float),
    }


def summarise(pq: dict[str, np.ndarray], kinds: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, v in pq.items():
        out[name] = float(v.mean())
        for k in KINDS:
            m = kinds == k
            out[f"{k}_{name}"] = float(v[m].mean()) if m.any() else float("nan")
    return out


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Mean delta (a - b) over queries with a 95% percentile CI from paired resampling."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def bootstrap_ci(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """95% percentile CI of the mean over queries."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------- storage


def storage_bytes(dim: int) -> dict[str, float]:
    """Bytes per vector for float32 and Weaviate RQ8 / RQ4 / RQ1."""
    return {"fp32": 4.0 * dim, "rq8": dim + 16.0, "rq4": 16 + dim / 2, "rq1": dim / 8 + 8.0}


# ---------------------------------------------------------------- scoring


def mrl_slice(x: np.ndarray, dim: int, full_dim: int) -> np.ndarray:
    if dim == full_dim:
        return x
    x = x[:, :dim]
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def score_variant(
    q: np.ndarray,
    chunks: "np.ndarray | list[np.ndarray]",
    jidx: "np.ndarray | list[np.ndarray]",
    n_judg: int,
    dim: int,
    cfg: ExactConfig,
) -> np.ndarray:
    """MaxP scores [n_judg, n_q] for cosine over the first `dim` dims (vectors re-normalised).

    `chunks` / `jidx` may be lists of parts (one per iteration delta); `jidx` holds global
    judgment indices.
    """
    parts = chunks if isinstance(chunks, list) else [chunks]
    jparts = jidx if isinstance(jidx, list) else [jidx]
    full = parts[0].shape[1]
    qs = np.ascontiguousarray(mrl_slice(q.astype(np.float32), dim, full))
    scores = np.full((n_judg, len(qs)), -np.inf, dtype=np.float32)
    if cfg.device != "cpu":
        import torch

        qt = torch.from_numpy(qs).to(cfg.device)
    for part, pj in zip(parts, jparts):
        for s in range(0, len(part), cfg.block_size):
            blk = mrl_slice(np.asarray(part[s : s + cfg.block_size], dtype=np.float32), dim, full)
            if cfg.device != "cpu":
                sims = (torch.from_numpy(blk).to(cfg.device) @ qt.T).cpu().numpy()
            else:
                sims = blk @ qs.T
            maxp_update(scores, sims, pj[s : s + cfg.block_size])
    return scores


def embedding_unit(variant: str) -> str:
    """`qwen06_d128_c2048` -> `qwen06_c2048`: the model+chunking whose vectors the variant slices."""
    m = re.match(r"^(.*?)(?:_d\d+)?_(c\d+)$", variant)
    return f"{m.group(1)}_{m.group(2)}" if m else variant


def decide_pruning(
    per_q: dict[str, np.ndarray], pruned_before: list[str], cfg: ExactConfig
) -> dict:
    """Pre-registered pruning rule on per-query nDCG@10 (cumulative view).

    The current best is the variant with the highest mean nDCG@10 among the survivors. Any other
    non-reference survivor whose paired-bootstrap delta vs the best has a CI upper bound below
    `cfg.prune_threshold` is pruned. Reference configs are always kept.
    """
    cand = {n: v for n, v in per_q.items() if n not in pruned_before}
    if not cand:
        return {
            "best": None,
            "decisions": [],
            "pruned_now": [],
            "pruned_cumulative": pruned_before,
            "embed_needed": [],
        }
    best = max(cand, key=lambda n: float(cand[n].mean()))
    decisions, pruned_now = [], []
    for name in sorted(cand, key=lambda n: -float(cand[n].mean())):
        if name == best:
            decisions.append(
                {"variant": name, "ndcg10": float(cand[name].mean()), "action": "best"}
            )
            continue
        delta, lo, hi = paired_bootstrap(cand[name], cand[best], cfg.n_boot, cfg.seed)
        is_ref = name in cfg.reference_configs
        prune = hi < cfg.prune_threshold and not is_ref
        if prune:
            pruned_now.append(name)
        decisions.append(
            {
                "variant": name,
                "ndcg10": float(cand[name].mean()),
                "delta_vs_best": delta,
                "ci_lo": lo,
                "ci_hi": hi,
                "threshold": cfg.prune_threshold,
                "reference": is_ref,
                "action": "pruned"
                if prune
                else ("kept_reference" if is_ref and hi < cfg.prune_threshold else "kept"),
            }
        )
    return {
        "best": best,
        "decisions": decisions,
        "pruned_now": pruned_now,
        "pruned_cumulative": sorted(set(pruned_before) | set(pruned_now)),
        # model+chunking units that still must be embedded for later iterations
        "embed_needed": sorted({embedding_unit(n) for n in cand if n not in pruned_now}),
    }


def decisions_markdown(dec: dict, k: int, cfg: ExactConfig) -> str:
    lines = [
        f"# Pruning decisions after iteration {k}\n",
        f"Rule (pre-registered): prune a non-reference variant when the 95% paired-bootstrap CI "
        f"upper bound of its nDCG@10 delta vs the current best is below {cfg.prune_threshold}. "
        f"Reference configs ({', '.join(cfg.reference_configs)}) are always kept. "
        f"Cumulative view, current best: `{dec['best']}`.\n",
        "| variant | nDCG@10 | delta vs best [95% CI] | action |",
        "|---|---|---|---|",
    ]
    for d in dec["decisions"]:
        ci = (
            f"{d['delta_vs_best']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"
            if "delta_vs_best" in d
            else ""
        )
        lines.append(f"| {d['variant']} | {d['ndcg10']:.4f} | {ci} | {d['action']} |")
    lines.append(
        f"\nPruned now: {dec['pruned_now'] or 'none'}. Cumulative: {dec['pruned_cumulative'] or 'none'}. "
        f"Still to embed for the next iteration: {dec['embed_needed']}."
    )
    return "\n".join(lines) + "\n"


def merge_throughput(metas: list[dict], ch: str) -> dict:
    """Aggregate the per-delta throughput of a chunking: total chunks / total seconds."""
    n = sum(m.get("throughput", {}).get(ch, {}).get("n", 0) for m in metas)
    sec = sum(m.get("throughput", {}).get(ch, {}).get("seconds", 0.0) for m in metas)
    return {"n": n, "seconds": sec, "chunks_per_s": n / sec if sec > 0 else None}


def _read_meta(path: Path) -> dict:
    return json.loads(path.read_text())


def run(cfg: ExactConfig | None = None) -> pd.DataFrame:
    cfg = env_override(cfg or ExactConfig(), "EVAL_")
    base = Path(cfg.data_dir)
    k = cfg.iter
    out = results_dir(base, k)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data: concatenate deltas 0..k
    q_parts = [pd.read_parquet(data_dir(base, j) / "queries.parquet") for j in range(k + 1)]
    queries = pd.concat(q_parts, ignore_index=True)
    n0 = len(q_parts[0])
    kinds = queries["kind"].to_numpy()
    logger.info(
        "iteration {}: {} queries {}", k, len(queries), queries["kind"].value_counts().to_dict()
    )

    chunk_j: dict[str, list[np.ndarray]] = {}
    delta_rows: dict[str, list[int]] = {}
    n_judg: dict[str, int] = {}
    target: dict[str, np.ndarray] = {}
    vpj: dict[str, float] = {}
    for ch in CHUNKINGS:
        paths = [data_dir(base, j) / f"chunks_{ch}.parquet" for j in range(k + 1)]
        if not all(p.exists() for p in paths):
            continue
        parts, uniq_all, offset, total = [], [], 0, 0
        delta_rows[ch] = []
        for p in paths:
            jid = pd.read_parquet(p, columns=["judgment_id"])["judgment_id"]
            codes, uniques = pd.factorize(jid)
            parts.append((codes + offset).astype(np.int64))
            uniq_all.append(uniques.to_numpy())
            offset += len(uniques)
            total += len(jid)
            delta_rows[ch].append(len(jid))
        allu = np.concatenate(uniq_all)
        pos = pd.Index(allu).get_indexer(queries["judgment_id"])
        if (pos < 0).any():
            raise ValueError(f"{ch}: {(pos < 0).sum()} query judgments missing from chunks")
        chunk_j[ch], n_judg[ch], target[ch] = parts, len(allu), pos
        vpj[ch] = total / len(allu)
        logger.info(
            "{}: {} chunks, {} judgments, {:.2f} vectors/judgment, random recall@100={:.4f}",
            ch,
            total,
            len(allu),
            vpj[ch],
            100 / len(allu),
        )

    pruned_before: list[str] = []
    if k > 0:
        prev = results_dir(base, k - 1) / "decisions.json"
        if prev.exists():
            pruned_before = _read_meta(prev)["pruned_cumulative"]
        else:
            logger.warning("no {}: assuming nothing was pruned earlier", prev)
    if pruned_before:
        logger.info("skipping variants pruned earlier: {}", pruned_before)

    rows: list[dict] = []
    per_q: dict[tuple[str, str], np.ndarray] = {}
    models_used: dict[str, dict] = {}
    tput: dict[str, dict] = {}
    for key in cfg.models:
        metas, missing = [], []
        for j in range(k + 1):
            meta = read_done_meta(emb_dir(base, key, j) / "meta.json")  # ignores placeholders
            if meta is not None:
                metas.append(meta)
            else:
                missing.append(j)
        if missing:
            logger.warning("{}: no meta.json for iteration(s) {}, skipping model", key, missing)
            continue
        for ch in metas[0]["chunkings"]:
            if ch not in chunk_j:
                logger.warning("{} {}: chunks parquet missing, skipping", key, ch)
                continue
            files = [emb_dir(base, key, j) / f"chunks_{ch}.npy" for j in range(k + 1)]
            if not all(f.exists() for f in files):
                logger.warning("{} {}: chunk vectors missing for some deltas, skipping", key, ch)
                continue
            chunks = [np.load(f, mmap_mode="r") for f in files]
            if [len(c) for c in chunks] != delta_rows[ch]:
                logger.warning("{} {}: vector rows do not match chunk parquet, skipping", key, ch)
                continue
            q = np.concatenate(
                [np.load(emb_dir(base, key, j) / "queries.npy") for j in range(k + 1)]
            )
            if len(q) != len(queries):
                logger.warning("{} {}: query vector rows do not match, skipping", key, ch)
                continue
            full = chunks[0].shape[1]
            models_used.setdefault(
                key,
                {
                    kk: metas[0].get(kk)
                    for kk in ("model_id", "revision", "backend", "dtype", "image", "prompts")
                },
            )
            tp = merge_throughput(metas, ch)
            tput.setdefault(key, {})[ch] = {
                "per_iteration": [m.get("throughput", {}).get(ch, {}) for m in metas],
                **tp,
            }
            for dim in (full, *[d for d in MRL_DIMS.get(key, ()) if d < full]):
                name = f"{key}_{ch}" if dim == full else f"{key}_d{dim}_{ch}"
                if name in pruned_before:
                    logger.info("{}: pruned earlier, skipping", name)
                    continue
                scores = score_variant(q, chunks, chunk_j[ch], n_judg[ch], dim, cfg)
                ranks = target_ranks(scores, target[ch])
                del scores
                sb = storage_bytes(dim)
                cps = tp["chunks_per_s"]
                n_corpus = vpj[ch] * cfg.corpus_judgments
                for view, sel in (("cumulative", slice(None)), ("fixed", slice(0, n0))):
                    pq = per_query_metrics(ranks[sel])
                    per_q[(view, name)] = pq["ndcg10"]
                    row = {
                        "iter": k,
                        "view": view,
                        "n_judgments": n_judg[ch],
                        "n_queries": len(pq["ndcg10"]),
                        "variant": name,
                        "model": key,
                        "chunking": ch,
                        "dim": dim,
                        **summarise(pq, kinds[sel]),
                        "n_chunks": sum(delta_rows[ch]),
                        "vectors_per_judgment": vpj[ch],
                        "corpus_vectors": n_corpus,
                        **{f"bytes_{kk}": v for kk, v in sb.items()},
                        **{
                            f"gb_{kk}_{cfg.store_vectors // 10**6}M": v * cfg.store_vectors / 1e9
                            for kk, v in sb.items()
                        },
                        **{f"gb_{kk}_corpus": v * n_corpus / 1e9 for kk, v in sb.items()},
                        "chunks_per_s": cps,
                        "gpu_hours_corpus": n_corpus / cps / 3600 if cps else float("nan"),
                    }
                    rows.append(row)
                logger.info(
                    "{}: cumulative ndcg10={:.4f} fixed ndcg10={:.4f}",
                    name,
                    per_q[("cumulative", name)].mean(),
                    per_q[("fixed", name)].mean(),
                )
            del chunks

    if not rows:
        logger.error("no embedded models found under {}", base / "embeddings")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    ex_cols = [
        "ndcg10_lo",
        "ndcg10_hi",
        "ndcg10_delta",
        "ndcg10_ci_lo",
        "ndcg10_ci_hi",
        "ci_excludes_0",
    ]
    for c in ex_cols:
        df[c] = np.nan
    df["ci_excludes_0"] = False
    for i, r in df.iterrows():
        v = per_q[(r["view"], r["variant"])]
        df.loc[i, ["ndcg10_lo", "ndcg10_hi"]] = bootstrap_ci(v, cfg.n_boot, cfg.seed)
        if (r["view"], cfg.baseline) in per_q:
            dl, lo, hi = paired_bootstrap(v, per_q[(r["view"], cfg.baseline)], cfg.n_boot, cfg.seed)
            df.loc[i, ex_cols[2:]] = [dl, lo, hi, bool(lo > 0 or hi < 0)]
    if not any(b == cfg.baseline for _, b in per_q):
        logger.warning("baseline {} not evaluated, no bootstrap deltas", cfg.baseline)
    df = df.sort_values(["view", "ndcg10"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(out / "results_exact.csv", index=False)
    md = "".join(
        f"## View: {view}\n\n"
        + to_markdown(df[df["view"] == view], cfg)
        .split("\n", 1)[1]
        .replace("## Storage", "### Storage")
        for view in ("cumulative", "fixed")
    )
    head = (
        f"# Exact dense retrieval, iteration {k} (MaxP over chunks)\n\n"
        f"Corpus = deltas 0..{k}. `cumulative` = queries 0..{k}; `fixed` = the {n0} iteration-0 "
        "queries against the same corpus.\n\n"
    )
    (out / "results_exact.md").write_text(head + md)

    cum = {n: v for (view, n), v in per_q.items() if view == "cumulative"}
    dec = decide_pruning(cum, pruned_before, cfg)
    dec.update({"iter": k, "view": "cumulative", "threshold": cfg.prune_threshold})
    (out / "decisions.json").write_text(json.dumps(dec, indent=2))
    (out / "decisions.md").write_text(decisions_markdown(dec, k, cfg))
    (out / "throughput.json").write_text(json.dumps(tput, indent=2))

    n_j = next(iter(n_judg.values()))
    manifest = {
        "iter": k,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "image": os.environ.get("EVAL_IMAGE", "juddges-embedding-eval:latest"),
        "n_judgments": n_j,
        "n_queries": len(queries),
        "n_queries_fixed_view": n0,
        "queries_by_kind": {str(a): int(b) for a, b in queries["kind"].value_counts().items()},
        "chunks": {ch: sum(v) for ch, v in delta_rows.items()},
        "chunks_per_delta": delta_rows,
        "dataset_seeds": [cfg.seed + j for j in range(k + 1)],
        "bootstrap": {"n_boot": cfg.n_boot, "seed": cfg.seed, "baseline": cfg.baseline},
        "pruning": {
            "threshold": cfg.prune_threshold,
            "reference_configs": list(cfg.reference_configs),
        },
        "models": models_used,
        "variants": sorted(set(df["variant"])),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    show(df[df["view"] == "cumulative"], cfg)
    logger.info("wrote results to {}", out)
    return df


# ---------------------------------------------------------------- reporting


def _f(x, fmt="{:.3f}") -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def _delta(r) -> str:
    if pd.isna(r["ndcg10_delta"]):
        return ""
    s = f"{r['ndcg10_delta']:+.3f} [{r['ndcg10_ci_lo']:+.3f}, {r['ndcg10_ci_hi']:+.3f}]"
    return s + (" *" if r["ci_excludes_0"] else "")


def to_markdown(df: pd.DataFrame, cfg: ExactConfig) -> str:
    m = cfg.store_vectors // 10**6
    q = [
        "| variant | dim | nDCG@10 | MRR@10 | R@1 | R@10 | R@100 | verb nDCG@10 | abstr nDCG@10 "
        f"| verb R@100 | abstr R@100 | dnDCG@10 vs {cfg.baseline} [95% CI] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        q.append(
            f"| {r['variant']} | {r['dim']} | {_f(r['ndcg10'])} | {_f(r['mrr10'])} "
            f"| {_f(r['recall1'])} | {_f(r['recall10'])} | {_f(r['recall100'])} "
            f"| {_f(r['verbatim_ndcg10'])} | {_f(r['abstractive_ndcg10'])} "
            f"| {_f(r['verbatim_recall100'])} | {_f(r['abstractive_recall100'])} | {_delta(r)} |"
        )
    s = [
        f"| variant | dim | vec/judgment | B/vec fp32 | rq8 | rq4 | rq1 | GB@{m}M fp32 | rq8 | rq4 "
        "| rq1 | corpus vectors (M) | corpus GB fp32 | rq8 | rq4 | rq1 | chunks/s | GPU-h corpus |",
        "|" + "---|" * 18,
    ]
    for _, r in df.iterrows():
        s.append(
            f"| {r['variant']} | {r['dim']} | {r['vectors_per_judgment']:.2f} "
            f"| {r['bytes_fp32']:.0f} | {r['bytes_rq8']:.0f} | {r['bytes_rq4']:.0f} | {r['bytes_rq1']:.0f} "
            + "".join(f"| {r[f'gb_{k}_{m}M']:.1f} " for k in ("fp32", "rq8", "rq4", "rq1"))
            + f"| {r['corpus_vectors'] / 1e6:.1f} "
            + "".join(f"| {r[f'gb_{k}_corpus']:.1f} " for k in ("fp32", "rq8", "rq4", "rq1"))
            + f"| {_f(r['chunks_per_s'], '{:.1f}')} | {_f(r['gpu_hours_corpus'], '{:.1f}')} |"
        )
    return (
        "# Exact dense retrieval (MaxP over chunks)\n\n"
        f"Sorted by overall nDCG@10. `*` = 95% paired-bootstrap CI ({cfg.n_boot} resamples, "
        f"seed {cfg.seed}) excludes 0. d = slice of first d dims, re-normalised.\n\n"
        + "\n".join(q)
        + f"\n\n## Storage and embedding cost\n\nRQ8 = dim+16 B, RQ4 = 16+dim/2 B, RQ1 = dim/8+8 B per "
        f"vector. Corpus = {cfg.corpus_judgments:,} judgments x vectors/judgment measured on the "
        "eval set. GPU-h = corpus vectors / measured chunks/s (full-dim model, GPU0).\n\n"
        + "\n".join(s)
        + "\n"
    )


def show(df: pd.DataFrame, cfg: ExactConfig) -> None:
    t = Table(title="exact dense eval (MaxP), sorted by nDCG@10")
    for c in (
        "variant",
        "dim",
        "nDCG@10",
        "MRR@10",
        "R@1",
        "R@10",
        "R@100",
        "verb nDCG",
        "abstr nDCG",
        f"d vs {cfg.baseline}",
        "B/vec rq8",
        f"GB@{cfg.store_vectors // 10**6}M rq8",
        "GPU-h",
    ):
        t.add_column(c, justify="left" if c == "variant" else "right")
    m = cfg.store_vectors // 10**6
    for _, r in df.iterrows():
        t.add_row(
            r["variant"],
            str(r["dim"]),
            _f(r["ndcg10"], "{:.4f}"),
            _f(r["mrr10"], "{:.4f}"),
            _f(r["recall1"]),
            _f(r["recall10"]),
            _f(r["recall100"]),
            _f(r["verbatim_ndcg10"]),
            _f(r["abstractive_ndcg10"]),
            _delta(r),
            f"{r['bytes_rq8']:.0f}",
            f"{r[f'gb_rq8_{m}M']:.0f}",
            _f(r["gpu_hours_corpus"], "{:.1f}"),
        )
    console.print(t)


if __name__ == "__main__":
    run()
