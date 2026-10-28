"""T3: exact (brute-force cosine) dense retrieval eval with MaxP judgment aggregation.

For every model with a `meta.json` and each of its chunkings, all chunk vectors are scored
against all queries, each judgment is scored as the max over its chunks (MaxP) and judgments
are ranked. Each query has exactly one relevant judgment. MRL variants are made by slicing
the first d dims and re-normalising.

Outputs under `<data_dir>`: `results_exact.csv` and `results_exact.md` (sorted by overall nDCG@10).
Re-runnable: evaluates whatever models are finished at the time.

Run: python -m scripts.embedding_eval.evaluate_exact   (configure with EVAL_* env vars,
`EVAL_DEVICE=cuda` to use the GPU; CPU by default)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

from scripts.embedding_eval.common import DEFAULT_OUT_DIR, env_override

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
    baseline: str = "bgem3_c512"
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
    chunks: np.ndarray,
    jidx: np.ndarray,
    n_judg: int,
    dim: int,
    cfg: ExactConfig,
) -> np.ndarray:
    """MaxP scores [n_judg, n_q] for cosine over the first `dim` dims (vectors re-normalised)."""
    full = chunks.shape[1]
    qs = np.ascontiguousarray(mrl_slice(q.astype(np.float32), dim, full))
    scores = np.full((n_judg, len(qs)), -np.inf, dtype=np.float32)
    if cfg.device != "cpu":
        import torch

        qt = torch.from_numpy(qs).to(cfg.device)
    for s in range(0, len(chunks), cfg.block_size):
        blk = mrl_slice(np.asarray(chunks[s : s + cfg.block_size], dtype=np.float32), dim, full)
        if cfg.device != "cpu":
            sims = (torch.from_numpy(blk).to(cfg.device) @ qt.T).cpu().numpy()
        else:
            sims = blk @ qs.T
        maxp_update(scores, sims, jidx[s : s + cfg.block_size])
    return scores


def run(cfg: ExactConfig | None = None) -> pd.DataFrame:
    cfg = env_override(cfg or ExactConfig(), "EVAL_")
    base = Path(cfg.data_dir)
    queries = pd.read_parquet(base / "queries.parquet")
    kinds = queries["kind"].to_numpy()
    logger.info("queries: {}", queries["kind"].value_counts().to_dict())

    chunk_j: dict[str, np.ndarray] = {}
    judg_ids: dict[str, np.ndarray] = {}
    target: dict[str, np.ndarray] = {}
    vpj: dict[str, float] = {}
    for ch in CHUNKINGS:
        p = base / f"chunks_{ch}.parquet"
        if not p.exists():
            continue
        jid = pd.read_parquet(p, columns=["judgment_id"])["judgment_id"]
        codes, uniques = pd.factorize(jid)
        chunk_j[ch], judg_ids[ch] = codes, uniques.to_numpy()
        pos = pd.Index(uniques).get_indexer(queries["judgment_id"])
        if (pos < 0).any():
            raise ValueError(f"{ch}: {(pos < 0).sum()} query judgments missing from chunks")
        target[ch] = pos
        vpj[ch] = len(jid) / len(uniques)
        logger.info(
            "{}: {} chunks, {} judgments, {:.2f} vectors/judgment, random recall@100={:.4f}",
            ch,
            len(jid),
            len(uniques),
            vpj[ch],
            100 / len(uniques),
        )

    rows: list[dict] = []
    per_q: dict[str, np.ndarray] = {}
    for key in cfg.models:
        if not (base / "embeddings" / key / "meta.json").exists():
            logger.warning("{}: no meta.json (not embedded yet), skipping", key)
            continue
        meta = json.loads((base / "embeddings" / key / "meta.json").read_text())
        for ch in meta["chunkings"]:
            if ch not in chunk_j:
                logger.warning("{} {}: chunks parquet missing, skipping", key, ch)
                continue
            q = np.load(base / "embeddings" / key / "queries.npy")
            chunks = np.load(base / "embeddings" / key / f"chunks_{ch}.npy", mmap_mode="r")
            full = chunks.shape[1]
            tp = meta.get("throughput", {}).get(ch, {})
            for dim in (full, *[d for d in MRL_DIMS.get(key, ()) if d < full]):
                name = f"{key}_{ch}" if dim == full else f"{key}_d{dim}_{ch}"
                scores = score_variant(q, chunks, chunk_j[ch], len(judg_ids[ch]), dim, cfg)
                pq = per_query_metrics(target_ranks(scores, target[ch]))
                per_q[name] = pq["ndcg10"]
                sb = storage_bytes(dim)
                cps = tp.get("chunks_per_s")
                n_corpus = vpj[ch] * cfg.corpus_judgments
                row = {
                    "variant": name,
                    "model": key,
                    "chunking": ch,
                    "dim": dim,
                    **summarise(pq, kinds),
                    "n_chunks": len(chunks),
                    "vectors_per_judgment": vpj[ch],
                    "corpus_vectors": n_corpus,
                    **{f"bytes_{k}": v for k, v in sb.items()},
                    **{
                        f"gb_{k}_{cfg.store_vectors // 10**6}M": v * cfg.store_vectors / 1e9
                        for k, v in sb.items()
                    },
                    **{f"gb_{k}_corpus": v * n_corpus / 1e9 for k, v in sb.items()},
                    "chunks_per_s": cps,
                    "gpu_hours_corpus": n_corpus / cps / 3600 if cps else float("nan"),
                }
                rows.append(row)
                logger.info(
                    "{}: ndcg10={:.4f} mrr10={:.4f} r@1={:.3f} r@10={:.3f} r@100={:.3f}",
                    name,
                    row["ndcg10"],
                    row["mrr10"],
                    row["recall1"],
                    row["recall10"],
                    row["recall100"],
                )
            del chunks

    if not rows:
        logger.error("no embedded models found under {}", base / "embeddings")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    d_cols = ["ndcg10_delta", "ndcg10_ci_lo", "ndcg10_ci_hi", "ci_excludes_0"]
    if cfg.baseline in per_q:
        for i, r in df.iterrows():
            dl, lo, hi = paired_bootstrap(
                per_q[r["variant"]], per_q[cfg.baseline], cfg.n_boot, cfg.seed
            )
            df.loc[i, d_cols] = [dl, lo, hi, bool(lo > 0 or hi < 0)]
    else:
        logger.warning("baseline {} not evaluated, no bootstrap deltas", cfg.baseline)
        for c in d_cols:
            df[c] = np.nan
    df = df.sort_values("ndcg10", ascending=False).reset_index(drop=True)
    df.to_csv(base / "results_exact.csv", index=False)
    md = to_markdown(df, cfg)
    (base / "results_exact.md").write_text(md)
    show(df, cfg)
    logger.info("wrote {} and {}", base / "results_exact.csv", base / "results_exact.md")
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
