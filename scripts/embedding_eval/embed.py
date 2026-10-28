"""T2: embed queries and chunks for every model in the registry.

Output layout (one directory per model key under `<data_dir>/embeddings/`):

    <key>/queries.npy          float32 [n_queries, dim], L2-normalised, rows = queries.parquet order
    <key>/chunks_c512.npy      float32 [n_chunks, dim],  rows = chunks_c512.parquet order
    <key>/chunks_c2048.npy     (only for models with that chunking)
    <key>/meta.json            model id, dim, prompts, chunkings, row order, throughput, sanity cosine

`meta.json` is written last; its presence marks the model as done (resume skips it).
Load with `scripts.embedding_eval.embed.load_embeddings`.

Iterative scaling (`EVAL_ITER=k`, k >= 1): embeds only the delta `iter_k/queries.parquet` and
`iter_k/chunks_<chunking>.parquet` into `<key>/iter_k/`. Which model+chunking units to embed comes
from `results/iter_{k-1}/decisions.json` (`embed_needed`), or from `EVAL_UNITS="bgem3_c512,..."`
(speculative runs). Chunkings already present in an existing `iter_k/meta.json` are kept and only
the missing ones are embedded. A `meta.json` whose `status` starts with `in_progress` is a placeholder
and never counts as done.

Run: python -m scripts.embedding_eval.embed   (configure with EVAL_* env vars)
"""

import asyncio
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

from scripts.embedding_eval.common import DEFAULT_OUT_DIR, env_override, out_dir
from scripts.embedding_eval.iterations import (
    data_dir,
    emb_dir,
    embed_needed,
    parse_units,
    read_done_meta,
)

console = Console()

QWEN_QUERY_PROMPT = (
    "Instruct: Given a legal thesis, retrieve the court judgment passages that support it\nQuery:"
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    dim: int
    chunkings: tuple[str, ...]
    query_prompt: str = ""
    passage_prompt: str = ""
    optional: bool = False
    batch_size: int = 64  # for c512; c2048 batches are 4x smaller
    # "st" = sentence-transformers in-process, "tei" = text-embeddings-inference over HTTP
    backend: str = "st"
    tei_url: str = ""


# Prompts verified against the model cards. qwen4b is the slowest, so it goes last.
REGISTRY: dict[str, ModelSpec] = {
    s.key: s
    for s in [
        ModelSpec(
            "qwen06", "Qwen/Qwen3-Embedding-0.6B", 1024, ("c512", "c2048"), QWEN_QUERY_PROMPT
        ),
        ModelSpec("bgem3", "BAAI/bge-m3", 1024, ("c512", "c2048")),
        ModelSpec(
            "mmlw_e5", "sdadas/mmlw-retrieval-e5-large", 1024, ("c512",), "query: ", "passage: "
        ),
        # card: queries prefixed with "[query]: ", passages without a prefix
        ModelSpec(
            "mmlw_rob2", "sdadas/mmlw-retrieval-roberta-large-v2", 1024, ("c512",), "[query]: "
        ),
        ModelSpec(
            "gemma",
            "google/embeddinggemma-300m",
            768,
            ("c512",),
            "task: search result | query: ",
            "title: none | text: ",
            optional=True,
        ),
        # sentence-transformers manages only ~8 chunks/s here, so qwen4b goes through a TEI container
        ModelSpec(
            "qwen4b",
            "Qwen/Qwen3-Embedding-4B",
            2560,
            ("c512",),
            QWEN_QUERY_PROMPT,
            batch_size=32,
            backend="tei",
            tei_url="http://127.0.0.1:8112",
        ),
    ]
}

# max_seq_length per chunking: c512 chunks are 480 bge-m3 tokens + prompt; c2048 plus margin.
MAX_SEQ_LEN = {"c512": 512, "c2048": 2560}


@dataclass
class EmbedConfig:
    data_dir: str = DEFAULT_OUT_DIR
    models: tuple[str, ...] = tuple(REGISTRY)
    block_size: int = 4096  # texts encoded per block (sorted by length)
    force: bool = False
    requery: bool = False  # re-embed only the queries of already finished models (prompt changed)
    sanity_n: int = 200
    iter: int = 0  # EVAL_ITER: 0 = original set, k >= 1 = delta k
    units: tuple[
        str, ...
    ] = ()  # EVAL_UNITS="bgem3_c512,qwen06_c2048": overrides embed_needed (k >= 1)


def length_order(texts: list[str]) -> np.ndarray:
    """Indices sorting texts by length, longest first (cuts padding, surfaces OOM early)."""
    return np.argsort([-len(t) for t in texts], kind="stable")


def restore_order(sorted_vecs: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Undo `length_order`: row order[i] of the result is sorted_vecs[i]."""
    out = np.empty_like(sorted_vecs)
    out[order] = sorted_vecs
    return out


def l2_normalise(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def sanity_cosine(
    q_vecs: np.ndarray, c_vecs: np.ndarray, q_jids: list[str], c_jids: list[str], n: int
) -> dict[str, float]:
    """Mean cosine of a query to its own judgment's best chunk vs. a random other judgment's best chunk."""
    jids = np.asarray(c_jids)
    rng = np.random.default_rng(0)
    all_j = np.unique(jids)
    own, other = [], []
    for i in range(min(n, len(q_jids))):
        sims = c_vecs @ q_vecs[i]
        own.append(sims[jids == q_jids[i]].max())
        j = rng.choice(all_j[all_j != q_jids[i]])
        other.append(sims[jids == j].max())
    return {"own": float(np.mean(own)), "random_other": float(np.mean(other)), "n": len(own)}


def load_embeddings(
    base_dir: str | Path, key: str, chunking: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (query_vecs, chunk_vecs, meta) for a model key and chunking (rows = parquet order)."""
    d = Path(base_dir) / "embeddings" / key
    meta = json.loads((d / "meta.json").read_text())
    return np.load(d / "queries.npy"), np.load(d / f"chunks_{chunking}.npy"), meta


def encode(
    model, texts: list[str], prefix: str, batch_size: int, block_size: int
) -> tuple[np.ndarray, int]:
    """Encode in length-sorted blocks; returns (normalised vecs in input order, token count)."""
    order = length_order(texts)
    ordered = [prefix + texts[i] for i in order]
    parts, n_tokens = [], 0
    for s in range(0, len(ordered), block_size):
        block = ordered[s : s + block_size]
        ids = model.tokenizer(block, truncation=True, max_length=model.max_seq_length)["input_ids"]
        n_tokens += sum(len(x) for x in ids)
        v = model.encode(
            block,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        parts.append(l2_normalise(v))
        logger.info("  encoded {}/{}", min(s + block_size, len(ordered)), len(ordered))
    return restore_order(np.concatenate(parts), order), n_tokens


TEI_CONCURRENCY = 16


async def _tei_post(client, url: str, batch: list[str], retries: int = 6) -> np.ndarray:
    for attempt in range(retries):
        try:
            r = await client.post(
                f"{url}/embed", json={"inputs": batch, "normalize": True, "truncate": True}
            )
            if r.status_code in (413, 429, 503) or r.status_code >= 500:
                raise RuntimeError(f"TEI HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            return np.asarray(r.json(), dtype=np.float32)
        except Exception as e:
            if attempt == retries - 1:
                raise
            logger.warning("TEI request failed ({!r}), retry {}", e, attempt + 1)
            await asyncio.sleep(2**attempt)
    raise AssertionError("unreachable")


async def _tei_block(url: str, block: list[str], batch_size: int) -> np.ndarray:
    import httpx

    sem = asyncio.Semaphore(TEI_CONCURRENCY)
    async with httpx.AsyncClient(timeout=600) as client:

        async def one(b: list[str]) -> np.ndarray:
            async with sem:
                return await _tei_post(client, url, b)

        parts = await asyncio.gather(
            *(one(block[i : i + batch_size]) for i in range(0, len(block), batch_size))
        )
    return np.concatenate(parts)


def encode_tei(
    url: str, tokenizer, texts: list[str], prefix: str, batch_size: int, block_size: int
) -> tuple[np.ndarray, int]:
    """Like `encode`, but sends concurrent batches to TEI `/embed` (normalize + truncate server-side)."""
    order = length_order(texts)
    ordered = [prefix + texts[i] for i in order]
    parts, n_tokens = [], 0
    for s in range(0, len(ordered), block_size):
        block = ordered[s : s + block_size]
        n_tokens += sum(len(x) for x in tokenizer(block)["input_ids"])
        parts.append(l2_normalise(asyncio.run(_tei_block(url, block, batch_size))))
        logger.info("  encoded {}/{}", min(s + block_size, len(ordered)), len(ordered))
    return restore_order(np.concatenate(parts), order), n_tokens


def plan_work(cfg: EmbedConfig, base: Path | None = None) -> dict[str, list[str]]:
    """Model key -> chunkings to embed for `cfg.iter`, restricted to `cfg.models` and each model's chunkings."""
    base = base or Path(cfg.data_dir)
    if cfg.iter == 0:
        wanted = {k: list(REGISTRY[k].chunkings) for k in cfg.models}
    else:
        wanted = parse_units(cfg.units) if cfg.units else embed_needed(base, cfg.iter)
    plan: dict[str, list[str]] = {}
    for key in cfg.models:
        chs = [c for c in wanted.get(key, []) if c in REGISTRY[key].chunkings]
        if chs:
            plan[key] = chs
    return plan


def missing_chunkings(meta: dict | None, wanted: list[str], d: Path) -> list[str]:
    """Wanted chunkings that a finished `meta` does not (yet) cover with an existing vector file."""
    if meta is None or not (d / "queries.npy").exists():
        return list(wanted)
    have = meta.get("chunkings", {})
    return [c for c in wanted if c not in have or not (d / f"chunks_{c}.npy").exists()]


def embed_model(
    spec: ModelSpec,
    cfg: EmbedConfig,
    queries: pd.DataFrame,
    chunks: dict[str, pd.DataFrame],
    wanted: list[str] | None = None,
) -> dict:
    tei = spec.backend == "tei"
    if tei:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        model = None
    else:
        import torch
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            spec.model_id, device="cuda", model_kwargs={"dtype": torch.float16}
        )
        model.eval()
        torch.cuda.reset_peak_memory_stats()

    def run_encode(texts: list[str], prefix: str, bs: int) -> tuple[np.ndarray, int]:
        if tei:
            return encode_tei(spec.tei_url, tokenizer, texts, prefix, bs, cfg.block_size)
        return encode(model, texts, prefix, bs, cfg.block_size)

    wanted = list(wanted if wanted is not None else spec.chunkings)
    d = out_dir(emb_dir(cfg.data_dir, spec.key, cfg.iter))
    prev = read_done_meta(d / "meta.json") if cfg.iter > 0 and not cfg.force else None
    meta: dict = {
        "key": spec.key,
        "model_id": spec.model_id,
        "dim": spec.dim,
        "dtype": "float16",
        "backend": spec.backend,
        "prompts": {"query": spec.query_prompt, "passage": spec.passage_prompt},
        "normalised": True,
        "chunkings": {},
        "throughput": {},
        "sanity_cosine": {},
    }

    if prev is not None and (d / "queries.npy").exists():
        # extend a partial delta: keep its queries and the chunkings already embedded
        meta = {**meta, **prev}
        qv = np.load(d / "queries.npy")
        logger.info(
            "{} iter {}: reusing existing queries, have {}",
            spec.key,
            cfg.iter,
            list(prev["chunkings"]),
        )
    else:
        if model is not None:
            model.max_seq_length = MAX_SEQ_LEN["c512"]
        t0 = time.perf_counter()
        qv, q_tok = run_encode(queries["text"].tolist(), spec.query_prompt, spec.batch_size)
        q_dt = time.perf_counter() - t0
        if qv.shape[1] != spec.dim:
            raise ValueError(f"{spec.key}: expected dim {spec.dim}, got {qv.shape[1]}")
        np.save(d / "queries.npy", qv)
        meta["queries"] = {
            "file": "queries.npy",
            "rows": "queries.parquet",
            "n": len(qv),
            "qid": queries["qid"].tolist(),
        }
        meta["throughput"]["queries"] = {
            "backend": spec.backend,
            "n": len(qv),
            "seconds": q_dt,
            "texts_per_s": len(qv) / q_dt,
            "tokens_per_s": q_tok / q_dt,
        }
    meta["iter"] = cfg.iter

    for name in missing_chunkings(prev, wanted, d) if prev is not None else wanted:
        df = chunks[name]
        if model is not None:
            model.max_seq_length = MAX_SEQ_LEN[name]
        bs = spec.batch_size if name == "c512" else max(spec.batch_size // 4, 1)
        logger.info("{} {}: {} chunks, batch {}", spec.key, name, len(df), bs)
        t0 = time.perf_counter()
        cv, n_tok = run_encode(df["text"].tolist(), spec.passage_prompt, bs)
        dt = time.perf_counter() - t0
        np.save(d / f"chunks_{name}.npy", cv)
        norms = np.linalg.norm(cv, axis=1)
        meta["chunkings"][name] = {
            "file": f"chunks_{name}.npy",
            "rows": f"chunks_{name}.parquet",
            "n": len(cv),
            "max_seq_length": MAX_SEQ_LEN[name],
            "batch_size": bs,
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "nan": int(np.isnan(cv).sum()),
            "chunk_uid": df["chunk_uid"].tolist(),
        }
        meta["throughput"][name] = {
            "backend": spec.backend,  # "tei" = HTTP throughput incl. client overhead, not in-process ST
            "n": len(cv),
            "seconds": dt,
            "chunks_per_s": len(cv) / dt,
            "tokens": n_tok,
            "tokens_per_s": n_tok / dt,
        }
        meta["sanity_cosine"][name] = sanity_cosine(
            qv, cv, queries["judgment_id"].tolist(), df["judgment_id"].tolist(), cfg.sanity_n
        )
        logger.info(
            "{} {}: {:.1f} chunks/s, sanity {}",
            spec.key,
            name,
            len(cv) / dt,
            meta["sanity_cosine"][name],
        )
    # TEI runs in another container; its GPU memory is not visible here
    meta["peak_gpu_mem_gb"] = None if tei else torch.cuda.max_memory_allocated() / 1e9
    meta.pop("status", None)
    (d / "meta.json").write_text(json.dumps(meta))  # written last = done marker

    if not tei:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return meta


def requery_model(spec: ModelSpec, cfg: EmbedConfig, queries: pd.DataFrame) -> dict:
    """Re-embed only the queries of a finished model (e.g. after a prompt fix); chunk vectors stay untouched."""
    import torch
    from sentence_transformers import SentenceTransformer

    d = Path(cfg.data_dir) / "embeddings" / spec.key
    meta = json.loads((d / "meta.json").read_text())
    model = SentenceTransformer(spec.model_id, device="cuda", model_kwargs={"dtype": torch.float16})
    model.eval()
    model.max_seq_length = MAX_SEQ_LEN["c512"]
    t0 = time.perf_counter()
    qv, q_tok = encode(
        model, queries["text"].tolist(), spec.query_prompt, spec.batch_size, cfg.block_size
    )
    q_dt = time.perf_counter() - t0
    np.save(d / "queries.npy", qv)
    meta["prompts"] = {"query": spec.query_prompt, "passage": spec.passage_prompt}
    meta["throughput"]["queries"] = {
        "n": len(qv),
        "seconds": q_dt,
        "texts_per_s": len(qv) / q_dt,
        "tokens_per_s": q_tok / q_dt,
    }
    for name in meta["chunkings"]:
        cv = np.load(d / f"chunks_{name}.npy", mmap_mode="r")
        chunk_df = pd.read_parquet(
            Path(cfg.data_dir) / f"chunks_{name}.parquet", columns=["judgment_id"]
        )
        meta["sanity_cosine"][name] = sanity_cosine(
            qv,
            np.asarray(cv),
            queries["judgment_id"].tolist(),
            chunk_df["judgment_id"].tolist(),
            cfg.sanity_n,
        )
        logger.info("{} {}: requeried, sanity {}", spec.key, name, meta["sanity_cosine"][name])
    (d / "meta.json").write_text(json.dumps(meta))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return meta


def update_throughput(base: Path, meta: dict) -> None:
    path = base / "throughput.json"
    all_tp = json.loads(path.read_text()) if path.exists() else {}
    all_tp[meta["key"]] = {
        "model_id": meta["model_id"],
        "peak_gpu_mem_gb": meta["peak_gpu_mem_gb"],
        **meta["throughput"],
    }
    path.write_text(json.dumps(all_tp, indent=2))


def run(cfg: EmbedConfig | None = None) -> None:
    cfg = env_override(cfg or EmbedConfig(), "EVAL_")
    base = Path(cfg.data_dir)
    k = cfg.iter
    ddir = data_dir(base, k)
    if cfg.requery and k != 0:
        raise ValueError("EVAL_REQUERY only supports iteration 0")
    plan = plan_work(cfg, base)
    logger.info("iteration {}: plan {}", k, plan)
    queries = pd.read_parquet(ddir / "queries.parquet")
    needed_ch = {c for chs in plan.values() for c in chs} if not cfg.requery else set(MAX_SEQ_LEN)
    chunks = {
        n: pd.read_parquet(
            ddir / f"chunks_{n}.parquet", columns=["chunk_uid", "judgment_id", "text"]
        )
        for n in MAX_SEQ_LEN
        if n in needed_ch
    }
    summary: list[tuple[str, str, str]] = []
    for key in cfg.models:
        spec = REGISTRY[key]
        d = emb_dir(base, key, k)
        meta = read_done_meta(d / "meta.json")
        if cfg.requery:
            if meta is not None:
                update_throughput(base, requery_model(spec, cfg, queries))
                summary.append((key, "requeried", ""))
            continue
        if key not in plan:
            logger.info("{}: nothing to embed for iteration {}", key, k)
            summary.append((key, "not needed", ""))
            continue
        if meta is None and (d / "meta.json").exists() and not cfg.force:
            logger.warning("{}: in-progress placeholder meta.json, another run owns it", key)
            summary.append((key, "in progress elsewhere", ""))
            continue
        todo = plan[key] if cfg.force else missing_chunkings(meta, plan[key], d)
        if not todo and meta is not None:
            logger.info("{}: already done, skipping", key)
            summary.append((key, "skipped (exists)", ""))
            continue
        try:
            logger.info("=== {} ({}) iter {} chunkings {}", key, spec.model_id, k, plan[key])
            meta = embed_model(spec, cfg, queries, chunks, plan[key])
            if k == 0:
                update_throughput(base, meta)
            summary.append((key, "ok", ""))
        except Exception as e:  # keep going with the other models
            (
                logger.opt(exception=True).warning
                if spec.optional
                else logger.opt(exception=True).error
            )("{}: failed: {!r}", key, e)
            summary.append((key, "SKIPPED" if spec.optional else "FAILED", repr(e)[:200]))
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
    table = Table(title="embed summary")
    for c in ("model", "status", "detail"):
        table.add_column(c)
    for row in summary:
        table.add_row(*row)
    console.print(table)


if __name__ == "__main__":
    run()
