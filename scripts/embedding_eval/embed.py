"""T2: embed queries and chunks for every model in the registry.

Output layout (one directory per model key under `<data_dir>/embeddings/`):

    <key>/queries.npy          float32 [n_queries, dim], L2-normalised, rows = queries.parquet order
    <key>/chunks_c512.npy      float32 [n_chunks, dim],  rows = chunks_c512.parquet order
    <key>/chunks_c2048.npy     (only for models with that chunking)
    <key>/meta.json            model id, dim, prompts, chunkings, row order, throughput, sanity cosine

`meta.json` is written last; its presence marks the model as done (resume skips it).
Load with `scripts.embedding_eval.embed.load_embeddings`.

Run: python -m scripts.embedding_eval.embed   (configure with EVAL_* env vars)
"""

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
        ModelSpec(
            "qwen4b", "Qwen/Qwen3-Embedding-4B", 2560, ("c512",), QWEN_QUERY_PROMPT, batch_size=32
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


def embed_model(
    spec: ModelSpec, cfg: EmbedConfig, queries: pd.DataFrame, chunks: dict[str, pd.DataFrame]
) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(spec.model_id, device="cuda", model_kwargs={"dtype": torch.float16})
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    d = out_dir(f"{cfg.data_dir}/embeddings/{spec.key}")
    meta: dict = {
        "key": spec.key,
        "model_id": spec.model_id,
        "dim": spec.dim,
        "dtype": "float16",
        "prompts": {"query": spec.query_prompt, "passage": spec.passage_prompt},
        "normalised": True,
        "chunkings": {},
        "throughput": {},
        "sanity_cosine": {},
    }

    model.max_seq_length = MAX_SEQ_LEN["c512"]
    t0 = time.perf_counter()
    qv, q_tok = encode(
        model, queries["text"].tolist(), spec.query_prompt, spec.batch_size, cfg.block_size
    )
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
        "n": len(qv),
        "seconds": q_dt,
        "texts_per_s": len(qv) / q_dt,
        "tokens_per_s": q_tok / q_dt,
    }

    for name in spec.chunkings:
        df = chunks[name]
        model.max_seq_length = MAX_SEQ_LEN[name]
        bs = spec.batch_size if name == "c512" else max(spec.batch_size // 4, 1)
        logger.info("{} {}: {} chunks, batch {}", spec.key, name, len(df), bs)
        t0 = time.perf_counter()
        cv, n_tok = encode(model, df["text"].tolist(), spec.passage_prompt, bs, cfg.block_size)
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
    meta["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    (d / "meta.json").write_text(json.dumps(meta))  # written last = done marker

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
    queries = pd.read_parquet(base / "queries.parquet")
    chunks = {
        n: pd.read_parquet(
            base / f"chunks_{n}.parquet", columns=["chunk_uid", "judgment_id", "text"]
        )
        for n in MAX_SEQ_LEN
    }
    summary: list[tuple[str, str, str]] = []
    for key in cfg.models:
        spec = REGISTRY[key]
        if cfg.requery:
            if (base / "embeddings" / key / "meta.json").exists():
                update_throughput(base, requery_model(spec, cfg, queries))
                summary.append((key, "requeried", ""))
            continue
        if (base / "embeddings" / key / "meta.json").exists() and not cfg.force:
            logger.info("{}: already done, skipping", key)
            summary.append((key, "skipped (exists)", ""))
            continue
        try:
            logger.info("=== {} ({})", key, spec.model_id)
            update_throughput(base, embed_model(spec, cfg, queries, chunks))
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
