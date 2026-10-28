"""T4: Weaviate eval (HNSW fp32 vs RQ8/RQ4/RQ1, BM25, hybrid) against the exact dense baseline.

For each `model:chunking` one collection is created with the properties `chunk_text` (BM25, Polish
stopwords as in `juddges/data/documents_weaviate_db.py`), `judgment_id`, `chunk_idx` and four
self-provided named vectors holding the same embeddings: `fp32` (HNSW, quantization explicitly
skipped, because the server runs with `DEFAULT_QUANTIZATION=rq-8`), `rq8`, `rq4`, `rq1`.

Per named vector: near_vector top-200 chunks -> MaxP judgments -> MRR/nDCG/Recall (overall and by
kind, bootstrap CI of the mean and paired-bootstrap CI of the delta vs exact search of the same
model), chunk-level ANN recall@100 vs the exact top-100 (numpy), latency p50/p95/p99 (sequential,
single-threaded). Also BM25 alone and hybrid (rq8, relativeScoreFusion) at several alphas.

Iteration aware (`EVAL_ITER=k`): corpus = deltas 0..k, queries = cumulative 0..k. Iteration 0
lives in the flat layout, iteration k in `iter_k/` (parquet) and `embeddings/<key>/iter_k/`.
Views: `cumulative` (all queries) and, for k > 0, `fixed` (iteration-0 queries only).

Outputs in `<data_dir>/results/iter_k/`: `results_weaviate.csv` (long format) and
`weaviate_config_iter_k.json` (schema read back from the server, shard stats, RAM estimates).

Run: python -m scripts.embedding_eval.evaluate_weaviate   (EVAL_* env vars, see WeaviateEvalConfig)
"""

import ast
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger
from rich.console import Console
from rich.table import Table

from scripts.embedding_eval.common import DEFAULT_OUT_DIR, env_override
from scripts.embedding_eval.evaluate_exact import (
    KINDS,
    bootstrap_ci,
    maxp_update,
    paired_bootstrap,
    per_query_metrics,
    storage_bytes,
    target_ranks,
)
from scripts.embedding_eval.iterations import data_dir, emb_dir, results_dir

console = Console(width=max(Console().width, 200))

VECTORS = ("fp32", "rq8", "rq4", "rq1")
RQ_BITS = {"rq8": 8, "rq4": 4, "rq1": 1}
LONG_COLS = [
    "iter",
    "n_judgments",
    "n_queries",
    "view",
    "variant",
    "metric",
    "value",
    "ci_lo",
    "ci_hi",
]
NO_RANK = 10**9  # rank of a relevant judgment that is absent from the returned hits
STOPWORDS_SRC = (
    Path(__file__).resolve().parents[2] / "juddges" / "data" / "documents_weaviate_db.py"
)


@dataclass
class WeaviateEvalConfig:
    data_dir: str = DEFAULT_OUT_DIR
    iter: int = 0  # EVAL_ITER
    weaviate_models: tuple[str, ...] = ()  # EVAL_WEAVIATE_MODELS="bgem3:c512,qwen06:c512"
    weaviate_host: str = "localhost"
    weaviate_port: int = 8110
    weaviate_grpc_port: int = 50061
    weaviate_api_key: str = ""  # EVAL_WEAVIATE_API_KEY
    keep_collections: bool = False  # EVAL_KEEP_COLLECTIONS=1 keeps the eval collections
    top_k: int = 200  # chunks fetched per query
    ann_k: int = 100  # chunk-level recall depth
    ef_construction: int = 128
    max_connections: int = 32
    ef: int = -1  # dynamic ef (server default)
    batch_size: int = 200
    index_timeout_s: int = 3600
    warmup_queries: int = 20
    hybrid_alphas: tuple[str, ...] = ("0.25", "0.5", "0.75")
    n_boot: int = 1000
    seed: int = 42
    block_size: int = 32768  # chunk rows per exact matmul block
    corpus_judgments: int = 3_000_000  # RAM extrapolation target
    baseline: str = "bgem3_c512"
    n_default_extra: int = 2  # best configs added to the default model list


# ---------------------------------------------------------------- layout


def load_iteration(base: Path, key: str, ch: str, k: int) -> dict:
    """Concatenate deltas 0..k: queries, chunks and their embeddings (same row order)."""
    qs, cs, qv, cv, n_q0 = [], [], [], [], 0
    for j in range(k + 1):
        pdir, edir = data_dir(base, j), emb_dir(base, key, j)
        q = pd.read_parquet(pdir / "queries.parquet")
        c = pd.read_parquet(pdir / f"chunks_{ch}.parquet")
        qe = np.load(edir / "queries.npy")
        ce = np.load(edir / f"chunks_{ch}.npy")
        if len(q) != len(qe) or len(c) != len(ce):
            raise ValueError(f"{key} {ch} iter {j}: parquet/embedding row mismatch")
        if j == 0:
            n_q0 = len(q)
        qs.append(q)
        cs.append(c)
        qv.append(qe)
        cv.append(ce)
    return {
        "queries": pd.concat(qs, ignore_index=True),
        "chunks": pd.concat(cs, ignore_index=True),
        "q_vec": np.concatenate(qv).astype(np.float32),
        "c_vec": np.concatenate(cv).astype(np.float32),
        "n_q0": n_q0,
    }


# ---------------------------------------------------------------- model selection


def parse_models(specs: tuple[str, ...]) -> list[tuple[str, str]]:
    out = []
    for s in specs:
        key, _, ch = s.partition(":")
        if not key or not ch:
            raise ValueError(f"bad model spec {s!r}, expected model:chunking")
        out.append((key, ch))
    return out


def top_configs(exact: pd.DataFrame, n: int, exclude: str) -> list[str]:
    """Best `n` full-dim `model_chunking` variants by overall nDCG@10 (wide or long exact CSV)."""
    if "metric" in exact.columns:
        d = exact[(exact["metric"] == "ndcg10") & (exact["view"] == "cumulative")]
        scores = d.set_index("variant")["value"]
    else:
        if "view" in exact.columns:
            exact = exact[exact["view"] == "cumulative"]
        scores = exact.set_index("variant")["ndcg10"]
    cands = [
        v
        for v in scores.sort_values(ascending=False).index
        if re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*_c\d+", v)
    ]
    cands = [v for v in cands if not re.search(r"_d\d+_c\d+$", v) and v != exclude]
    return cands[:n]


def default_models(base: Path, k: int, cfg: WeaviateEvalConfig) -> list[tuple[str, str]]:
    p = results_dir(base, k) / "results_exact.csv"
    if not p.exists() and k == 0:
        p = base / "results_exact.csv"
    models = [("bgem3", "c512")]
    if p.exists():
        for v in top_configs(pd.read_csv(p), cfg.n_default_extra, cfg.baseline):
            key, _, ch = v.rpartition("_")
            models.append((key, ch))
    else:
        logger.warning("no exact results at {}, defaulting to bgem3:c512 only", p)
    return models


# ---------------------------------------------------------------- pure metric parts


def merge_topk(
    top_s: np.ndarray, top_i: np.ndarray, blk_s: np.ndarray, blk_i: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Merge running per-query top-k (scores, ids) [nq, k] with a block's candidates [nq, m]."""
    s = np.concatenate([top_s, blk_s], axis=1)
    i = np.concatenate([top_i, blk_i], axis=1)
    sel = np.argpartition(-s, min(k, s.shape[1]) - 1, axis=1)[:, :k]
    return np.take_along_axis(s, sel, axis=1), np.take_along_axis(i, sel, axis=1)


def exact_search(
    q: np.ndarray, chunks: np.ndarray, jidx: np.ndarray, n_judg: int, k: int, block: int
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force cosine (vectors are L2-normalised). Returns MaxP scores [n_judg, nq] and the
    exact top-k chunk row ids per query [nq, k]."""
    nq = len(q)
    scores = np.full((n_judg, nq), -np.inf, dtype=np.float32)
    top_s = np.full((nq, k), -np.inf, dtype=np.float32)
    top_i = np.full((nq, k), -1, dtype=np.int64)
    for s in range(0, len(chunks), block):
        sims = chunks[s : s + block] @ q.T  # [b, nq]
        maxp_update(scores, sims, jidx[s : s + block])
        m = min(k, sims.shape[0])
        part = np.argpartition(-sims, m - 1, axis=0)[:m].T  # [nq, m]
        blk_s = np.take_along_axis(sims.T, part, axis=1)
        top_s, top_i = merge_topk(top_s, top_i, blk_s, part + s, k)
    return scores, top_i


def ann_recall(ann_ids: list[list[int]], exact_ids: np.ndarray, k: int) -> np.ndarray:
    """Per-query |ANN top-k ∩ exact top-k| / k (chunk row ids; ANN list truncated to k)."""
    out = np.zeros(len(ann_ids))
    for i, (a, e) in enumerate(zip(ann_ids, exact_ids)):
        out[i] = len(set(a[:k]) & set(e[:k].tolist())) / k
    return out


def ranks_from_hits(hits: list[list[tuple[str, int, float]]], target_jids: list[str]) -> np.ndarray:
    """MaxP over (judgment_id, chunk_idx, score) hits, then 1-based rank of the target judgment
    (strictly-greater count + 1, like `target_ranks`). Absent target -> NO_RANK."""
    ranks = np.empty(len(hits), dtype=np.int64)
    for i, (h, t) in enumerate(zip(hits, target_jids)):
        best: dict[str, float] = {}
        for jid, _, sc in h:
            if sc > best.get(jid, -np.inf):
                best[jid] = sc
        if t not in best:
            ranks[i] = NO_RANK
        else:
            ranks[i] = 1 + sum(1 for v in best.values() if v > best[t])
    return ranks


def latency_summary(ms: list[float] | np.ndarray) -> dict[str, float]:
    a = np.asarray(ms, dtype=float)
    p50, p95, p99 = np.percentile(a, [50, 95, 99])
    return {
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
    }


def ram_gb(dim: int, vectors_per_judgment: float, corpus_judgments: int) -> dict[str, float]:
    """Vector-only RAM (GB) per variant at `corpus_judgments` judgments (no HNSW graph)."""
    n = vectors_per_judgment * corpus_judgments
    return {k: v * n / 1e9 for k, v in storage_bytes(dim).items()}


def build_rows(
    meta: dict,
    variant: str,
    view: str,
    pq: dict[str, np.ndarray],
    kinds: np.ndarray,
    exact_pq: dict[str, np.ndarray] | None,
    n_boot: int,
    seed: int,
) -> list[dict]:
    """Long-format rows: each metric (overall + per kind) with a bootstrap CI of its mean, and for
    non-exact variants `<metric>_delta` = mean delta vs exact with a paired-bootstrap CI."""
    rows = []

    def add(metric: str, value: float, lo: float = np.nan, hi: float = np.nan) -> None:
        rows.append(
            {
                **meta,
                "view": view,
                "variant": variant,
                "metric": metric,
                "value": value,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )

    for name, v in pq.items():
        for scope in ("all", *KINDS):
            m = np.ones(len(v), dtype=bool) if scope == "all" else kinds == scope
            if not m.any():
                continue
            label = name if scope == "all" else f"{scope}_{name}"
            lo, hi = bootstrap_ci(v[m], n_boot, seed)
            add(label, float(v[m].mean()), lo, hi)
            if exact_pq is not None:
                d, dlo, dhi = paired_bootstrap(v[m], exact_pq[name][m], n_boot, seed)
                add(f"{label}_delta", d, dlo, dhi)
    return rows


# ---------------------------------------------------------------- weaviate


def polish_stopwords(src: Path = STOPWORDS_SRC) -> list[str]:
    """POLISH_STOPWORDS from the production schema module (parsed, so no heavy imports)."""
    for node in ast.parse(src.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "POLISH_STOPWORDS" for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise ValueError(f"POLISH_STOPWORDS not found in {src}")


def collection_name(key: str, ch: str, k: int) -> str:
    return f"EmbEval_{key}_{ch}_i{k}".replace("-", "_")


def connect(cfg: WeaviateEvalConfig):
    from weaviate.classes.init import Auth

    import weaviate

    return weaviate.connect_to_local(
        host=cfg.weaviate_host,
        port=cfg.weaviate_port,
        grpc_port=cfg.weaviate_grpc_port,
        auth_credentials=Auth.api_key(cfg.weaviate_api_key),
    )


def create_collection(client, name: str, cfg: WeaviateEvalConfig):
    import weaviate.classes.config as wc

    if client.collections.exists(name):
        client.collections.delete(name)

    def hnsw(quantizer):
        return wc.Configure.VectorIndex.hnsw(
            distance_metric=wc.VectorDistances.COSINE,
            ef_construction=cfg.ef_construction,
            max_connections=cfg.max_connections,
            ef=cfg.ef,
            quantizer=quantizer,
        )

    q = wc.Configure.VectorIndex.Quantizer
    vector_config = [
        wc.Configure.Vectors.self_provided(name="fp32", vector_index_config=hnsw(q.none())),
        *[
            wc.Configure.Vectors.self_provided(name=n, vector_index_config=hnsw(q.rq(bits=b)))
            for n, b in RQ_BITS.items()
        ],
    ]
    return client.collections.create(
        name,
        properties=[
            wc.Property(
                name="chunk_text",
                data_type=wc.DataType.TEXT,
                tokenization=wc.Tokenization.WORD,
                index_searchable=True,
            ),
            wc.Property(
                name="judgment_id",
                data_type=wc.DataType.TEXT,
                tokenization=wc.Tokenization.FIELD,
                index_searchable=False,
                index_filterable=True,
            ),
            wc.Property(name="chunk_idx", data_type=wc.DataType.INT),
        ],
        inverted_index_config=wc.Configure.inverted_index(
            bm25_b=0.75,
            bm25_k1=1.2,
            stopwords_preset=wc.StopwordsPreset.EN,
            stopwords_additions=polish_stopwords(),
        ),
        vector_config=vector_config,
    )


def raw_get(cfg: WeaviateEvalConfig, path: str) -> dict:
    r = requests.get(
        f"http://{cfg.weaviate_host}:{cfg.weaviate_port}{path}",
        headers={"Authorization": f"Bearer {cfg.weaviate_api_key}"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def shard_stats(cfg: WeaviateEvalConfig, name: str) -> dict:
    """`/v1/nodes?output=verbose` restricted to the shards of one collection."""
    nodes = raw_get(cfg, "/v1/nodes?output=verbose")["nodes"]
    return {
        "nodes": [
            {
                **{k: v for k, v in n.items() if k not in ("shards", "stats")},
                "stats": n.get("stats"),
                "shards": [s for s in n.get("shards") or [] if s.get("class") == name],
            }
            for n in nodes
        ]
    }


def wait_indexed(cfg: WeaviateEvalConfig, name: str, n_objects: int) -> float:
    """Poll shard stats until all objects are stored and the vector queues are empty."""
    t0 = time.perf_counter()
    while True:
        shards = [s for n in shard_stats(cfg, name)["nodes"] for s in n["shards"]]
        count = sum(s.get("objectCount", 0) for s in shards)
        queue = sum(s.get("vectorQueueLength", 0) for s in shards)
        ready = all(s.get("vectorIndexingStatus", "READY") == "READY" for s in shards)
        if shards and count >= n_objects and queue == 0 and ready:
            return time.perf_counter() - t0
        if time.perf_counter() - t0 > cfg.index_timeout_s:
            raise TimeoutError(f"{name}: indexing not finished (objects={count}, queue={queue})")
        logger.info("{}: waiting for async indexing, objects={} queue={}", name, count, queue)
        time.sleep(5)


def insert_chunks(
    col, chunks: pd.DataFrame, vecs: np.ndarray, name: str, cfg: WeaviateEvalConfig
) -> float:
    t0 = time.perf_counter()
    uids = chunks["chunk_uid"].tolist()
    jids = chunks["judgment_id"].tolist()
    cidx = chunks["chunk_idx"].tolist()
    texts = chunks["text"].tolist()
    with col.batch.fixed_size(batch_size=cfg.batch_size) as batch:
        for i in range(len(chunks)):
            v = vecs[i].tolist()
            batch.add_object(
                properties={
                    "chunk_text": texts[i],
                    "judgment_id": jids[i],
                    "chunk_idx": int(cidx[i]),
                },
                uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"{name}/{uids[i]}"),
                vector={n: v for n in VECTORS},
            )
            if batch.number_errors > 100:
                raise RuntimeError("too many batch insert errors")
    failed = col.batch.failed_objects
    if failed:
        raise RuntimeError(f"{len(failed)} objects failed to insert, first: {failed[0].message}")
    return time.perf_counter() - t0


def vector_summary(schema: dict) -> dict:
    """Effective per-named-vector index config as read back from the server schema."""
    out = {}
    for vec, vc in schema["vectorConfig"].items():
        ic = vc["vectorIndexConfig"]
        out[vec] = {
            "index_type": vc["vectorIndexType"],
            "distance": ic["distance"],
            "ef_construction": ic["efConstruction"],
            "max_connections": ic["maxConnections"],
            "ef": ic["ef"],
            "dynamic_ef": [ic["dynamicEfMin"], ic["dynamicEfMax"], ic["dynamicEfFactor"]],
            "rq": ic["rq"],
            "bq_enabled": ic["bq"]["enabled"],
            "pq_enabled": ic["pq"]["enabled"],
            "sq_enabled": ic["sq"]["enabled"],
            "skip_default_quantization": ic.get("skipDefaultQuantization", False),
            "track_default_quantization": ic.get("trackDefaultQuantization", False),
        }
    return out


class Searcher:
    """Runs one query against one collection, returning (judgment_id, chunk_idx, score) hits."""

    def __init__(self, col, top_k: int):
        from weaviate.classes.query import HybridFusion, MetadataQuery

        self.col, self.top_k = col, top_k
        self.fusion = HybridFusion.RELATIVE_SCORE
        self.md_dist, self.md_score = MetadataQuery(distance=True), MetadataQuery(score=True)
        self.props = ["judgment_id", "chunk_idx"]

    @staticmethod
    def _hits(res, score) -> list[tuple[str, int, float]]:
        return [
            (o.properties["judgment_id"], int(o.properties["chunk_idx"]), score(o))
            for o in res.objects
        ]

    def dense(self, vec: np.ndarray, name: str):
        res = self.col.query.near_vector(
            near_vector=vec.tolist(),
            target_vector=name,
            limit=self.top_k,
            return_properties=self.props,
            return_metadata=self.md_dist,
        )
        return self._hits(res, lambda o: 1.0 - o.metadata.distance)

    def bm25(self, text: str):
        res = self.col.query.bm25(
            query=text,
            query_properties=["chunk_text"],
            limit=self.top_k,
            return_properties=self.props,
            return_metadata=self.md_score,
        )
        return self._hits(res, lambda o: o.metadata.score)

    def hybrid(self, text: str, vec: np.ndarray, alpha: float):
        res = self.col.query.hybrid(
            query=text,
            vector=vec.tolist(),
            target_vector="rq8",
            alpha=alpha,
            fusion_type=self.fusion,
            query_properties=["chunk_text"],
            limit=self.top_k,
            return_properties=self.props,
            return_metadata=self.md_score,
        )
        return self._hits(res, lambda o: o.metadata.score)


def timed_run(fn, n: int, warmup: int) -> tuple[list, np.ndarray]:
    """Sequentially run fn(i) for i in range(n); a few warm-up calls first. Returns results, ms."""
    for i in range(min(warmup, n)):
        fn(i)
    out, ms = [], np.empty(n)
    for i in range(n):
        t = time.perf_counter()
        out.append(fn(i))
        ms[i] = (time.perf_counter() - t) * 1000
    return out, ms


# ---------------------------------------------------------------- orchestration


def eval_model(
    client, key: str, ch: str, cfg: WeaviateEvalConfig, base: Path
) -> tuple[list[dict], dict]:
    k = cfg.iter
    tag = f"{key}_{ch}"
    d = load_iteration(base, key, ch, k)
    queries, chunks, qv, cv = d["queries"], d["chunks"], d["q_vec"], d["c_vec"]
    kinds = queries["kind"].to_numpy()
    codes, uniques = pd.factorize(chunks["judgment_id"])
    n_judg, dim = len(uniques), cv.shape[1]
    target = pd.Index(uniques).get_indexer(queries["judgment_id"])
    if (target < 0).any():
        raise ValueError(f"{tag}: query judgments missing from the corpus")
    target_jids = queries["judgment_id"].tolist()
    meta = {"iter": k, "n_judgments": n_judg, "n_queries": len(queries)}
    logger.info(
        "{}: {} chunks, {} judgments, {} queries, dim {}",
        tag,
        len(chunks),
        n_judg,
        len(queries),
        dim,
    )

    t = time.perf_counter()
    scores, exact_top = exact_search(qv, cv, codes, n_judg, cfg.ann_k, cfg.block_size)
    logger.info("{}: exact search {:.1f}s", tag, time.perf_counter() - t)
    exact_pq = per_query_metrics(target_ranks(scores, target))
    del scores

    row_of = {
        (j, int(c)): i for i, (j, c) in enumerate(zip(chunks["judgment_id"], chunks["chunk_idx"]))
    }
    name = collection_name(key, ch, k)
    col = create_collection(client, name, cfg)
    schema_created = raw_get(cfg, f"/v1/schema/{name}")
    t_insert = insert_chunks(col, chunks, cv, name, cfg)
    t_wait = wait_indexed(cfg, name, len(chunks))
    logger.info("{}: inserted in {:.1f}s, indexing finished {:.1f}s later", tag, t_insert, t_wait)
    schema = raw_get(cfg, f"/v1/schema/{name}")
    shards_after = shard_stats(cfg, name)
    count = col.aggregate.over_all(total_count=True).total_count
    if count != len(chunks):
        raise RuntimeError(f"{name}: {count} objects stored, expected {len(chunks)}")

    ram = ram_gb(dim, len(chunks) / n_judg, cfg.corpus_judgments)
    config_entry = {
        "collection": name,
        "index_params": {
            "distance": "cosine",
            "ef_construction": cfg.ef_construction,
            "max_connections": cfg.max_connections,
            "ef": cfg.ef,
            "batch_size": cfg.batch_size,
            "top_k": cfg.top_k,
        },
        "n_chunks": len(chunks),
        "n_judgments": n_judg,
        "dim": dim,
        "insert_seconds": t_insert,
        "async_index_wait_seconds": t_wait,
        "total_index_seconds": t_insert + t_wait,
        "vector_config_effective": vector_summary(schema),
        "schema_after_create": schema_created,
        "schema_after_indexing": schema,
        "nodes_verbose_shards": shards_after,
        "ram_gb_vectors_only": {f"{cfg.corpus_judgments}_judgments": ram},
    }

    searcher = Searcher(col, cfg.top_k)
    n_q = len(queries)
    texts = queries["text"].tolist()
    rows: list[dict] = []
    views = [("cumulative", slice(None))]
    if k > 0:
        views.append(("fixed", slice(0, d["n_q0"])))

    def emit(variant: str, hits: list, ms: np.ndarray | None, ann: list | None) -> None:
        pq = per_query_metrics(ranks_from_hits(hits, target_jids))
        for view, sl in views:
            m = {**meta, "n_queries": len(queries[sl]) if view == "fixed" else n_q}
            rows.extend(
                build_rows(
                    m,
                    variant,
                    view,
                    {n: v[sl] for n, v in pq.items()},
                    kinds[sl],
                    {n: v[sl] for n, v in exact_pq.items()},
                    cfg.n_boot,
                    cfg.seed,
                )
            )
            if ann is not None:
                a = np.asarray(ann)[sl]
                lo, hi = bootstrap_ci(a, cfg.n_boot, cfg.seed)
                rows.append(
                    {
                        **m,
                        "view": view,
                        "variant": variant,
                        "metric": "ann_recall100",
                        "value": float(a.mean()),
                        "ci_lo": lo,
                        "ci_hi": hi,
                    }
                )
        if ms is not None:
            for mname, val in latency_summary(ms).items():
                rows.append(
                    {
                        **meta,
                        "view": "cumulative",
                        "variant": variant,
                        "metric": mname,
                        "value": val,
                        "ci_lo": np.nan,
                        "ci_hi": np.nan,
                    }
                )

    # exact reference (same per-query metrics, no delta rows)
    rows.extend(pq_to_rows(exact_pq, kinds, views, meta, n_q, queries, f"{tag}|exact", cfg))

    for vec in VECTORS:
        hits, ms = timed_run(lambda i, v=vec: searcher.dense(qv[i], v), n_q, cfg.warmup_queries)
        ann = ann_recall(
            [[row_of[(j, c)] for j, c, _ in h if (j, c) in row_of] for h in hits],
            exact_top,
            cfg.ann_k,
        )
        emit(f"{tag}|{vec}", hits, ms, list(ann))
        logger.info(
            "{} {}: ann recall@100={:.4f} p50={:.1f}ms", tag, vec, ann.mean(), np.median(ms)
        )

    hits, ms = timed_run(lambda i: searcher.bm25(texts[i]), n_q, cfg.warmup_queries)
    emit(f"{tag}|bm25", hits, ms, None)
    for a in cfg.hybrid_alphas:
        hits, ms = timed_run(
            lambda i, a=float(a): searcher.hybrid(texts[i], qv[i], a), n_q, cfg.warmup_queries
        )
        emit(f"{tag}|hybrid_rq8_a{a}", hits, ms, None)

    static = {
        "index_seconds": t_insert + t_wait,
        "insert_seconds": t_insert,
        "n_chunks": float(len(chunks)),
    }
    for vec in VECTORS:
        rows.append(
            {
                **meta,
                "view": "cumulative",
                "variant": f"{tag}|{vec}",
                "metric": f"ram_gb_{cfg.corpus_judgments // 10**6}M_judgments",
                "value": ram[vec],
                "ci_lo": np.nan,
                "ci_hi": np.nan,
            }
        )
    for mname in ("index_seconds", "insert_seconds", "n_chunks"):
        rows.append(
            {
                **meta,
                "view": "cumulative",
                "variant": f"{tag}|all",
                "metric": mname,
                "value": static[mname],
                "ci_lo": np.nan,
                "ci_hi": np.nan,
            }
        )

    if not cfg.keep_collections:
        client.collections.delete(name)
        logger.info("deleted collection {}", name)
    return rows, config_entry


def pq_to_rows(exact_pq, kinds, views, meta, n_q, queries, variant, cfg) -> list[dict]:
    rows = []
    for view, sl in views:
        m = {**meta, "n_queries": len(queries[sl]) if view == "fixed" else n_q}
        rows.extend(
            build_rows(
                m,
                variant,
                view,
                {n: v[sl] for n, v in exact_pq.items()},
                kinds[sl],
                None,
                cfg.n_boot,
                cfg.seed,
            )
        )
    return rows


def show(df: pd.DataFrame) -> None:
    d = df[(df["view"] == "cumulative")]
    t = Table(title="weaviate eval (cumulative view)")
    cols = [
        "ndcg10",
        "mrr10",
        "recall1",
        "recall10",
        "recall100",
        "d_ndcg10",
        "ann_recall100",
        "p50_ms",
        "p95_ms",
        "p99_ms",
    ]
    t.add_column("variant")
    for c in cols:
        t.add_column(c, justify="right")
    piv = d.pivot_table(index="variant", columns="metric", values="value", aggfunc="first")
    for v, r in piv.iterrows():
        if "ndcg10" not in r or pd.isna(r.get("ndcg10")):
            continue
        t.add_row(
            v,
            *[
                "" if pd.isna(r.get(c)) else f"{r[c]:.4f}" if "_ms" not in c else f"{r[c]:.1f}"
                for c in cols
            ],
        )
    console.print(t)


def run(cfg: WeaviateEvalConfig | None = None) -> pd.DataFrame:
    cfg = env_override(cfg or WeaviateEvalConfig(), "EVAL_")
    if not cfg.weaviate_api_key:
        raise ValueError("set EVAL_WEAVIATE_API_KEY")
    base = Path(cfg.data_dir)
    k = cfg.iter
    models = (
        parse_models(cfg.weaviate_models) if cfg.weaviate_models else default_models(base, k, cfg)
    )
    logger.info("iteration {}: models {}", k, models)
    out = results_dir(base, k)
    out.mkdir(parents=True, exist_ok=True)

    client = connect(cfg)
    all_rows: list[dict] = []
    server = client.get_meta()
    config: dict = {
        "iter": k,
        "eval_config": {**asdict(cfg), "weaviate_api_key": "***"},
        "server": server,
        "server_version": server.get("version"),
        "collections": {},
    }
    try:
        for key, ch in models:
            rows, entry = eval_model(client, key, ch, cfg, base)
            all_rows += rows
            config["collections"][f"{key}_{ch}"] = entry
            df = pd.DataFrame(all_rows, columns=LONG_COLS)
            df.to_csv(out / "results_weaviate.csv", index=False)
            (out / f"weaviate_config_iter_{k}.json").write_text(
                json.dumps(config, indent=1, default=str)
            )
    finally:
        client.close()
    df = pd.DataFrame(all_rows, columns=LONG_COLS)
    show(df)
    logger.info("wrote {} and weaviate_config_iter_{}.json", out / "results_weaviate.csv", k)
    return df


if __name__ == "__main__":
    run()
