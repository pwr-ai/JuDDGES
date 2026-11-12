"""Judgment search over `JudgmentChunks`: hybrid chunk retrieval, then MaxP to judgments.

Defaults follow the embedding evaluation: hybrid with `alpha=HYBRID_ALPHA` (0 = BM25 only,
1 = vector only) over `chunk_text`, and each judgment scored by its best chunk.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import weaviate
from juddges.data.weaviate_v2.schema import HYBRID_ALPHA, JUDGMENT_CHUNKS_COLLECTION
from juddges.settings import VectorName
from weaviate.classes.query import HybridFusion, MetadataQuery


@dataclass(frozen=True)
class ChunkHit:
    judgment_id: str
    chunk_idx: int
    chunk_text: str
    score: float


@dataclass(frozen=True)
class JudgmentHit:
    judgment_id: str
    score: float
    best_chunk: ChunkHit


def maxp(chunks: Iterable[ChunkHit], k: int) -> list[JudgmentHit]:
    """Judgments ranked by their best chunk; ties keep the chunk order of the input."""
    best: dict[str, ChunkHit] = {}
    for hit in chunks:
        current = best.get(hit.judgment_id)
        if current is None or hit.score > current.score:
            best[hit.judgment_id] = hit
    ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)
    return [JudgmentHit(h.judgment_id, h.score, h) for h in ranked[:k]]


def search_chunks(
    client: weaviate.WeaviateClient,
    query: str,
    query_vector: Sequence[float],
    *,
    alpha: float = HYBRID_ALPHA,
    chunk_limit: int = 100,
    filters: Any = None,
) -> list[ChunkHit]:
    """Top chunks for a query; `query_vector` must be the `"query: "`-prefixed embedding."""
    chunks = client.collections.get(JUDGMENT_CHUNKS_COLLECTION)
    response = chunks.query.hybrid(
        query=query,
        vector=list(query_vector),
        alpha=alpha,
        query_properties=["chunk_text"],
        target_vector=VectorName.DEFAULT,
        fusion_type=HybridFusion.RELATIVE_SCORE,
        limit=chunk_limit,
        filters=filters,
        return_properties=["judgment_id", "chunk_idx", "chunk_text"],
        return_metadata=MetadataQuery(score=True),
    )
    return [
        ChunkHit(
            judgment_id=o.properties["judgment_id"],
            chunk_idx=int(o.properties["chunk_idx"]),
            chunk_text=o.properties["chunk_text"],
            score=float(o.metadata.score or 0.0),
        )
        for o in response.objects
    ]


def search_judgments(
    client: weaviate.WeaviateClient,
    query: str,
    query_vector: Sequence[float],
    *,
    k: int = 10,
    alpha: float = HYBRID_ALPHA,
    chunk_limit: int = 100,
    filters: Any = None,
) -> list[JudgmentHit]:
    """Top `k` judgments for a query (hybrid chunk search + MaxP)."""
    hits = search_chunks(
        client, query, query_vector, alpha=alpha, chunk_limit=chunk_limit, filters=filters
    )
    return maxp(hits, k)
