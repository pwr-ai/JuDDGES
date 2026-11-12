"""Turn `JuDDGES/pl-court-raw` rows into v2 objects: chunks, properties and TEI vectors.

Chunking matches the embedding evaluation (`scripts/embedding_eval/build_dataset.py`, config
`c512`): a `RecursiveCharacterTextSplitter` over the BGE-M3 tokenizer with `CHUNK_TOKENS` /
`CHUNK_OVERLAP_TOKENS`. Passages and queries get the mmlw-e5 prefixes; TEI adds none itself.
"""

from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import httpx
from loguru import logger

from juddges.data.weaviate_v2.schema import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TOKENS,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    judgment_chunks_properties,
    judgments_properties,
)

CHUNK_TOKENIZER = "BAAI/bge-m3"
EMBEDDING_MODEL = "sdadas/mmlw-retrieval-e5-large"
EMBEDDING_DIM = 1024

# pl-court-raw column -> Judgments property (same name when absent)
_RENAMED = {
    "docket_number": "document_number",
    "judgment_date": "date_issued",
}
_COPIED = (
    "judgment_id",
    "court_type",
    "judgment_type",
    "court_id",
    "department_id",
    "court_name",
    "department_name",
    "publication_date",
    "full_text",
    "thesis",
    "keywords",
    "presiding_judge",
    "judges",
    "legal_bases",
    "references",
    "country",
)
_LEGAL_BASIS_KEYS = ("address", "art", "isap_id", "text", "title")
# Judgment fields copied onto each chunk so filtered queries need no join
CHUNK_FILTER_FIELDS = (
    "document_type",
    "language",
    "court_type",
    "judgment_type",
    "court_name",
    "date_issued",
)


def make_splitter(tokenizer: Any = None):
    """Splitter identical to the eval's `build_chunks` (480 tokens, 64 overlap, BGE-M3 tokens)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(CHUNK_TOKENIZER)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=CHUNK_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS
    )


def split_judgment(splitter: Any, text: str | None) -> list[str]:
    """Chunks of a judgment's full text, empty pieces dropped."""
    return [p for p in splitter.split_text(text or "") if p.strip()]


def _as_utc(value: Any) -> datetime | None:
    """Weaviate `date` needs a timezone; pl-court-raw timestamps are naive (Polish dates, no time)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return None


def _clean(props: Mapping[str, Any], allowed: set[str]) -> dict[str, Any]:
    """Drop unknown names, None and empty lists (Weaviate stores absent as null anyway)."""
    return {
        k: v for k, v in props.items() if k in allowed and v is not None and v != [] and v != ""
    }


_JUDGMENT_NAMES = {p.name for p in judgments_properties()}
_CHUNK_NAMES = {p.name for p in judgment_chunks_properties()}


def judgment_properties(row: Mapping[str, Any], ingested_at: datetime) -> dict[str, Any]:
    """`Judgments` properties of a pl-court-raw row."""
    props: dict[str, Any] = {name: row.get(name) for name in _COPIED}
    for column, name in _RENAMED.items():
        props[name] = row.get(column)
    props.update(
        source="pl-court",
        source_id=row.get("judgment_id"),
        document_type="judgment",
        language="pl",
        date_issued=_as_utc(props["date_issued"]),
        publication_date=_as_utc(props["publication_date"]),
        ingestion_date=_as_utc(ingested_at),
        extracted_legal_bases=[
            {k: str(b[k]) for k in _LEGAL_BASIS_KEYS if b.get(k) is not None}
            for b in (row.get("extracted_legal_bases") or [])
        ]
        or None,
    )
    for name in ("keywords", "judges", "legal_bases", "references"):
        if props[name] is not None:
            props[name] = [str(x) for x in props[name] if x]
    return _clean(props, _JUDGMENT_NAMES)


def chunk_properties(
    judgment: Mapping[str, Any], chunk_text: str, chunk_idx: int
) -> dict[str, Any]:
    """`JudgmentChunks` properties: text, position and the judgment's filter fields."""
    props = {f: judgment.get(f) for f in CHUNK_FILTER_FIELDS}
    props.update(chunk_text=chunk_text, judgment_id=judgment["judgment_id"], chunk_idx=chunk_idx)
    return _clean(props, _CHUNK_NAMES)


def batched(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class TeiEmbedder:
    """mmlw-e5 vectors from a TEI server. Refuses to run against another model."""

    def __init__(
        self,
        url: str,
        batch_size: int = 64,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.client = httpx.Client(base_url=url.rstrip("/"), timeout=timeout, transport=transport)
        self.batch_size = batch_size
        model = self.client.get("/info").raise_for_status().json().get("model_id", "")
        if model != EMBEDDING_MODEL:
            raise RuntimeError(f"TEI at {url} serves {model!r}, expected {EMBEDDING_MODEL!r}")

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in batched(texts, self.batch_size):
            resp = self.client.post("/embed", json={"inputs": list(batch), "truncate": True})
            vectors.extend(resp.raise_for_status().json())
        if any(len(v) != EMBEDDING_DIM for v in vectors):
            raise RuntimeError(f"TEI returned vectors that are not {EMBEDDING_DIM}-dimensional")
        return vectors

    def passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Chunk, thesis and summary vectors (`"passage: "` prefix)."""
        return self._embed([PASSAGE_PREFIX + t for t in texts])

    def query(self, text: str) -> list[float]:
        """Query vector (`"query: "` prefix)."""
        return self._embed([QUERY_PREFIX + text])[0]

    def close(self) -> None:
        self.client.close()
        logger.debug("TEI client closed")
