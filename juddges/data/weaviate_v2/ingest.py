"""Turn `JuDDGES/pl-court-raw` and `JuDDGES/pl-nsa` rows into v2 objects: chunks, properties
and TEI vectors.

Chunking matches the embedding evaluation (`scripts/embedding_eval/build_dataset.py`, config
`c512`): a `RecursiveCharacterTextSplitter` over the BGE-M3 tokenizer with `CHUNK_TOKENS` /
`CHUNK_OVERLAP_TOKENS`. Passages and queries get the mmlw-e5 prefixes; TEI adds none itself.
"""

import re
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlparse

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


NSA_URL = "https://orzeczenia.nsa.gov.pl"
# end of the leading "TEZY" section: the next section header after a blank-line gap
_NSA_AFTER_THESIS = re.compile(r"\n\n\n(?=SENTENCJA|UZASADNIENIE|ZDANIE ODR)")


def _local_day_as_utc(value: Any) -> datetime | None:
    """Calendar day of an aware timestamp as UTC midnight.

    NSA dates are midnight Europe/Warsaw; converting them to UTC would move them to the previous
    day, while pl-court-raw days are stored as UTC midnight. Keeping the local day makes date
    filters agree across both sources.
    """
    if not isinstance(value, datetime):
        return None
    return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)


def _isap_id(link: str | None) -> str | None:
    if not link:
        return None
    return (parse_qs(urlparse(link).query).get("id") or [None])[0]


def _nsa_legal_basis(basis: Mapping[str, Any]) -> dict[str, str]:
    text = " ".join(x for x in (basis.get("journal"), basis.get("article")) if x)
    mapped = {
        "address": basis.get("link"),
        "art": basis.get("article"),
        "isap_id": _isap_id(basis.get("link")),
        "text": text or None,
        "title": basis.get("law"),
    }
    return {k: str(v) for k, v in mapped.items() if v}


def nsa_judgment_properties(row: Mapping[str, Any], ingested_at: datetime) -> dict[str, Any]:
    """`Judgments` properties of a `JuDDGES/pl-nsa` row (CBOSA: NSA and the WSA courts).

    - `court_type` is normalized to `administrative_court` (pl-court-raw: `ordinary_court`).
    - `judgment_type` keeps only the kind of decision (`Wyrok`, `Postanowienie`, `Uchwała`);
      the court part of e.g. "Wyrok WSA w Opolu" is already in `court_name`.
    - `submission_date` is the filing date, not a publication date, so `publication_date` stays
      empty. `decision` (list) becomes `outcome`.
    - `finality`, `judge_rapporteur`, `case_type_description`, `challenged_authority`,
      `related_docket_numbers`, `official_collection`, `glosa_information` and the per-section
      texts have no `Judgments` property and are not stored.
    """
    bases = [_nsa_legal_basis(b) for b in (row.get("extracted_legal_bases") or [])]
    bases = [b for b in bases if b]
    judgment_type = (row.get("judgment_type") or "").split(" ", 1)[0] or None
    props: dict[str, Any] = {
        "judgment_id": row.get("judgment_id"),
        "document_number": row.get("docket_number"),
        "source": "nsa",
        "source_id": row.get("judgment_id"),
        "source_url": NSA_URL + row["judgment_id"] if row.get("judgment_id") else None,
        "document_type": "judgment",
        "language": "pl",
        "country": row.get("country") or "Poland",
        "court_type": "administrative_court",
        "judgment_type": judgment_type,
        "outcome": "; ".join(d for d in (row.get("decision") or []) if d) or None,
        "court_name": row.get("court_name"),
        "date_issued": _local_day_as_utc(row.get("judgment_date")),
        "ingestion_date": _as_utc(ingested_at),
        "full_text": row.get("full_text"),
        "thesis": row.get("thesis"),
        "keywords": row.get("keywords"),
        "presiding_judge": row.get("presiding_judge"),
        "judges": row.get("judges"),
        "legal_bases": [b["text"] for b in bases if b.get("text")],
        "extracted_legal_bases": bases or None,
    }
    for name in ("keywords", "judges"):
        if props[name] is not None:
            props[name] = [str(x) for x in props[name] if x]
    return _clean(props, _JUDGMENT_NAMES)


def nsa_chunk_text(row: Mapping[str, Any]) -> str:
    """Text to chunk for an NSA judgment: `full_text` without its leading "TEZY" section.

    `full_text` is "TEZY" (thesis, ~2% of rows) + "SENTENCJA" + "UZASADNIENIE" + "ZDANIE
    ODRĘBNE", and unlike the `sentence` + `reasons_for_judgment` columns it also covers rows
    whose sections were not split out. The thesis is dropped because it is stored and embedded
    separately (`thesis` vector); leaving it in a chunk would let thesis queries match it
    verbatim. A document that is only a thesis keeps it, so it is still searchable.
    """
    text = row.get("full_text") or ""
    if text.startswith("TEZY"):
        parts = _NSA_AFTER_THESIS.split(text, maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1]
    return text


def court_chunk_text(row: Mapping[str, Any]) -> str:
    """Text to chunk for a pl-court-raw judgment (its thesis is a separate column)."""
    return row.get("full_text") or ""


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
