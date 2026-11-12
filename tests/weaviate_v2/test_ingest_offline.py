"""Offline tests of the v2 ingest mapping, MaxP aggregation and the TEI model guard."""

import json
from datetime import datetime, timezone

import httpx
import pytest

from juddges.data.weaviate_v2.ingest import (
    EMBEDDING_MODEL,
    TeiEmbedder,
    chunk_properties,
    judgment_properties,
)
from juddges.data.weaviate_v2.schema import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    judgment_chunks_properties,
    judgments_properties,
)
from juddges.data.weaviate_v2.search import ChunkHit, maxp

NOW = datetime(2026, 11, 12, tzinfo=timezone.utc)
ROW = {
    "judgment_id": "155000000000503_I_ACa_000123_2021_Uz_2022-03-01_001",
    "docket_number": "I ACa 123/21",
    "judgment_date": datetime(2022, 3, 1),
    "publication_date": None,
    "court_name": "Sąd Apelacyjny w Warszawie",
    "court_type": "appeal",
    "judgment_type": "SENTENCE",
    "full_text": "Sentencja ... Uzasadnienie ...",
    "thesis": "Klauzula przeliczeniowa jest abuzywna.",
    "keywords": ["kredyt", None, ""],
    "judges": [],
    "extracted_legal_bases": [{"art": "385(1)", "title": "Kodeks cywilny", "extra": "x"}],
}


def test_judgment_properties_use_schema_names_only():
    props = judgment_properties(ROW, NOW)
    assert set(props) <= {p.name for p in judgments_properties()}
    assert props["document_number"] == "I ACa 123/21"
    assert props["source_id"] == ROW["judgment_id"]
    assert props["document_type"] == "judgment"


def test_judgment_dates_are_timezone_aware_and_missing_ones_dropped():
    props = judgment_properties(ROW, NOW)
    assert props["date_issued"] == datetime(2022, 3, 1, tzinfo=timezone.utc)
    assert "publication_date" not in props


def test_lists_are_cleaned_and_empty_lists_dropped():
    props = judgment_properties(ROW, NOW)
    assert props["keywords"] == ["kredyt"]
    assert "judges" not in props
    assert props["extracted_legal_bases"] == [{"art": "385(1)", "title": "Kodeks cywilny"}]


def test_chunk_properties_copy_filter_fields():
    judgment = judgment_properties(ROW, NOW)
    props = chunk_properties(judgment, "tekst", 3)
    assert set(props) <= {p.name for p in judgment_chunks_properties()}
    assert props["chunk_idx"] == 3
    assert props["court_name"] == judgment["court_name"]
    assert props["date_issued"] == judgment["date_issued"]


def test_maxp_ranks_judgments_by_best_chunk():
    chunks = [
        ChunkHit("a", 0, "a0", 0.4),
        ChunkHit("b", 0, "b0", 0.9),
        ChunkHit("a", 1, "a1", 0.95),
        ChunkHit("c", 0, "c0", 0.1),
    ]
    hits = maxp(chunks, k=2)
    assert [h.judgment_id for h in hits] == ["a", "b"]
    assert hits[0].best_chunk.chunk_idx == 1


def _tei(model: str, seen: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/info":
            return httpx.Response(200, json={"model_id": model})
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json=[[0.0] * 1024 for _ in body["inputs"]])

    return httpx.MockTransport(handler)


def test_tei_embedder_refuses_another_model():
    with pytest.raises(RuntimeError, match="expected"):
        TeiEmbedder("http://tei", transport=_tei("BAAI/bge-m3", []))


def test_tei_embedder_adds_prefixes_and_batches():
    seen: list[dict] = []
    tei = TeiEmbedder("http://tei", batch_size=2, transport=_tei(EMBEDDING_MODEL, seen))
    assert len(tei.passages(["x", "y", "z"])) == 3
    tei.query("q")
    assert [len(b["inputs"]) for b in seen] == [2, 1, 1]
    assert seen[0]["inputs"][0] == PASSAGE_PREFIX + "x"
    assert seen[-1]["inputs"] == [QUERY_PREFIX + "q"]
