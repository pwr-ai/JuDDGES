"""Integration test of the v2 collections against a throwaway Weaviate 1.40.

Skipped unless WEAVIATE_TEST_URL is set (e.g. http://127.0.0.1:8119, no auth). gRPC port from
WEAVIATE_TEST_GRPC_PORT (default 50069). Start one with:

    docker run -d --rm --name wv-schema-test -p 127.0.0.1:8119:8080 -p 127.0.0.1:50069:50051 \
        -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e DEFAULT_VECTORIZER_MODULE=none \
        -e PERSISTENCE_DATA_PATH=/tmp/w -e CLUSTER_HOSTNAME=n1 -e RAFT_BOOTSTRAP_EXPECT=1 \
        cr.weaviate.io/semitechnologies/weaviate:1.40.2

The test deletes and recreates `Judgments` and `JudgmentChunks` on that instance.
"""

import os
import random
from datetime import datetime, timezone
from urllib.parse import urlparse

import pytest
import requests

import weaviate
import weaviate.classes.config as wvcc
from juddges.data.weaviate_v2.ie_properties import (
    add_ie_schema,
    build_ie_payload,
    ie_chunk_payload,
    load_ie_schema,
)
from juddges.data.weaviate_v2.schema import (
    JUDGMENT_CHUNKS_COLLECTION,
    JUDGMENTS_COLLECTION,
    PARENT_JUDGMENT_REF,
    JudgmentVector,
    SchemaConfig,
    chunk_uuid,
    create_collections,
    judgment_uuid,
)
from juddges.settings import CONFIG_PATH, VectorName
from weaviate.classes.query import Filter, MetadataQuery

pytestmark = [pytest.mark.requires_weaviate, pytest.mark.integration]

DIM = 1024
IE_DIR = CONFIG_PATH / "ie_schema"
URL = os.getenv("WEAVIATE_TEST_URL")


def _vec(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(DIM)]


@pytest.fixture(scope="module")
def base_url() -> str:
    if not URL:
        pytest.skip("WEAVIATE_TEST_URL not set")
    return URL.rstrip("/")


@pytest.fixture(scope="module")
def client(base_url):
    parsed = urlparse(base_url)
    c = weaviate.connect_to_local(
        host=parsed.hostname,
        port=parsed.port or 8080,
        grpc_port=int(os.getenv("WEAVIATE_TEST_GRPC_PORT", "50069")),
    )
    try:
        yield c
    finally:
        for name in (JUDGMENT_CHUNKS_COLLECTION, JUDGMENTS_COLLECTION):
            c.collections.delete(name)
        c.close()


@pytest.fixture(scope="module")
def ie_franc():
    return load_ie_schema(IE_DIR / "swiss_franc_loans.yaml")


@pytest.fixture(scope="module")
def created(client, ie_franc):
    for name in (JUDGMENT_CHUNKS_COLLECTION, JUDGMENTS_COLLECTION):
        client.collections.delete(name)
    create_collections(client, SchemaConfig(judgment_shards=2, chunk_shards=3), [ie_franc])
    return client


def _rest_schema(base_url: str, name: str) -> dict:
    r = requests.get(f"{base_url}/v1/schema/{name}", timeout=30)
    r.raise_for_status()
    return r.json()


def _props(schema: dict) -> dict[str, dict]:
    return {p["name"]: p for p in schema["properties"]}


def test_create_is_idempotent(created, ie_franc) -> None:
    create_collections(created, SchemaConfig(), [ie_franc])  # exists -> skipped, no error


@pytest.mark.parametrize(
    ("collection", "vectors"),
    [
        (JUDGMENTS_COLLECTION, {JudgmentVector.THESIS, JudgmentVector.SUMMARY}),
        (JUDGMENT_CHUNKS_COLLECTION, {VectorName.DEFAULT}),
    ],
)
def test_every_named_vector_is_hnsw_cosine_rq4(created, collection, vectors) -> None:
    cfg = created.collections.get(collection).config.get()
    assert set(cfg.vector_config) == vectors
    for name, vec in cfg.vector_config.items():
        idx = vec.vector_index_config
        assert vec.vectorizer.vectorizer == wvcc.Vectorizers.NONE, name
        assert idx.distance_metric == wvcc.VectorDistances.COSINE, name
        assert idx.ef_construction == 128, name
        assert idx.max_connections == 32, name
        assert idx.ef == -1, name
        assert idx.quantizer is not None, name
        assert type(idx.quantizer).__name__ == "_RQConfig", name
        assert idx.quantizer.bits == 4, name


def test_shard_counts(created, base_url) -> None:
    assert _rest_schema(base_url, JUDGMENTS_COLLECTION)["shardingConfig"]["desiredCount"] == 2
    assert _rest_schema(base_url, JUDGMENT_CHUNKS_COLLECTION)["shardingConfig"]["desiredCount"] == 3


def test_judgments_full_text_is_stored_only(created, base_url) -> None:
    props = _props(_rest_schema(base_url, JUDGMENTS_COLLECTION))
    assert props["full_text"]["indexSearchable"] is False
    assert props["full_text"]["indexFilterable"] is False
    assert "raw_content" not in props
    assert props["date_issued"]["indexRangeFilters"] is True
    assert props["court_name"]["tokenization"] == "field"
    assert props["document_number"]["tokenization"] == "word"
    assert props["extracted_legal_bases"]["dataType"] == ["object[]"]


def test_chunk_denormalized_fields_are_filterable(created, base_url) -> None:
    schema = _rest_schema(base_url, JUDGMENT_CHUNKS_COLLECTION)
    props = _props(schema)
    for name in ("judgment_id", "court_type", "judgment_type", "court_name", "document_type"):
        assert props[name]["indexFilterable"] is True, name
        assert props[name]["tokenization"] == "field", name
    assert props["language"]["indexFilterable"] is True
    assert props["date_issued"]["dataType"] == ["date"]
    assert props["date_issued"]["indexRangeFilters"] is True
    assert props["chunk_text"]["indexSearchable"] is True
    assert props["chunk_text"]["tokenization"] == "word"
    assert props["chunk_idx"]["dataType"] == ["int"]
    # IE enum copies are denormalized onto chunks, other IE kinds are not
    assert props["ie_swiss_franc_loans__typ_sadu"]["indexFilterable"] is True
    assert "ie_swiss_franc_loans__data_wyroku" not in props
    assert "ie_swiss_franc_loans" not in props
    assert {r["name"] for r in props.values() if r["dataType"] == [JUDGMENTS_COLLECTION]} == {
        PARENT_JUDGMENT_REF
    }


def test_polish_stopwords_in_both_collections(created, base_url) -> None:
    for name in (JUDGMENTS_COLLECTION, JUDGMENT_CHUNKS_COLLECTION):
        stopwords = _rest_schema(base_url, name)["invertedIndexConfig"]["stopwords"]
        assert "oraz" in stopwords["additions"]
        assert stopwords["preset"] == "en"


def test_ie_object_and_meta_present(created, base_url) -> None:
    props = _props(_rest_schema(base_url, JUDGMENTS_COLLECTION))
    obj = props["ie_swiss_franc_loans"]
    assert obj["dataType"] == ["object"]
    assert any(
        n["name"] == "data_wyroku" and n["dataType"] == ["date"] for n in obj["nestedProperties"]
    )
    assert props["ie_swiss_franc_loans_meta__extracted_at"]["dataType"] == ["date"]
    assert props["ie_swiss_franc_loans__data_wyroku"]["indexRangeFilters"] is True


def test_add_ie_schema_is_idempotent(created, base_url) -> None:
    judgments = created.collections.get(JUDGMENTS_COLLECTION)
    chunks = created.collections.get(JUDGMENT_CHUNKS_COLLECTION)
    path = IE_DIR / "personal_rights.yaml"

    first = add_ie_schema(judgments, path, chunk_collection=chunks)
    assert "ie_personal_rights" in first
    assert "ie_personal_rights__naruszenie_dobr_osobistych" in first
    assert add_ie_schema(judgments, path, chunk_collection=chunks) == []

    judgment_props = _props(_rest_schema(base_url, JUDGMENTS_COLLECTION))
    chunk_props = _props(_rest_schema(base_url, JUDGMENT_CHUNKS_COLLECTION))
    assert judgment_props["ie_personal_rights__skala_naruszenia"]["dataType"] == ["int"]
    assert "ie_personal_rights__naruszenie_dobr_osobistych" in chunk_props
    assert "ie_personal_rights__skala_naruszenia" not in chunk_props


def test_insert_filter_hybrid(created, ie_franc) -> None:
    judgments = created.collections.get(JUDGMENTS_COLLECTION)
    chunks = created.collections.get(JUDGMENT_CHUNKS_COLLECTION)

    ie_result = {
        "typ_sadu": None,
        "data_wyroku": "2021-03-05",
        "data_aneksu": "2020-01-15",
        "opis": "ignored",
    }
    ie_payload = build_ie_payload(
        ie_franc, ie_result, schema_version="1", model="m", extracted_at="2025-01-02"
    )
    assert "ie_swiss_franc_loans__data_wyroku" in ie_payload

    docs = [
        ("J1", "2021-03-05", "Sąd Okręgowy w Warszawie", "kredyt frankowy nieważność umowy"),
        ("J2", "2015-06-01", "Sąd Apelacyjny w Krakowie", "wypadek drogowy odszkodowanie"),
    ]
    for i, (jid, issued, court, text) in enumerate(docs):
        props = {
            "judgment_id": jid,
            "document_type": "judgment",
            "language": "pl",
            "court_type": "ordinary court",
            "judgment_type": "wyrok",
            "court_name": court,
            "date_issued": issued + "T00:00:00Z",
            "full_text": text,
            "extracted_legal_bases": [{"art": "385", "isap_id": "WDU1", "title": "KC"}],
            "parties": '[{"role": "powód"}]',
        }
        if jid == "J1":
            props.update(ie_payload)
        judgments.data.insert(
            properties=props,
            uuid=judgment_uuid(jid),
            vector={JudgmentVector.THESIS: _vec(100 + i)},  # summary left out on purpose
        )
        chunk_props = {
            "chunk_text": text,
            "judgment_id": jid,
            "chunk_idx": 0,
            **{k: props[k] for k in ("document_type", "language", "court_type", "judgment_type")},
            "court_name": court,
            "date_issued": props["date_issued"],
            **ie_chunk_payload(ie_franc, ie_payload),
        }
        chunks.data.insert(
            properties=chunk_props,
            uuid=chunk_uuid(jid, 0),
            vector={VectorName.DEFAULT: _vec(i)},
            references={PARENT_JUDGMENT_REF: judgment_uuid(jid)},
        )

    # flat IE filters: date range and meta on judgments
    res = judgments.query.fetch_objects(
        filters=Filter.by_property("ie_swiss_franc_loans__data_wyroku").greater_or_equal(
            datetime(2021, 1, 1, tzinfo=timezone.utc)
        )
        & Filter.by_property("ie_swiss_franc_loans_meta__model").equal("m"),
        return_properties=["judgment_id", "ie_swiss_franc_loans"],
    )
    assert [o.properties["judgment_id"] for o in res.objects] == ["J1"]
    stored = res.objects[0].properties["ie_swiss_franc_loans"]
    assert stored["data_aneksu"].date().isoformat() == "2020-01-15"

    # nested paths are NOT filterable on this Weaviate version (why flat copies exist)
    with pytest.raises(Exception):
        judgments.query.fetch_objects(
            filters=Filter.by_property("ie_swiss_franc_loans.data_wyroku").is_none(False)
        )

    # structured legal bases round-trip
    got = judgments.query.fetch_object_by_id(
        judgment_uuid("J1"), return_properties=["extracted_legal_bases"]
    )
    assert got.properties["extracted_legal_bases"][0]["isap_id"] == "WDU1"

    # judgment vectors are optional: J1 has no summary vector
    got = judgments.query.fetch_object_by_id(judgment_uuid("J1"), include_vector=True)
    assert set(got.vector) == {JudgmentVector.THESIS}

    # filter on denormalized chunk fields
    res = chunks.query.fetch_objects(
        filters=Filter.by_property("court_name").equal("Sąd Apelacyjny w Krakowie")
        & Filter.by_property("date_issued").less_than(datetime(2020, 1, 1, tzinfo=timezone.utc))
    )
    assert [o.properties["judgment_id"] for o in res.objects] == ["J2"]

    # hybrid on chunks: BM25 on chunk_text + dense, filtered by a denormalized field
    res = chunks.query.hybrid(
        query="kredyt frankowy",
        vector=_vec(0),
        alpha=0.25,
        target_vector=VectorName.DEFAULT,
        query_properties=["chunk_text"],
        filters=Filter.by_property("language").equal("pl"),
        limit=5,
        return_metadata=MetadataQuery(score=True),
    )
    assert res.objects
    assert res.objects[0].properties["judgment_id"] == "J1"

    # cross-reference to the parent judgment
    res = chunks.query.fetch_objects(
        filters=Filter.by_ref(PARENT_JUDGMENT_REF).by_property("court_name").equal(docs[0][2])
    )
    assert [o.properties["judgment_id"] for o in res.objects] == ["J1"]
