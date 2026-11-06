"""Offline tests of the v2 collections: property lists and `create_collections` on a fake client."""

import re
from types import SimpleNamespace

import pytest

import weaviate.classes.config as wvcc
from juddges.data.weaviate_v2.ie_properties import (
    ie_chunk_properties,
    ie_properties,
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
    judgment_chunks_properties,
    judgment_uuid,
    judgments_properties,
)
from juddges.settings import CONFIG_PATH, VectorName


@pytest.mark.parametrize("make", [judgments_properties, judgment_chunks_properties])
def test_property_names_are_unique_and_valid(make) -> None:
    names = [p.name for p in make()]
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[_A-Za-z][_0-9A-Za-z]*", n) for n in names)


def test_no_dead_legacy_fields() -> None:
    names = {p.name for p in judgments_properties()}
    assert names.isdisjoint({"raw_content", "issuing_body", "processing_status", "last_updated"})


def test_uuids_are_deterministic_and_distinct() -> None:
    assert judgment_uuid("J1") == judgment_uuid("J1")
    assert judgment_uuid("J1") != judgment_uuid("J2")
    assert chunk_uuid("J1", 0) == chunk_uuid("J1", 0)
    assert len({chunk_uuid("J1", 0), chunk_uuid("J1", 1), chunk_uuid("J2", 0)}) == 3
    assert chunk_uuid("J1", 0) != judgment_uuid("J1")
    assert re.fullmatch(r"[0-9a-f-]{36}", judgment_uuid("J1"))


class _FakeClient:
    """Records `collections.create` calls; `existing` maps name -> property configs."""

    def __init__(self, existing: dict | None = None) -> None:
        self.created: dict[str, dict] = {}
        self.existing = existing or {}
        self.added: dict[str, list[str]] = {}
        self.collections = self

    def exists(self, name: str) -> bool:
        return name in self.existing or name in self.created

    def create(self, **kwargs) -> None:
        self.created[kwargs["name"]] = kwargs

    def get(self, name: str):
        props = self.existing[name]
        added = self.added.setdefault(name, [])
        return SimpleNamespace(
            name=name,
            config=SimpleNamespace(
                get=lambda: SimpleNamespace(properties=props),
                add_property=lambda p: added.append(p.name),
            ),
        )


def _vectors(kwargs: dict) -> dict[str, dict]:
    return {v.name: v._to_dict() for v in kwargs["vector_config"]}


def test_create_collections_arguments() -> None:
    client = _FakeClient()
    create_collections(client, SchemaConfig(judgment_shards=2, chunk_shards=5))
    assert list(client.created) == [JUDGMENTS_COLLECTION, JUDGMENT_CHUNKS_COLLECTION]
    judgments, chunks = (
        client.created[JUDGMENTS_COLLECTION],
        client.created[JUDGMENT_CHUNKS_COLLECTION],
    )

    for kwargs, names in (
        (judgments, {JudgmentVector.THESIS, JudgmentVector.SUMMARY}),
        (chunks, {VectorName.DEFAULT}),
    ):
        vectors = _vectors(kwargs)
        assert set(vectors) == names
        for vec in vectors.values():
            assert vec["vectorizer"] == {"none": {}}  # self-provided
            assert vec["vectorIndexType"] == "hnsw"
            cfg = vec["vectorIndexConfig"]
            assert (cfg["efConstruction"], cfg["maxConnections"], cfg["ef"]) == (128, 32, -1)
            assert cfg["rq"] == {"enabled": True, "bits": 4}
            assert "bq" not in cfg and "pq" not in cfg

    assert judgments["sharding_config"].desiredCount == 2
    assert chunks["sharding_config"].desiredCount == 5
    for kwargs in (judgments, chunks):
        inverted = kwargs["inverted_index_config"]
        assert "oraz" in inverted.stopwords.additions
        assert (inverted.bm25.b, inverted.bm25.k1) == (0.75, 1.2)

    [ref] = chunks["references"]
    assert (ref.name, ref.target_collection) == (PARENT_JUDGMENT_REF, JUDGMENTS_COLLECTION)
    assert "references" not in judgments


def test_create_collections_with_ie_schema_adds_properties_to_both() -> None:
    ie = load_ie_schema(CONFIG_PATH / "ie_schema" / "personal_rights.yaml")
    client = _FakeClient()
    create_collections(client, ie_schemas=[ie])
    judgment_names = {p.name for p in client.created[JUDGMENTS_COLLECTION]["properties"]}
    chunk_names = {p.name for p in client.created[JUDGMENT_CHUNKS_COLLECTION]["properties"]}
    assert {p.name for p in ie_properties(ie)} <= judgment_names
    assert {p.name for p in ie_chunk_properties(ie)} <= chunk_names
    assert "ie_personal_rights" not in chunk_names


def test_create_collections_on_existing_adds_missing_properties_only() -> None:
    ie = load_ie_schema(CONFIG_PATH / "ie_schema" / "personal_rights.yaml")

    def as_existing(props):
        return [
            SimpleNamespace(
                name=p.name,
                data_type=p.dataType,
                index_filterable=p.indexFilterable,
                index_searchable=p.indexSearchable,
                index_range_filters=p.indexRangeFilters,
                tokenization=p.tokenization,
                nested_properties=[
                    SimpleNamespace(name=n.name, data_type=n.dataType)
                    for n in p.nestedProperties or ()
                ],
            )
            for p in props
        ]

    client = _FakeClient(
        {
            JUDGMENTS_COLLECTION: as_existing(judgments_properties()),
            JUDGMENT_CHUNKS_COLLECTION: as_existing(judgment_chunks_properties()),
        }
    )
    create_collections(client, ie_schemas=[ie])
    assert client.created == {}  # nothing re-created
    assert set(client.added[JUDGMENTS_COLLECTION]) == {p.name for p in ie_properties(ie)}
    assert set(client.added[JUDGMENT_CHUNKS_COLLECTION]) == {
        p.name for p in ie_chunk_properties(ie)
    }
    # second run: everything exists now
    client.existing[JUDGMENTS_COLLECTION] += as_existing(ie_properties(ie))
    client.existing[JUDGMENT_CHUNKS_COLLECTION] += as_existing(ie_chunk_properties(ie))
    client.added.clear()
    create_collections(client, ie_schemas=[ie])
    assert all(not v for v in client.added.values())


def test_create_collections_rejects_colliding_schemas_before_creating() -> None:
    a = load_ie_schema(CONFIG_PATH / "ie_schema" / "personal_rights.yaml")
    client = _FakeClient()
    with pytest.raises(ValueError, match="given twice"):
        create_collections(client, ie_schemas=[a, a])
    assert client.created == {}


def test_hnsw_rq_helper_keeps_rq_bits_configurable() -> None:
    client = _FakeClient()
    create_collections(client, SchemaConfig(rq_bits=8))
    vec = next(iter(_vectors(client.created[JUDGMENT_CHUNKS_COLLECTION]).values()))
    assert vec["vectorIndexConfig"]["rq"]["bits"] == 8
    assert wvcc.Configure.VectorIndex.Quantizer.rq(bits=8) is not None
