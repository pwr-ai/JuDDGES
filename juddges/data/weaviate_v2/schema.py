"""Collections of the Weaviate v2 stack (Weaviate 1.40, vectors computed outside by TEI).

Retrieval unit is a chunk: `JudgmentChunks` carries BM25 (`chunk_text`) and the dense vector,
results are aggregated to the judgment by max chunk score (MaxP). `Judgments` holds metadata, the
full text (stored only) and two judgment-level vectors (`thesis`, `summary`).

Choices are measured, see docs/explanation/embedding-and-quantization.md:
- sdadas/mmlw-retrieval-e5-large, 1024d; clients prefix queries with "query: ", chunks with
  "passage: " (QUERY_PREFIX / PASSAGE_PREFIX)
- HNSW cosine, ef_construction=128, max_connections=32, ef=-1 (dynamic), RQ 4-bit
- hybrid search on chunks with alpha ~0.25 (BM25 alone beats dense by +0.16 nDCG)

Full reference: docs/reference/weaviate-v2-schema.md
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from loguru import logger

import weaviate
import weaviate.classes.config as wvcc
from juddges.data.documents_weaviate_db import POLISH_STOPWORDS
from juddges.data.weaviate_v2.ie_properties import IeSchema, ie_chunk_properties, ie_properties
from juddges.settings import VectorName

JUDGMENTS_COLLECTION = "Judgments"
JUDGMENT_CHUNKS_COLLECTION = "JudgmentChunks"
PARENT_JUDGMENT_REF = "parent_judgment"

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Chunking the vectors were evaluated with (BGE-M3 tokenizer, model context is 512 tokens)
CHUNK_TOKENS = 480
CHUNK_OVERLAP_TOKENS = 64

HYBRID_ALPHA = 0.25  # weight of the dense part; 0 = BM25 only


class JudgmentVector:
    """Named vectors of the Judgments collection (chunks use `VectorName.DEFAULT`)."""

    THESIS = "thesis"
    SUMMARY = "summary"


@dataclass(frozen=True)
class SchemaConfig:
    """Tunables of the v2 collections.

    Shards: Judgments ~3M objects -> 4 (as the legacy stack). JudgmentChunks ~54M objects -> 8:
    ~7M chunks per shard keeps each HNSW graph / LSM store at a size that builds, compacts and
    restarts in reasonable time, and queries fan out over shards in parallel. All shards live on
    one node; changing the count later needs a full re-ingest, so set it before ingesting.
    """

    judgment_shards: int = 4
    chunk_shards: int = 8
    ef_construction: int = 128
    max_connections: int = 32
    ef: int = -1  # dynamic: ef = clamp(limit * factor, min, max)
    rq_bits: int = 4
    bm25_b: float = 0.75
    bm25_k1: float = 1.2
    cleanup_interval_seconds: int = 60


def _hnsw(config: SchemaConfig) -> wvcc.VectorIndexConfigHNSWCreate:
    return wvcc.Configure.VectorIndex.hnsw(
        distance_metric=wvcc.VectorDistances.COSINE,
        ef_construction=config.ef_construction,
        max_connections=config.max_connections,
        ef=config.ef,
        quantizer=wvcc.Configure.VectorIndex.Quantizer.rq(bits=config.rq_bits),
    )


def _vector(name: str, config: SchemaConfig) -> wvcc.VectorConfigCreate:
    return wvcc.Configure.Vectors.self_provided(name=name, vector_index_config=_hnsw(config))


def _inverted_index(config: SchemaConfig) -> wvcc.InvertedIndexConfigCreate:
    return wvcc.Configure.inverted_index(
        bm25_b=config.bm25_b,
        bm25_k1=config.bm25_k1,
        cleanup_interval_seconds=config.cleanup_interval_seconds,
        stopwords_preset=wvcc.StopwordsPreset.EN,
        stopwords_additions=POLISH_STOPWORDS,
    )


# --- property helpers --------------------------------------------------------------------------


def _enum(name: str, description: str) -> wvcc.Property:
    """Exact-match, filterable categorical text (FIELD tokenization, no BM25)."""
    return wvcc.Property(
        name=name,
        data_type=wvcc.DataType.TEXT,
        description=description,
        index_filterable=True,
        index_searchable=False,
        tokenization=wvcc.Tokenization.FIELD,
    )


def _date(name: str, description: str) -> wvcc.Property:
    return wvcc.Property(
        name=name,
        data_type=wvcc.DataType.DATE,
        description=description,
        index_filterable=True,
        index_range_filters=True,
    )


def _stored(name: str, data_type: wvcc.DataType, description: str) -> wvcc.Property:
    """Stored and returned, but without any inverted index."""
    return wvcc.Property(
        name=name,
        data_type=data_type,
        description=description,
        index_filterable=False,
        index_searchable=False,
    )


def _searchable(name: str, description: str, *, filterable: bool = False) -> wvcc.Property:
    return wvcc.Property(
        name=name,
        data_type=wvcc.DataType.TEXT,
        description=description,
        index_filterable=filterable,
        index_searchable=True,
        tokenization=wvcc.Tokenization.WORD,
    )


def _word_array(name: str, description: str, *, searchable: bool = False) -> wvcc.Property:
    return wvcc.Property(
        name=name,
        data_type=wvcc.DataType.TEXT_ARRAY,
        description=description,
        index_filterable=True,
        index_searchable=searchable,
        tokenization=wvcc.Tokenization.WORD,
    )


def judgments_properties() -> list[wvcc.Property]:
    """Static properties of `Judgments` (IE properties are added by the caller)."""
    T, A = wvcc.DataType.TEXT, wvcc.DataType.TEXT_ARRAY
    return [
        # identifiers
        _enum("judgment_id", "Unique identifier of the judgment (UUID5 is derived from it)"),
        wvcc.Property(
            name="document_number",
            data_type=T,
            description="Case signature (docket number), searchable by tokens",
            index_filterable=True,
            index_searchable=True,
            tokenization=wvcc.Tokenization.WORD,
        ),
        _enum("source_id", "Identifier in the source system"),
        _enum("source", "Data source, e.g. pl-court, nsa"),
        _stored("source_url", T, "Source URL"),
        # categorical metadata
        _enum("document_type", "judgment, tax_interpretation, legal_act"),
        _enum("language", "Language code (pl, en)"),
        _enum("country", "Country of origin"),
        _enum("court_type", "Type of court"),
        _enum("judgment_type", "Type of judgment"),
        _enum("outcome", "Outcome of the case"),
        _enum("court_id", "System id of the court"),
        _enum("department_id", "System id of the court department"),
        _enum("court_name", "Name of the court (exact match)"),
        _enum("department_name", "Name of the court department (exact match)"),
        # dates
        _date("date_issued", "Date of the judgment"),
        _date("publication_date", "Date the judgment was published"),
        _date("ingestion_date", "When the judgment was ingested"),
        # text
        _stored("full_text", T, "Full text; BM25 and vectors live on JudgmentChunks"),
        _searchable("title", "Title"),
        _searchable("summary", "Summary (also embedded as the `summary` vector)"),
        _searchable("thesis", "Legal thesis (also embedded as the `thesis` vector)"),
        _word_array("keywords", "Keywords / topics", searchable=True),
        _searchable("factual_state", "Factual circumstances (LLM-extracted)"),
        _searchable("legal_state", "Legal state (LLM-extracted)"),
        # people and legal content
        wvcc.Property(
            name="presiding_judge",
            data_type=T,
            description="Presiding judge",
            index_filterable=True,
            index_searchable=False,
            tokenization=wvcc.Tokenization.WORD,
        ),
        _word_array("judges", "Judges of the panel"),
        _word_array("legal_bases", "Legal bases (plain text)"),
        _word_array("references", "Plain-text references to legal acts"),
        wvcc.Property(
            name="extracted_legal_bases",
            data_type=wvcc.DataType.OBJECT_ARRAY,
            description="Legal bases linked to the ISAP repository",
            nested_properties=[
                _stored(n, T, d)
                for n, d in (
                    ("address", "URL in the legal acts repository"),
                    ("art", "Article"),
                    ("isap_id", "ISAP identifier of the act"),
                    ("text", "Text as found in the judgment"),
                    ("title", "Title of the act"),
                )
            ],
        ),
        # LLM-extracted, free-form JSON (shape is not fixed by the extractor) kept as strings
        _stored("parties", T, "Parties with roles (JSON string)"),
        _stored("legal_concepts", A, "Legal concepts (LLM-extracted)"),
        _stored("legal_analysis", T, "Structured legal reasoning (JSON string)"),
        _stored("judgment_specific", T, "Judgment-specific fields (JSON string)"),
        # visualisation
        _stored("x", wvcc.DataType.NUMBER, "X coordinate for visualization"),
        _stored("y", wvcc.DataType.NUMBER, "Y coordinate for visualization"),
    ]


def judgment_chunks_properties() -> list[wvcc.Property]:
    """Static properties of `JudgmentChunks`: text, position and denormalized filter fields."""
    return [
        wvcc.Property(
            name="chunk_text",
            data_type=wvcc.DataType.TEXT,
            description="Chunk text (BM25; the vector is computed from 'passage: ' + text)",
            index_filterable=False,
            index_searchable=True,
            tokenization=wvcc.Tokenization.WORD,
        ),
        _enum("judgment_id", "Parent judgment id"),
        wvcc.Property(
            name="chunk_idx",
            data_type=wvcc.DataType.INT,
            description="Position of the chunk in the judgment",
            index_filterable=True,
            index_searchable=False,
        ),
        # copied from the judgment so that filtered ANN search needs no join
        _enum("document_type", "Copy of Judgments.document_type"),
        _enum("language", "Copy of Judgments.language"),
        _enum("court_type", "Copy of Judgments.court_type"),
        _enum("judgment_type", "Copy of Judgments.judgment_type"),
        _enum("court_name", "Copy of Judgments.court_name"),
        _date("date_issued", "Copy of Judgments.date_issued"),
    ]


def create_collections(
    client: weaviate.WeaviateClient,
    config: SchemaConfig = SchemaConfig(),
    ie_schemas: Sequence[IeSchema] = (),
) -> None:
    """Create `Judgments` and `JudgmentChunks` if they do not exist yet.

    `ie_schemas` are added at creation time, including the flat chunk copies. Further IE schemas
    can be added later with `ie_properties.add_ie_schema`.
    """
    judgment_props = judgments_properties()
    chunk_props = judgment_chunks_properties()
    for ie in ie_schemas:
        judgment_props += ie_properties(ie)
        chunk_props += ie_chunk_properties(ie)

    if client.collections.exists(JUDGMENTS_COLLECTION):
        logger.info("Collection '{}' already exists, skipping", JUDGMENTS_COLLECTION)
    else:
        client.collections.create(
            name=JUDGMENTS_COLLECTION,
            description="Court judgments: metadata, full text, thesis/summary vectors",
            properties=judgment_props,
            vector_config=[
                _vector(JudgmentVector.THESIS, config),
                _vector(JudgmentVector.SUMMARY, config),
            ],
            inverted_index_config=_inverted_index(config),
            sharding_config=wvcc.Configure.sharding(desired_count=config.judgment_shards),
        )
        logger.info("Collection '{}' created", JUDGMENTS_COLLECTION)

    if client.collections.exists(JUDGMENT_CHUNKS_COLLECTION):
        logger.info("Collection '{}' already exists, skipping", JUDGMENT_CHUNKS_COLLECTION)
    else:
        client.collections.create(
            name=JUDGMENT_CHUNKS_COLLECTION,
            description="Judgment chunks (480 tokens, 64 overlap) for hybrid retrieval + MaxP",
            properties=chunk_props,
            references=[
                wvcc.ReferenceProperty(
                    name=PARENT_JUDGMENT_REF,
                    target_collection=JUDGMENTS_COLLECTION,
                    description="Parent judgment",
                )
            ],
            vector_config=[_vector(VectorName.DEFAULT, config)],
            inverted_index_config=_inverted_index(config),
            sharding_config=wvcc.Configure.sharding(desired_count=config.chunk_shards),
        )
        logger.info("Collection '{}' created", JUDGMENT_CHUNKS_COLLECTION)


def judgment_uuid(judgment_id: str) -> str:
    """Deterministic object id of a judgment."""
    return weaviate.util.generate_uuid5(judgment_id)


def chunk_uuid(judgment_id: str, chunk_idx: int) -> str:
    """Deterministic object id of a chunk (re-ingest overwrites instead of duplicating)."""
    return weaviate.util.generate_uuid5(f"{judgment_id}_chunk_{chunk_idx}")


def connect_local(
    host: str = "localhost",
    port: int = 8110,
    grpc_port: int = 50061,
    api_key: Optional[str] = None,
) -> weaviate.WeaviateClient:
    """Connect to the v2 stack (ports from weaviate/v2/docker-compose.yaml)."""
    auth = weaviate.auth.Auth.api_key(api_key) if api_key else None
    return weaviate.connect_to_local(
        host=host, port=port, grpc_port=grpc_port, auth_credentials=auth
    )
