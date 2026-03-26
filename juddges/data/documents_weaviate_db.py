"""
Weaviate database for legal documents with BlockMax WAND optimizations.

Key optimizations:
- 4-shard configuration for parallel aggregate queries
- Polish stopwords for BM25 keyword search
- Proper tokenization (FIELD for enums, WORD for text)
- Optimized index_searchable settings (only essential fields)
- DATE types with range filters
- Cross-reference from DocumentChunks to LegalDocuments
"""

from typing import Any, ClassVar, Dict, List, Optional, Union

from loguru import logger

import weaviate
import weaviate.classes.config as wvcc
from weaviate.classes.config import ReferenceProperty
from juddges.data.base_weaviate_db import BaseWeaviateDB
from juddges.data.schemas import (
    DocumentChunk,
    LegalDocument,
)
from juddges.settings import VectorName


# Polish stopwords for BM25 - common words that don't add search value
POLISH_STOPWORDS = [
    "aby",
    "ale",
    "albo",
    "ani",
    "aż",
    "bardzo",
    "bez",
    "bo",
    "bowiem",
    "by",
    "był",
    "była",
    "byli",
    "było",
    "być",
    "będzie",
    "będą",
    "ci",
    "cię",
    "co",
    "czy",
    "czyli",
    "dla",
    "do",
    "go",
    "gdy",
    "gdyż",
    "gdzie",
    "i",
    "ich",
    "im",
    "inne",
    "inny",
    "ja",
    "jak",
    "jakie",
    "jakiś",
    "jako",
    "je",
    "jednak",
    "jego",
    "jej",
    "jest",
    "jeszcze",
    "jeśli",
    "jeżeli",
    "już",
    "każdy",
    "kiedy",
    "kilka",
    "kto",
    "która",
    "które",
    "który",
    "ku",
    "lat",
    "lub",
    "ma",
    "mają",
    "mi",
    "mimo",
    "między",
    "mnie",
    "może",
    "można",
    "mu",
    "musi",
    "my",
    "na",
    "nad",
    "nam",
    "nas",
    "nawet",
    "nic",
    "nich",
    "nie",
    "niech",
    "niego",
    "niej",
    "nim",
    "niż",
    "no",
    "o",
    "ob",
    "od",
    "on",
    "ona",
    "one",
    "oni",
    "ono",
    "oraz",
    "pan",
    "pani",
    "po",
    "pod",
    "podczas",
    "ponad",
    "ponieważ",
    "przed",
    "przede",
    "przez",
    "przy",
    "roku",
    "również",
    "są",
    "się",
    "sobie",
    "sposób",
    "swoje",
    "ta",
    "tak",
    "także",
    "tam",
    "te",
    "tego",
    "tej",
    "ten",
    "teraz",
    "to",
    "tobie",
    "tu",
    "tutaj",
    "twój",
    "ty",
    "tym",
    "tylko",
    "w",
    "we",
    "więc",
    "więcej",
    "właśnie",
    "wszystko",
    "wtedy",
    "wy",
    "z",
    "za",
    "zaś",
    "ze",
    "że",
    "żeby",
]


class WeaviateLegalDocumentsDatabase(BaseWeaviateDB):
    """Database for legal documents with BlockMax WAND optimizations."""

    LEGAL_DOCUMENTS_COLLECTION: ClassVar[str] = "LegalDocuments"
    DOCUMENT_CHUNKS_COLLECTION: ClassVar[str] = "DocumentChunks"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @property
    def legal_documents_collection(self) -> weaviate.collections.Collection:
        return self.client.collections.get(self.LEGAL_DOCUMENTS_COLLECTION)

    @property
    def document_chunks_collection(self) -> weaviate.collections.Collection:
        return self.client.collections.get(self.DOCUMENT_CHUNKS_COLLECTION)

    @property
    def legal_documents_properties(self) -> list[str]:
        """Get list of property names for the legal documents collection."""
        config = self.legal_documents_collection.config.get()
        return [prop.name for prop in config.properties]

    @property
    def document_chunks_properties(self) -> list[str]:
        """Get list of property names for the document chunks collection."""
        config = self.document_chunks_collection.config.get()
        return [prop.name for prop in config.properties]

    def get_collection(self, collection_name: str) -> weaviate.collections.Collection:
        return self.client.collections.get(collection_name)

    def get_collection_size(self, collection: weaviate.collections.Collection) -> int:
        """Get the number of objects in a collection."""
        response = collection.aggregate.over_all(total_count=True)
        return response.total_count

    def safe_create_collection(
        self,
        name: str,
        description: str,
        properties: List[wvcc.Property],
        vectorizer_config: Any,
        inverted_index_config: Any = None,
        sharding_config: Any = None,
        references: Optional[List[ReferenceProperty]] = None,
    ) -> None:
        """Safely create a collection if it doesn't already exist."""
        try:
            self.client.collections.create(
                name=name,
                description=description,
                properties=properties,
                vectorizer_config=vectorizer_config,
                inverted_index_config=inverted_index_config,
                sharding_config=sharding_config,
                references=references,
            )
            logger.info(f"Collection '{name}' created successfully")
        except weaviate.exceptions.UnexpectedStatusCodeError as err:
            if "already exists" in str(err) and err.status_code == 422:
                logger.info(f"Collection '{name}' already exists, skipping creation")
            else:
                logger.error(f"Error creating collection '{name}': {err}")
                raise

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection if it exists."""
        try:
            self.client.collections.delete(collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
        except weaviate.exceptions.UnexpectedStatusCodeError as err:
            if "not found" in str(err).lower():
                logger.info(f"Collection '{collection_name}' does not exist, skipping deletion")
            else:
                logger.error(f"Error deleting collection '{collection_name}': {err}")
                raise

    def create_collections(self) -> None:
        """Create optimized collections with BlockMax WAND support."""

        # Inverted index config with Polish stopwords for BM25
        inverted_index_config = wvcc.Configure.inverted_index(
            bm25_b=0.75,
            bm25_k1=1.2,
            cleanup_interval_seconds=60,
            stopwords_preset=wvcc.StopwordsPreset.EN,
            stopwords_additions=POLISH_STOPWORDS,
        )

        # 4-shard config for ~3M documents (optimal for aggregates)
        sharding_config = wvcc.Configure.sharding(
            desired_count=4,
            desired_virtual_count=128,
            virtual_per_physical=32,
        )

        # ============================================================
        # LEGAL DOCUMENTS COLLECTION
        # ============================================================
        self.safe_create_collection(
            name=self.LEGAL_DOCUMENTS_COLLECTION,
            description="Collection of legal documents with BlockMax WAND",
            inverted_index_config=inverted_index_config,
            sharding_config=sharding_config,
            properties=[
                # === IDENTIFIERS (filterable, not BM25 searchable) ===
                wvcc.Property(
                    name="document_id",
                    data_type=wvcc.DataType.TEXT,
                    description="Unique identifier for the document",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.FIELD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="document_number",
                    data_type=wvcc.DataType.TEXT,
                    description="Official reference number (e.g., case number)",
                    index_filterable=True,
                    index_searchable=True,  # Users search by case numbers
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="source_url",
                    data_type=wvcc.DataType.TEXT,
                    description="Source URL of the document",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.FIELD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="source_id",
                    data_type=wvcc.DataType.TEXT,
                    description="Source-specific ID",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.FIELD,
                    skip_vectorization=True,
                ),
                # === ENUM FIELDS (FIELD tokenization for exact match) ===
                wvcc.Property(
                    name="document_type",
                    data_type=wvcc.DataType.TEXT,
                    description="Type: judgment, tax_interpretation, legal_act",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.FIELD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="language",
                    data_type=wvcc.DataType.TEXT,
                    description="Document language (pl, en, etc.)",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.FIELD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="country",
                    data_type=wvcc.DataType.TEXT,
                    description="Country of origin",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.FIELD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="court_type",
                    data_type=wvcc.DataType.TEXT,
                    description="Type of court",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.FIELD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="judgment_type",
                    data_type=wvcc.DataType.TEXT,
                    description="Type of judgment",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.FIELD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="processing_status",
                    data_type=wvcc.DataType.TEXT,
                    description="Processing status",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.FIELD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="outcome",
                    data_type=wvcc.DataType.TEXT,
                    description="Outcome/result of the case",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.FIELD,
                    skip_vectorization=True,
                ),
                # === DATE FIELDS (proper DATE type with range filters) ===
                wvcc.Property(
                    name="date_issued",
                    data_type=wvcc.DataType.DATE,
                    description="When the document was issued",
                    index_filterable=True,
                    index_range_filters=True,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="publication_date",
                    data_type=wvcc.DataType.DATE,
                    description="When the document was published",
                    index_filterable=True,
                    index_range_filters=True,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="ingestion_date",
                    data_type=wvcc.DataType.DATE,
                    description="When document was ingested",
                    index_filterable=True,
                    index_range_filters=True,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="last_updated",
                    data_type=wvcc.DataType.DATE,
                    description="When document was last updated",
                    index_filterable=True,
                    index_range_filters=True,
                    skip_vectorization=True,
                ),
                # === SEARCHABLE TEXT FIELDS (BM25 enabled) ===
                wvcc.Property(
                    name="full_text",
                    data_type=wvcc.DataType.TEXT,
                    description="Full text content (main search field)",
                    index_filterable=False,
                    index_searchable=True,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=False,
                ),
                wvcc.Property(
                    name="title",
                    data_type=wvcc.DataType.TEXT,
                    description="Document title",
                    index_filterable=True,
                    index_searchable=True,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=False,
                ),
                wvcc.Property(
                    name="summary",
                    data_type=wvcc.DataType.TEXT,
                    description="Document summary/abstract",
                    index_filterable=False,
                    index_searchable=True,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=False,
                ),
                wvcc.Property(
                    name="thesis",
                    data_type=wvcc.DataType.TEXT,
                    description="Legal thesis or main ruling",
                    index_filterable=False,
                    index_searchable=True,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=False,
                ),
                wvcc.Property(
                    name="keywords",
                    data_type=wvcc.DataType.TEXT_ARRAY,
                    description="Keywords/tags",
                    index_filterable=True,
                    index_searchable=True,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=False,
                ),
                # === COURT/JUDGE INFO (filterable, not BM25) ===
                wvcc.Property(
                    name="court_name",
                    data_type=wvcc.DataType.TEXT,
                    description="Name of the court",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="department_name",
                    data_type=wvcc.DataType.TEXT,
                    description="Court department",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="presiding_judge",
                    data_type=wvcc.DataType.TEXT,
                    description="Presiding judge",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="judges",
                    data_type=wvcc.DataType.TEXT_ARRAY,
                    description="List of judges",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="issuing_body",
                    data_type=wvcc.DataType.TEXT,
                    description="Body that issued the document",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=True,
                ),
                # === LEGAL CONTENT (filterable, not BM25) ===
                wvcc.Property(
                    name="legal_bases",
                    data_type=wvcc.DataType.TEXT_ARRAY,
                    description="Legal bases for the judgment",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="extracted_legal_bases",
                    data_type=wvcc.DataType.TEXT,
                    description="LLM-extracted legal bases",
                    index_filterable=False,
                    index_searchable=False,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="references",
                    data_type=wvcc.DataType.TEXT_ARRAY,
                    description="References to other documents",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="parties",
                    data_type=wvcc.DataType.TEXT,
                    description="Parties involved (JSON string)",
                    index_filterable=False,
                    index_searchable=False,
                    skip_vectorization=True,
                ),
                # === LLM-EXTRACTED FIELDS (BM25 searchable) ===
                wvcc.Property(
                    name="factual_state",
                    data_type=wvcc.DataType.TEXT,
                    description="Factual circumstances (LLM-extracted)",
                    index_filterable=False,
                    index_searchable=True,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=False,
                ),
                wvcc.Property(
                    name="legal_state",
                    data_type=wvcc.DataType.TEXT,
                    description="Legal basis and applicable law (LLM-extracted)",
                    index_filterable=False,
                    index_searchable=True,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=False,
                ),
                wvcc.Property(
                    name="legal_concepts",
                    data_type=wvcc.DataType.TEXT,
                    description="Legal concepts discussed (LLM-extracted)",
                    index_filterable=False,
                    index_searchable=False,
                    skip_vectorization=False,
                ),
                wvcc.Property(
                    name="legal_analysis",
                    data_type=wvcc.DataType.TEXT,
                    description="Legal analysis (LLM-extracted)",
                    index_filterable=False,
                    index_searchable=False,
                    skip_vectorization=False,
                ),
                # === DOCUMENT-TYPE SPECIFIC (JSON strings, no index) ===
                wvcc.Property(
                    name="judgment_specific",
                    data_type=wvcc.DataType.TEXT,
                    description="Judgment-specific fields (JSON)",
                    index_filterable=False,
                    index_searchable=False,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="tax_interpretation_specific",
                    data_type=wvcc.DataType.TEXT,
                    description="Tax interpretation fields (JSON)",
                    index_filterable=False,
                    index_searchable=False,
                    skip_vectorization=True,
                ),
                # === RAW CONTENT (storage only, no index) ===
                wvcc.Property(
                    name="raw_content",
                    data_type=wvcc.DataType.TEXT,
                    description="Raw unprocessed content",
                    index_filterable=False,
                    index_searchable=False,
                    skip_vectorization=True,
                ),
                # === SOURCE INFO ===
                wvcc.Property(
                    name="source",
                    data_type=wvcc.DataType.TEXT,
                    description="Data source",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.FIELD,
                    skip_vectorization=True,
                ),
                # === VISUALIZATION COORDINATES ===
                wvcc.Property(
                    name="x",
                    data_type=wvcc.DataType.NUMBER,
                    description="X coordinate for visualization",
                    index_filterable=False,
                    index_searchable=False,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="y",
                    data_type=wvcc.DataType.NUMBER,
                    description="Y coordinate for visualization",
                    index_filterable=False,
                    index_searchable=False,
                    skip_vectorization=True,
                ),
            ],
            vectorizer_config=[
                wvcc.Configure.NamedVectors.none(
                    name=VectorName.DEFAULT,
                    vector_index_config=wvcc.Configure.VectorIndex.hnsw(
                        ef_construction=64,
                        max_connections=16,
                        distance_metric=wvcc.VectorDistances.COSINE,
                        quantizer=wvcc.Configure.VectorIndex.Quantizer.bq(
                            rescore_limit=200,
                        ),
                    ),
                ),
            ],
        )

        # ============================================================
        # DOCUMENT CHUNKS COLLECTION (with cross-reference)
        # ============================================================
        self.safe_create_collection(
            name=self.DOCUMENT_CHUNKS_COLLECTION,
            description="Document chunks with cross-reference to parent",
            inverted_index_config=inverted_index_config,
            sharding_config=sharding_config,
            properties=[
                # === IDENTIFIERS ===
                wvcc.Property(
                    name="document_id",
                    data_type=wvcc.DataType.TEXT,
                    description="Parent document ID",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.FIELD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="chunk_id",
                    data_type=wvcc.DataType.INT,
                    description="Chunk sequence number",
                    index_filterable=True,
                    index_range_filters=True,
                    skip_vectorization=True,
                ),
                # === ENUM FIELDS ===
                wvcc.Property(
                    name="document_type",
                    data_type=wvcc.DataType.TEXT,
                    description="Type of parent document",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.FIELD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="language",
                    data_type=wvcc.DataType.TEXT,
                    description="Language of the chunk",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.FIELD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="segment_type",
                    data_type=wvcc.DataType.TEXT,
                    description="Type of segment",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.FIELD,
                    skip_vectorization=True,
                ),
                # === MAIN CONTENT (BM25 enabled) ===
                wvcc.Property(
                    name="chunk_text",
                    data_type=wvcc.DataType.TEXT,
                    description="Text content of the chunk",
                    index_filterable=False,
                    index_searchable=True,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=False,
                ),
                # === POSITION/ORDER ===
                wvcc.Property(
                    name="position",
                    data_type=wvcc.DataType.INT,
                    description="Position in document",
                    index_filterable=True,
                    index_range_filters=True,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="parent_segment_id",
                    data_type=wvcc.DataType.TEXT,
                    description="ID of parent segment",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.FIELD,
                    skip_vectorization=True,
                ),
                # === METADATA ===
                wvcc.Property(
                    name="confidence_score",
                    data_type=wvcc.DataType.NUMBER,
                    description="ML confidence score",
                    index_filterable=True,
                    index_range_filters=True,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="tags",
                    data_type=wvcc.DataType.TEXT_ARRAY,
                    description="Semantic tags",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="cited_references",
                    data_type=wvcc.DataType.TEXT_ARRAY,
                    description="References cited in chunk",
                    index_filterable=True,
                    index_searchable=False,
                    tokenization=wvcc.Tokenization.WORD,
                    skip_vectorization=True,
                ),
                # === VISUALIZATION COORDINATES ===
                wvcc.Property(
                    name="x",
                    data_type=wvcc.DataType.NUMBER,
                    description="X coordinate for visualization",
                    index_filterable=False,
                    index_searchable=False,
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="y",
                    data_type=wvcc.DataType.NUMBER,
                    description="Y coordinate for visualization",
                    index_filterable=False,
                    index_searchable=False,
                    skip_vectorization=True,
                ),
            ],
            # Cross-reference to parent document
            references=[
                ReferenceProperty(
                    name="parent_document",
                    target_collection=self.LEGAL_DOCUMENTS_COLLECTION,
                    description="Reference to parent LegalDocument",
                ),
            ],
            vectorizer_config=[
                wvcc.Configure.NamedVectors.none(
                    name=VectorName.DEFAULT,
                    vector_index_config=wvcc.Configure.VectorIndex.hnsw(
                        ef_construction=64,
                        max_connections=16,
                        distance_metric=wvcc.VectorDistances.COSINE,
                        quantizer=wvcc.Configure.VectorIndex.Quantizer.bq(
                            rescore_limit=200,
                        ),
                    ),
                ),
            ],
        )

        logger.info("Collections created with BQ quantization and single vector")

    @staticmethod
    def uuid_from_document_id(document_id: str) -> str:
        """Generate deterministic UUID for a document."""
        return weaviate.util.generate_uuid5(document_id)

    @staticmethod
    def uuid_from_document_chunk_id(document_id: str, chunk_id: int) -> str:
        """Generate deterministic UUID for a document chunk."""
        return weaviate.util.generate_uuid5(f"{document_id}_chunk_{chunk_id}")

    def search_by_segment_type(self, segment_type: str, limit: int = 10) -> list[dict]:
        """Search for document chunks by segment type."""
        response = self.document_chunks_collection.query.fetch_objects(
            filters=weaviate.classes.query.Filter.by_property("segment_type").equal(segment_type),
            limit=limit,
        )
        return [item.properties for item in response.objects]

    async def search_by_tags(self, tags: list[str], limit: int = 10) -> list[dict]:
        """Search for document chunks containing specific tags."""
        tag_filters = None
        for tag in tags:
            if tag_filters is None:
                tag_filters = weaviate.classes.query.Filter.by_property("tags").contains(tag)
            else:
                tag_filters = tag_filters.or_.by_property("tags").contains(tag)

        response = await self.document_chunks_collection.query.fetch_objects(
            filters=tag_filters,
            limit=limit,
        )
        return [item.properties for item in response.objects]

    async def semantic_search_in_segment_type(
        self, query: str, segment_type: str, vector_name: str = "base", limit: int = 10
    ) -> list[dict]:
        """Semantic search within specific segment types."""
        response = await self.document_chunks_collection.query.near_text(
            query=query,
            target_vectors=[vector_name],
            filters=weaviate.classes.query.Filter.by_property("segment_type").equal(segment_type),
            limit=limit,
        )
        return [item.properties for item in response.objects]

    def semantic_search(
        self,
        query: str,
        target_vector: str = "base",
        limit: int = 10,
        document_type: Optional[str] = None,
        collection_name: str = LEGAL_DOCUMENTS_COLLECTION,
    ) -> list[dict]:
        """Perform semantic search using a specified named vector."""
        collection = self.client.collections.get(collection_name)

        filters = None
        if document_type:
            filters = weaviate.classes.query.Filter.by_property("document_type").equal(
                document_type
            )

        response = collection.query.near_text(
            query=query,
            target_vector=target_vector,
            filters=filters,
            limit=limit,
        )

        return [item.properties for item in response.objects]

    def bm25_search(
        self,
        query: str,
        limit: int = 10,
        document_type: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search (uses BlockMax WAND)."""
        if collection_name is None:
            collection_name = self.LEGAL_DOCUMENTS_COLLECTION

        collection = self.client.collections.get(collection_name)

        filters = None
        if document_type:
            filters = weaviate.classes.query.Filter.by_property("document_type").equal(
                document_type
            )

        response = collection.query.bm25(
            query=query,
            filters=filters,
            limit=limit,
            return_metadata=weaviate.classes.query.MetadataQuery(score=True),
        )

        return [{**item.properties, "_score": item.metadata.score} for item in response.objects]

    def hybrid_search(
        self,
        query: str,
        alpha: float = 0.5,
        limit: int = 10,
        document_type: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search (BM25 + vector).

        Args:
            query: Search query
            alpha: Balance between vector (1.0) and BM25 (0.0)
            limit: Maximum results
            document_type: Optional filter
            collection_name: Collection to search
        """
        if collection_name is None:
            collection_name = self.LEGAL_DOCUMENTS_COLLECTION

        collection = self.client.collections.get(collection_name)

        filters = None
        if document_type:
            filters = weaviate.classes.query.Filter.by_property("document_type").equal(
                document_type
            )

        response = collection.query.hybrid(
            query=query,
            alpha=alpha,
            filters=filters,
            limit=limit,
            target_vector=VectorName.DEFAULT,
            return_metadata=weaviate.classes.query.MetadataQuery(score=True),
        )

        return [{**item.properties, "_score": item.metadata.score} for item in response.objects]

    def insert(
        self, document: Union[LegalDocument, DocumentChunk], collection_name: Optional[str] = None
    ) -> None:
        """Insert a single document or chunk into the appropriate collection."""
        if collection_name is None:
            if isinstance(document, LegalDocument):
                collection = self.legal_documents_collection
            elif isinstance(document, DocumentChunk):
                collection = self.document_chunks_collection
            else:
                raise ValueError(f"Unsupported document type: {type(document)}")
        else:
            collection = self.get_collection(collection_name)

        try:
            properties = document.dict(exclude_none=True)
            collection.data.insert(properties)
            logger.info(f"Successfully inserted document {document.document_id}")
        except Exception as e:
            logger.error(f"Error inserting document {document.document_id}: {str(e)}")
            raise

    def insert_document_with_chunks(
        self,
        document: LegalDocument,
        chunks: List[DocumentChunk],
    ) -> str:
        """Insert a document and its chunks with cross-references.

        Args:
            document: The legal document to insert
            chunks: List of document chunks

        Returns:
            UUID of the inserted document
        """
        doc_uuid = self.uuid_from_document_id(document.document_id)

        doc_properties = document.dict(exclude_none=True)
        self.legal_documents_collection.data.insert(
            properties=doc_properties,
            uuid=doc_uuid,
        )
        logger.info(f"Inserted document {document.document_id} with UUID {doc_uuid}")

        for chunk in chunks:
            chunk_uuid = self.uuid_from_document_chunk_id(document.document_id, chunk.chunk_id)
            chunk_properties = chunk.dict(exclude_none=True)

            self.document_chunks_collection.data.insert(
                properties=chunk_properties,
                uuid=chunk_uuid,
                references={"parent_document": doc_uuid},
            )

        logger.info(f"Inserted {len(chunks)} chunks for document {document.document_id}")
        return doc_uuid

    def get_chunks_for_document(self, document_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all chunks for a document using the cross-reference."""
        doc_uuid = self.uuid_from_document_id(document_id)

        response = self.document_chunks_collection.query.fetch_objects(
            filters=weaviate.classes.query.Filter.by_ref("parent_document").by_id().equal(doc_uuid),
            limit=limit,
            sort=weaviate.classes.query.Sort.by_property("position", ascending=True),
        )

        return [obj.properties for obj in response.objects]

    def search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        vector_name: str = "base",
        limit: int = 10,
        filters: Optional[weaviate.classes.query.Filter] = None,
    ) -> List[Dict]:
        """Search for documents using semantic search."""
        if collection_name is None:
            collection = self.legal_documents_collection
        else:
            collection = self.get_collection(collection_name)

        try:
            response = collection.query.near_text(
                query=query,
                target_vectors=[vector_name],
                filters=filters,
                limit=limit,
            )
            return [item.properties for item in response.objects]
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

    def delete(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> None:
        """Delete a document from the specified collection."""
        if collection_name is None:
            collection = self.legal_documents_collection
        else:
            collection = self.get_collection(collection_name)

        try:
            collection.data.delete(document_id)
            logger.info(f"Successfully deleted document {document_id}")
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise

    def filter_by_raw_content_presence(
        self, has_raw_content: bool = True, limit: int = 100
    ) -> List[Dict]:
        """Filter documents by raw_content field presence."""
        collection = self.legal_documents_collection

        if has_raw_content:
            filters = weaviate.classes.query.Filter.by_property("raw_content").is_none(False)
        else:
            filters = weaviate.classes.query.Filter.by_property("raw_content").is_none(True)

        response = collection.query.fetch_objects(filters=filters, limit=limit)
        return [obj.properties for obj in response.objects]

    def get_raw_content_statistics(self) -> Dict[str, int]:
        """Get statistics about raw_content field coverage."""
        collection = self.legal_documents_collection

        total_response = collection.aggregate.over_all(total_count=True)
        total = total_response.total_count

        with_raw_content_response = collection.aggregate.over_all(
            total_count=True,
            filters=weaviate.classes.query.Filter.by_property("raw_content").is_none(False),
        )
        with_raw_content = with_raw_content_response.total_count

        without_raw_content = total - with_raw_content

        return {
            "total_documents": total,
            "with_raw_content": with_raw_content,
            "without_raw_content": without_raw_content,
            "coverage_percentage": round((with_raw_content / total * 100) if total > 0 else 0, 2),
        }

    def filter_by_document_type_and_raw_content(
        self, document_type: str, has_raw_content: bool = True, limit: int = 100
    ) -> List[Dict]:
        """Filter documents by type and raw_content presence."""
        collection = self.legal_documents_collection

        type_filter = weaviate.classes.query.Filter.by_property("document_type").equal(
            document_type
        )

        if has_raw_content:
            raw_content_filter = weaviate.classes.query.Filter.by_property("raw_content").is_none(
                False
            )
        else:
            raw_content_filter = weaviate.classes.query.Filter.by_property("raw_content").is_none(
                True
            )

        combined_filter = type_filter & raw_content_filter

        response = collection.query.fetch_objects(filters=combined_filter, limit=limit)
        return [obj.properties for obj in response.objects]

    def compare_content_fields(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Compare full_text and raw_content for a document."""
        doc = self.get_weaviate_document(document_id=document_id)
        if not doc:
            return None

        full_text = doc.get("full_text", "")
        raw_content = doc.get("raw_content", "")

        return {
            "document_id": document_id,
            "has_full_text": bool(full_text),
            "has_raw_content": bool(raw_content),
            "full_text_length": len(full_text) if full_text else 0,
            "raw_content_length": len(raw_content) if raw_content else 0,
            "length_difference": abs(len(full_text or "") - len(raw_content or "")),
            "length_ratio": (
                round(len(full_text) / len(raw_content), 3)
                if full_text and raw_content and len(raw_content) > 0
                else None
            ),
        }

    def get_weaviate_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a Weaviate document by document_id."""
        collection = self.legal_documents_collection
        filters = weaviate.classes.query.Filter.by_property("document_id").equal(document_id)

        response = collection.query.fetch_objects(filters=filters, limit=1)

        if response.objects:
            return response.objects[0].properties
        return None
