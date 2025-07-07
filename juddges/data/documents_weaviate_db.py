from typing import Any, ClassVar, Dict, List, Optional, Union

from loguru import logger

import weaviate
import weaviate.classes.config as wvcc
from juddges.data.base_weaviate_db import BaseWeaviateDB
from juddges.data.schemas import (
    DocumentChunk,
    LegalDocument,
)
from juddges.settings import VectorName


class WeaviateLegalDocumentsDatabase(BaseWeaviateDB):
    """Database for legal documents including both judgments and tax interpretations."""

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
        """Get list of property names for the legal documents collection.

        Returns:
            list[str]: List of property names in the legal documents collection.
        """
        config = self.legal_documents_collection.config.get()
        return [prop.name for prop in config.properties]

    @property
    def document_chunks_properties(self) -> list[str]:
        """Get list of property names for the document chunks collection.

        Returns:
            list[str]: List of property names in the document chunks collection.
        """
        config = self.document_chunks_collection.config.get()
        return [prop.name for prop in config.properties]

    def get_collection(self, collection_name: str) -> weaviate.collections.Collection:
        return self.client.collections.get(collection_name)

    def get_collection_size(self, collection: weaviate.collections.Collection) -> int:
        """Get the number of objects in a collection.

        Args:
            collection: The collection to get the size of.

        Returns:
            The number of objects in the collection.
        """
        response = collection.aggregate.over_all(total_count=True)
        return response.total_count

    def safe_create_collection(
        self, name: str, description: str, properties: List[wvcc.Property], vectorizer_config: Any
    ) -> None:
        """Safely create a collection if it doesn't already exist.

        Args:
            name: Name of the collection to create
            description: Description of the collection
            properties: List of property configurations
            vectorizer_config: Vectorizer configuration
        """
        try:
            self.client.collections.create(
                name=name,
                description=description,
                properties=properties,
                vectorizer_config=vectorizer_config,
            )
            logger.info(f"Collection '{name}' created successfully")
        except weaviate.exceptions.UnexpectedStatusCodeError as err:
            if "already exists" in str(err) and err.status_code == 422:
                logger.info(f"Collection '{name}' already exists, skipping creation")
            else:
                logger.error(f"Error creating collection '{name}': {err}")
                raise

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection if it exists.

        Args:
            collection_name: Name of the collection to delete
        """
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
        # Create LegalDocument collection
        self.safe_create_collection(
            name=self.LEGAL_DOCUMENTS_COLLECTION,
            description="Collection of legal documents",
            properties=[
                wvcc.Property(
                    name="document_id",
                    data_type=wvcc.DataType.TEXT,
                    description="Unique identifier for the document",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="document_type",
                    data_type=wvcc.DataType.TEXT,
                    description="Type of legal document (judgment, tax interpretation, etc.)",
                    skip_vectorization=False,
                ),
                wvcc.Property(
                    name="title",
                    data_type=wvcc.DataType.TEXT,
                    description="Document title/name",
                    skip_vectorization=False,
                ),
                wvcc.Property(
                    name="date_issued",
                    data_type=wvcc.DataType.TEXT,
                    description="When the document was published, ISO format date",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="document_number",
                    data_type=wvcc.DataType.TEXT,
                    description="Official reference number",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="language",
                    data_type=wvcc.DataType.TEXT,
                    description="Document language",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="country",
                    data_type=wvcc.DataType.TEXT,
                    description="Country of origin",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="full_text",
                    data_type=wvcc.DataType.TEXT,
                    description="Raw full text of the document",
                    skip_vectorization=False,
                ),
                wvcc.Property(
                    name="summary",
                    data_type=wvcc.DataType.TEXT,
                    description="Abstract or summary",
                    skip_vectorization=False,
                ),
                wvcc.Property(
                    name="thesis",
                    data_type=wvcc.DataType.TEXT,
                    description="Thesis or main point of the document",
                    skip_vectorization=False,
                ),
                wvcc.Property(
                    name="keywords",
                    data_type=wvcc.DataType.TEXT_ARRAY,
                    description="Keywords describing the document",
                    skip_vectorization=False,
                ),
                wvcc.Property(
                    name="x",
                    data_type=wvcc.DataType.NUMBER,
                    description="X coordinate for visualization",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="y",
                    data_type=wvcc.DataType.NUMBER,
                    description="Y coordinate for visualization",
                    skip_vectorization=True,
                ),
                # Flattened fields for previously nested objects (store as JSON string)
                wvcc.Property(
                    name="issuing_body",
                    data_type=wvcc.DataType.TEXT,
                    description="JSON string: Information about the body that issued the legal document",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="ingestion_date",
                    data_type=wvcc.DataType.DATE,
                    description="When document was ingested (ISO format datetime)",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="last_updated",
                    data_type=wvcc.DataType.DATE,
                    description="When document was last updated (ISO format datetime)",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="processing_status",
                    data_type=wvcc.DataType.TEXT,
                    description="Processing status of the document",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="source_url",
                    data_type=wvcc.DataType.TEXT,
                    description="Source URL of the document",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="confidence_score",
                    data_type=wvcc.DataType.NUMBER,
                    description="Confidence score for data extracted via ML",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="legal_references",
                    data_type=wvcc.DataType.TEXT,
                    description="JSON string: References to legal acts, regulations, or previous cases",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="legal_concepts",
                    data_type=wvcc.DataType.TEXT,
                    description="JSON string: Legal concepts or topics discussed in the document",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="parties",
                    data_type=wvcc.DataType.TEXT,
                    description="JSON string: Parties involved in the legal case or interpretation",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="outcome",
                    data_type=wvcc.DataType.TEXT,
                    description="JSON string: The outcome or result of the legal document",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="judgment_specific",
                    data_type=wvcc.DataType.TEXT,
                    description="JSON string: Fields specific to judgment documents",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="tax_interpretation_specific",
                    data_type=wvcc.DataType.TEXT,
                    description="JSON string: Fields specific to tax interpretation documents",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="legal_act_specific",
                    data_type=wvcc.DataType.TEXT,
                    description="JSON string: Fields specific to legal acts (statutes, regulations, etc.)",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="relationships",
                    data_type=wvcc.DataType.TEXT,
                    description="JSON string: Relationships to other documents",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="legal_analysis",
                    data_type=wvcc.DataType.TEXT,
                    description="JSON string: Analysis elements common across document types",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="structured_content",
                    data_type=wvcc.DataType.TEXT,
                    description="JSON string: Structured representation of document content",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="section_embeddings",
                    data_type=wvcc.DataType.TEXT,
                    description="JSON string: Vector embeddings for each section for semantic search",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="metadata",
                    data_type=wvcc.DataType.TEXT,
                    description="JSON string: System metadata for the document",
                    skip_vectorization=True,
                ),
                # Additional properties from mapping
                wvcc.Property(
                    name="publication_date",
                    data_type=wvcc.DataType.TEXT,
                    description="Publication date of the document",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="source_id",
                    data_type=wvcc.DataType.TEXT,
                    description="Court ID",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="judgment_type",
                    data_type=wvcc.DataType.TEXT,
                    description="Type of judgment",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="raw_content",
                    data_type=wvcc.DataType.TEXT,
                    description="XML content of the document",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="presiding_judge",
                    data_type=wvcc.DataType.TEXT,
                    description="Presiding judge information",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="judges",
                    data_type=wvcc.DataType.TEXT,
                    description="Judges involved in the case",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="legal_bases",
                    data_type=wvcc.DataType.TEXT,
                    description="Legal bases for the judgment",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="publisher",
                    data_type=wvcc.DataType.TEXT,
                    description="Publisher of the document",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="recorder",
                    data_type=wvcc.DataType.TEXT,
                    description="Recorder of the document",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="reviser",
                    data_type=wvcc.DataType.TEXT,
                    description="Reviser of the document",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="num_pages",
                    data_type=wvcc.DataType.NUMBER,
                    description="Number of pages in the document",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="volume_number",
                    data_type=wvcc.DataType.TEXT,
                    description="Volume number",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="volume_type",
                    data_type=wvcc.DataType.TEXT,
                    description="Volume type",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="court_name",
                    data_type=wvcc.DataType.TEXT,
                    description="Name of the court",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="department_name",
                    data_type=wvcc.DataType.TEXT,
                    description="Name of the department",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="extracted_legal_bases",
                    data_type=wvcc.DataType.TEXT,
                    description="Extracted legal bases",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="references",
                    data_type=wvcc.DataType.TEXT,
                    description="References in the document",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="court_type",
                    data_type=wvcc.DataType.TEXT,
                    description="Type of court",
                    skip_vectorization=True,
                ),
            ],
            vectorizer_config=[
                wvcc.Configure.NamedVectors.text2vec_transformers(
                    name=VectorName.BASE,
                    vectorize_collection_name=False,
                    source_properties=["full_text"],
                    vector_index_config=wvcc.Configure.VectorIndex.hnsw(),
                ),
                wvcc.Configure.NamedVectors.text2vec_transformers(
                    name=VectorName.DEV,
                    vectorize_collection_name=False,
                    source_properties=["full_text"],
                    vector_index_config=wvcc.Configure.VectorIndex.hnsw(),
                ),
                wvcc.Configure.NamedVectors.text2vec_transformers(
                    name=VectorName.FAST,
                    vectorize_collection_name=False,
                    source_properties=["full_text"],
                    vector_index_config=wvcc.Configure.VectorIndex.hnsw(),
                ),
            ],
        )

        # Create DocumentChunks collection
        self.safe_create_collection(
            name=self.DOCUMENT_CHUNKS_COLLECTION,
            description="Collection of document chunks",
            properties=[
                wvcc.Property(
                    name="document_id",
                    data_type=wvcc.DataType.TEXT,
                    description="ID of the parent document",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="document_type",
                    data_type=wvcc.DataType.TEXT,
                    description="Type of document",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="language",
                    data_type=wvcc.DataType.TEXT,
                    description="Language of the document chunk",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="chunk_id",
                    data_type=wvcc.DataType.NUMBER,
                    description="Chunk identifier",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="chunk_text",
                    data_type=wvcc.DataType.TEXT,
                    description="Text content of the chunk",
                    skip_vectorization=False,
                ),
                wvcc.Property(
                    name="segment_type",
                    data_type=wvcc.DataType.TEXT,
                    description="Type of segment",
                    skip_vectorization=False,
                ),
                wvcc.Property(
                    name="position",
                    data_type=wvcc.DataType.NUMBER,
                    description="Order in document",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="confidence_score",
                    data_type=wvcc.DataType.NUMBER,
                    description="Confidence of segment classification",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="cited_references",
                    data_type=wvcc.DataType.TEXT,
                    description="JSON string: References cited in this chunk",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="tags",
                    data_type=wvcc.DataType.TEXT,
                    description="JSON string: Custom semantic tags for this chunk",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="parent_segment_id",
                    data_type=wvcc.DataType.TEXT,
                    description="ID of parent segment",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="x",
                    data_type=wvcc.DataType.NUMBER,
                    description="X coordinate for visualization",
                    skip_vectorization=True,
                ),
                wvcc.Property(
                    name="y",
                    data_type=wvcc.DataType.NUMBER,
                    description="Y coordinate for visualization",
                    skip_vectorization=True,
                ),
            ],
            vectorizer_config=[
                wvcc.Configure.NamedVectors.text2vec_transformers(
                    name=VectorName.BASE,
                    vectorize_collection_name=False,
                    source_properties=["chunk_text"],
                    vector_index_config=wvcc.Configure.VectorIndex.hnsw(),
                ),
                wvcc.Configure.NamedVectors.text2vec_transformers(
                    name=VectorName.DEV,
                    vectorize_collection_name=False,
                    source_properties=["chunk_text"],
                    vector_index_config=wvcc.Configure.VectorIndex.hnsw(),
                ),
                wvcc.Configure.NamedVectors.text2vec_transformers(
                    name=VectorName.FAST,
                    vectorize_collection_name=False,
                    source_properties=["chunk_text"],
                    vector_index_config=wvcc.Configure.VectorIndex.hnsw(),
                ),
            ],
        )

        logger.info("Collections created successfully")

    @staticmethod
    def uuid_from_document_chunk_id(document_id: str, chunk_id: int) -> str:
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
        """Semantic search within specific segment types using the specified vector."""
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
