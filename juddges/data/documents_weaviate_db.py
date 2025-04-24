import weaviate.classes.config as wvcc
from typing import ClassVar, Optional, Any, Union, List, Dict
import datetime
from loguru import logger

import weaviate
from juddges.data.base_weaviate_db import BaseWeaviateDB
from juddges.data.schemas import (
    LegalDocumentSchema,
    LegalDocumentSectionSchema,
    LegalDocumentSentenceSchema,
)


class WeaviateLegalDocumentsDatabase(BaseWeaviateDB):
    """Database for legal documents including both judgments and tax interpretations."""
    
    LEGAL_DOCUMENTS_COLLECTION: ClassVar[str] = "legal_documents"
    DOCUMENT_CHUNKS_COLLECTION: ClassVar[str] = "document_chunks"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._documents_collection: Optional[weaviate.collections.Collection] = None
        self._sections_collection: Optional[weaviate.collections.Collection] = None
        self._sentences_collection: Optional[weaviate.collections.Collection] = None

    @property
    def documents_collection(self) -> weaviate.collections.Collection:
        if self._documents_collection is None:
            self._documents_collection = self.client.collections.get("LegalDocument")
        return self._documents_collection

    @property
    def sections_collection(self) -> weaviate.collections.Collection:
        if self._sections_collection is None:
            self._sections_collection = self.client.collections.get("LegalDocumentSection")
        return self._sections_collection

    @property
    def sentences_collection(self) -> weaviate.collections.Collection:
        if self._sentences_collection is None:
            self._sentences_collection = self.client.collections.get("LegalDocumentSentence")
        return self._sentences_collection

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
        response = collection.aggregate.over_all(
            total_count=True
        )
        return response.total_count

    def create_collections(self) -> None:
        # Create LegalDocument collection
        self.safe_create_collection(
            name="LegalDocument",
            description="Collection of legal documents",
            properties=[
                wvcc.Property(
                    name=name,
                    data_type=wvcc.DataType.TEXT if dtype == str else wvcc.DataType.NUMBER,
                    description=description,
                    skip_vectorization=not vectorize,
                )
                for name, dtype, description, vectorize in LegalDocumentSchema
            ],
            vectorizer_config=wvcc.Configure.Vectorizer.text2vec(
                vectorize_collection_name=False,
                model="sentence-transformers/all-mpnet-base-v2",
            ),
        )

        # Create LegalDocumentSection collection
        self.safe_create_collection(
            name="LegalDocumentSection",
            description="Collection of legal document sections",
            properties=[
                wvcc.Property(
                    name=name,
                    data_type=wvcc.DataType.TEXT if dtype == str else wvcc.DataType.NUMBER,
                    description=description,
                    skip_vectorization=not vectorize,
                )
                for name, dtype, description, vectorize in LegalDocumentSectionSchema
            ] + [
                wvcc.ReferenceProperty(
                    name="document",
                    target_collection="LegalDocument",
                ),
            ],
            vectorizer_config=wvcc.Configure.Vectorizer.text2vec(
                vectorize_collection_name=False,
                model="sentence-transformers/all-mpnet-base-v2",
            ),
        )

        # Create LegalDocumentSentence collection
        self.safe_create_collection(
            name="LegalDocumentSentence",
            description="Collection of legal document sentences",
            properties=[
                wvcc.Property(
                    name=name,
                    data_type=wvcc.DataType.TEXT if dtype == str else wvcc.DataType.NUMBER,
                    description=description,
                    skip_vectorization=not vectorize,
                )
                for name, dtype, description, vectorize in LegalDocumentSentenceSchema
            ] + [
                wvcc.ReferenceProperty(
                    name="document",
                    target_collection="LegalDocument",
                ),
                wvcc.ReferenceProperty(
                    name="section",
                    target_collection="LegalDocumentSection",
                ),
            ],
            vectorizer_config=wvcc.Configure.Vectorizer.text2vec(
                vectorize_collection_name=False,
                model="sentence-transformers/all-mpnet-base-v2",
            ),
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

    def get_document_relationships(
        self,
        document_id: str,
        relationship_type: Optional[str] = None,
        as_source: bool = True,
        as_target: bool = False,
    ) -> list[dict]:
        """Get relationships for a document.
        
        Args:
            document_id: The ID of the document to get relationships for.
            relationship_type: Optional relationship type to filter by.
            as_source: Whether to include relationships where this document is the source.
            as_target: Whether to include relationships where this document is the target.
        
        Returns:
            A list of relationship objects.
        """
        relationships_collection = self.client.collections.get("document_relationships")
        
        # Build filters
        filters = None
        
        if as_source and not as_target:
            filters = weaviate.classes.query.Filter.by_property("source_id").equal(document_id)
        elif as_target and not as_source:
            filters = weaviate.classes.query.Filter.by_property("target_id").equal(document_id)
        elif as_source and as_target:
            source_filter = weaviate.classes.query.Filter.by_property("source_id").equal(document_id)
            target_filter = weaviate.classes.query.Filter.by_property("target_id").equal(document_id)
            filters = source_filter.or_.merge(target_filter)
        else:
            return []
        
        # Add relationship type filter if specified
        if relationship_type:
            filters = filters.and_.by_property("relationship_type").equal(relationship_type)
        
        # Execute query
        response = relationships_collection.query.fetch_objects(
            filters=filters,
            limit=1000,  # Use a high limit
        )
        
        return [item.properties for item in response.objects]
        
    def semantic_search(
        self, query: str, vector_name: str = "base", limit: int = 10, 
        document_type: Optional[str] = None, collection_name: str = LEGAL_DOCUMENTS_COLLECTION
    ) -> list[dict]:
        """Perform semantic search using a specified named vector.
        
        Args:
            query: The search query.
            vector_name: The named vector to use for search (base, dev, or semantic).
            limit: Maximum number of results to return.
            document_type: Optional filter for document type.
            collection_name: Collection to search in.
            
        Returns:
            List of matching documents.
        """
        collection = self.client.collections.get(collection_name)
        
        # Build filters if document_type is provided
        filters = None
        if document_type:
            filters = weaviate.classes.query.Filter.by_property("document_type").equal(document_type)
        
        response = collection.query.near_text(
            query=query,
            target_vectors=[vector_name],
            filters=filters,
            limit=limit,
        )
        
        return [item.properties for item in response.objects]

    async def search_by_tags(self, tags: list[str], limit: int = 10) -> list[dict]:
        """Search for document chunks containing specific tags."""
        # For multiple tags, we need to build a filter with OR conditions
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
        """Semantic search within specific segment types using the specified vector.
        
        Args:
            query: The search query.
            segment_type: The segment type to search in.
            vector_name: The named vector to use for search (base, dev, or semantic).
            limit: Maximum number of results to return.
            
        Returns:
            List of matching document chunks.
        """
        response = await self.document_chunks_collection.query.near_text(
            query=query,
            target_vectors=[vector_name],
            filters=weaviate.classes.query.Filter.by_property("segment_type").equal(segment_type),
            limit=limit,
        )
        return [item.properties for item in response.objects]
        
    async def create_document_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        description: Optional[str] = None,
        context_segment_id: Optional[str] = None,
        confidence_score: Optional[float] = None,
        bidirectional: bool = False,
        metadata: Optional[dict] = None,
    ) -> str:
        """Create a relationship between two documents."""
        
        # Create a UUID for the relationship
        rel_id = weaviate.util.generate_uuid5(f"{source_id}_{relationship_type}_{target_id}")
        
        # Create the relationship object
        relationship = {
            "source_id": source_id,
            "target_id": target_id,
            "relationship_type": relationship_type,
            "description": description,
            "context_segment_id": context_segment_id,
            "confidence_score": confidence_score,
            "bidirectional": bidirectional,
            "creation_date": datetime.datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        # Insert the relationship
        relationships_collection = self.client.collections.get("document_relationships")
        await relationships_collection.data.insert(relationship, rel_id)
        
        return rel_id

    async def get_citing_documents(self, document_id: str, limit: int = 10) -> list[dict]:
        """Get documents that cite the specified document."""
        # Get relationships where the document is the target and relationship type is 'cites'
        relationships = await self.get_document_relationships(
            document_id=document_id,
            relationship_type="cites",
            as_source=False,
            as_target=True,
        )
        
        # Get the citing document IDs
        citing_ids = [rel["source_id"] for rel in relationships]
        
        # Limit to the requested number
        citing_ids = citing_ids[:limit]
        
        # If no citing documents, return empty list
        if not citing_ids:
            return []
        
        # Get the actual documents
        result = []
        for doc_id in citing_ids:
            try:
                document = await self.legal_documents_collection.data.get_by_id(doc_id)
                result.append(document.properties)
            except Exception as e:
                logger.error(f"Error retrieving citing document {doc_id}: {str(e)}")
        
        return result
        
    async def semantic_search(
        self, query: str, vector_name: str = "base", limit: int = 10, 
        document_type: Optional[str] = None, collection_name: str = LEGAL_DOCUMENTS_COLLECTION
    ) -> list[dict]:
        """Perform semantic search using a specified named vector.
        
        Args:
            query: The search query.
            vector_name: The named vector to use for search (base, dev, or semantic).
            limit: Maximum number of results to return.
            document_type: Optional filter for document type.
            collection_name: Collection to search in.
            
        Returns:
            List of matching documents.
        """
        collection = self.client.collections.get(collection_name)
        
        # Build filters if document_type is provided
        filters = None
        if document_type:
            filters = weaviate.classes.query.Filter.by_property("document_type").equal(document_type)
        
        response = await collection.query.near_text(
            query=query,
            target_vectors=[vector_name],
            filters=filters,
            limit=limit,
        )
        
        return [item.properties for item in response.objects]

    def insert(self, document: Union[LegalDocument, DocumentChunk], collection_name: Optional[str] = None) -> None:
        """Insert a single document or chunk into the appropriate collection.
        
        Args:
            document: The document or chunk to insert
            collection_name: Optional collection name override
        """
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
        """Search for documents using semantic search.
        
        Args:
            query: The search query
            collection_name: Optional collection name override
            vector_name: The named vector to use for search (base, dev, or semantic)
            limit: Maximum number of results to return
            filters: Optional query filters
            
        Returns:
            List of matching documents
        """
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
        """Delete a document from the specified collection.
        
        Args:
            document_id: ID of the document to delete
            collection_name: Optional collection name override
        """
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