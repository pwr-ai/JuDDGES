"""
Utilities for mapping Hugging Face datasets to Weaviate documents.

This module provides functionality to:
- Map dataset records to Weaviate documents using document_id/document_number
- Retrieve raw text from datasets for Weaviate ingestion
- Query Weaviate documents and join with original dataset data
"""

from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset
from loguru import logger

import weaviate
from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase


class DatasetToWeaviateMapper:
    """Maps Hugging Face dataset records to Weaviate documents."""

    def __init__(
        self,
        db: WeaviateLegalDocumentsDatabase,
        dataset_name: Optional[str] = None,
        dataset: Optional[Dataset] = None,
    ):
        """
        Initialize the mapper.

        Args:
            db: WeaviateLegalDocumentsDatabase instance
            dataset_name: Name of HuggingFace dataset to load (e.g., "juddges/pl-court-raw")
            dataset: Pre-loaded dataset (alternative to dataset_name)
        """
        self.db = db
        self.dataset_name = dataset_name
        self._dataset = dataset
        self._dataset_index: Optional[Dict[str, Dict[str, Any]]] = None

    @property
    def dataset(self) -> Dataset:
        """Lazy load the dataset."""
        if self._dataset is None:
            if self.dataset_name is None:
                raise ValueError("Either dataset_name or dataset must be provided")
            logger.info(f"Loading dataset {self.dataset_name}...")
            self._dataset = load_dataset(self.dataset_name, split="train")
        return self._dataset

    def build_index(
        self, id_field: str = "judgment_id", secondary_id_field: Optional[str] = "docket_number"
    ) -> None:
        """
        Build an index for fast lookup by document_id or document_number.

        Args:
            id_field: Primary ID field in the dataset (e.g., "judgment_id", "id")
            secondary_id_field: Secondary ID field for fallback lookup (e.g., "docket_number")
        """
        logger.info("Building dataset index...")
        self._dataset_index = {}

        for row in self.dataset:
            # Index by primary ID
            primary_id = row.get(id_field)
            if primary_id:
                self._dataset_index[primary_id] = row

            # Index by secondary ID if available
            if secondary_id_field:
                secondary_id = row.get(secondary_id_field)
                if secondary_id:
                    self._dataset_index[secondary_id] = row

        logger.info(f"Built index with {len(self._dataset_index)} entries")

    def get_dataset_record(
        self, document_id: Optional[str] = None, document_number: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a dataset record by document_id or document_number.

        Args:
            document_id: Document ID from Weaviate
            document_number: Document number from Weaviate

        Returns:
            Dataset record dict or None if not found
        """
        if self._dataset_index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Try document_id first
        if document_id and document_id in self._dataset_index:
            return self._dataset_index[document_id]

        # Fallback to document_number
        if document_number and document_number in self._dataset_index:
            return self._dataset_index[document_number]

        return None

    def get_weaviate_document(
        self, document_id: Optional[str] = None, document_number: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a Weaviate document by document_id or document_number.

        Args:
            document_id: Document ID to search for
            document_number: Document number to search for

        Returns:
            Weaviate document properties or None if not found
        """
        collection = self.db.legal_documents_collection

        # Build filter
        filters = None
        if document_id:
            filters = weaviate.classes.query.Filter.by_property("document_id").equal(document_id)
        elif document_number:
            filters = weaviate.classes.query.Filter.by_property("document_number").equal(
                document_number
            )
        else:
            raise ValueError("Either document_id or document_number must be provided")

        response = collection.query.fetch_objects(filters=filters, limit=1)

        if response.objects:
            return response.objects[0].properties
        return None

    def join_dataset_to_weaviate(
        self,
        weaviate_documents: List[Dict[str, Any]],
        dataset_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Join Weaviate documents with their corresponding dataset records.

        Args:
            weaviate_documents: List of Weaviate document properties
            dataset_fields: Specific fields to include from dataset (None = all fields)

        Returns:
            List of documents enriched with dataset fields
        """
        if self._dataset_index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        enriched_docs = []
        for weaviate_doc in weaviate_documents:
            doc_id = weaviate_doc.get("document_id")
            doc_number = weaviate_doc.get("document_number")

            dataset_record = self.get_dataset_record(document_id=doc_id, document_number=doc_number)

            enriched_doc = weaviate_doc.copy()
            if dataset_record:
                if dataset_fields:
                    # Only add specified fields
                    for field in dataset_fields:
                        if field in dataset_record:
                            enriched_doc[f"dataset_{field}"] = dataset_record[field]
                else:
                    # Add all dataset fields with prefix
                    for key, value in dataset_record.items():
                        enriched_doc[f"dataset_{key}"] = value

            enriched_docs.append(enriched_doc)

        return enriched_docs

    def update_raw_content_from_dataset(
        self,
        raw_content_field: str = "full_text",
        batch_size: int = 100,
        document_id_field: str = "judgment_id",
    ) -> int:
        """
        Update Weaviate documents with raw_content from the dataset.

        Args:
            raw_content_field: Field in dataset containing raw text
            batch_size: Number of documents to process per batch
            document_id_field: Field name for document ID in dataset

        Returns:
            Number of documents updated
        """
        if self._dataset_index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        collection = self.db.legal_documents_collection
        updated_count = 0

        # Fetch all documents from Weaviate
        logger.info("Fetching Weaviate documents...")
        response = collection.query.fetch_objects(limit=10000)  # Adjust limit as needed

        logger.info(f"Processing {len(response.objects)} documents...")

        # Process in batches
        for i in range(0, len(response.objects), batch_size):
            batch = response.objects[i : i + batch_size]

            with collection.batch.dynamic() as batch_context:
                for obj in batch:
                    props = obj.properties
                    doc_id = props.get("document_id")
                    doc_number = props.get("document_number")

                    # Get dataset record
                    dataset_record = self.get_dataset_record(
                        document_id=doc_id, document_number=doc_number
                    )

                    if dataset_record and raw_content_field in dataset_record:
                        raw_content = dataset_record[raw_content_field]

                        # Update document with raw_content
                        batch_context.update(
                            uuid=obj.uuid,
                            properties={"raw_content": raw_content},
                        )
                        updated_count += 1

            logger.info(f"Updated {updated_count} documents so far...")

        logger.info(f"Completed. Updated {updated_count} documents total.")
        return updated_count

    def get_missing_raw_content_documents(self) -> List[Dict[str, Any]]:
        """
        Find Weaviate documents that are missing raw_content.

        Returns:
            List of document properties missing raw_content
        """
        collection = self.db.legal_documents_collection

        # Query documents where raw_content is null or empty
        response = collection.query.fetch_objects(
            filters=weaviate.classes.query.Filter.by_property("raw_content").is_none(True),
            limit=10000,
        )

        return [obj.properties for obj in response.objects]
