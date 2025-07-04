"""
Collection ingester classes for Weaviate.
"""

import gc
import math
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from juddges.data.config import IngestConfig
from juddges.data.loaders import DATASET_COLUMN_MAPPINGS, remap_row
from juddges.data.schemas import DocumentChunk, DocumentType, LegalDocument
from juddges.data.utils import generate_deterministic_uuid
from juddges.utils.date_utils import process_judgment_dates

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase


class CollectionIngester(ABC):
    """Base class for Weaviate collection ingesters."""

    def __init__(
        self,
        collection_name: str,
        db: Optional["WeaviateLegalDocumentsDatabase"] = None,
        config: Optional[IngestConfig] = None,
        columns_to_ingest: list[str] = None,
    ):
        """
        Initialize a collection ingester.

        Args:
            collection_name: Name of the Weaviate collection to ingest into
            db: Optional database connection to reuse
            config: Optional ingestion configuration
        """
        self.collection_name = collection_name
        self.db = db
        self.config = config or IngestConfig()
        self.columns_to_ingest = columns_to_ingest

    def ingest(self, dataset: Dataset) -> int:
        """
        Ingest the dataset into Weaviate.

        Args:
            dataset: The dataset to ingest
            
        Returns:
            Number of documents actually added to the collection
        """
        # Remap columns if mapping is available
        dataset_name = getattr(self.config, "dataset_name", None)
        mapping = DATASET_COLUMN_MAPPINGS.get(dataset_name)
        if mapping:
            dataset = dataset.map(lambda row: remap_row(row, mapping))

        # Limit dataset size if max_documents is specified
        if self.config.max_documents is not None:
            logger.info(f"Limiting ingestion to first {self.config.max_documents} documents")
            dataset = dataset.select(range(min(self.config.max_documents, dataset.num_rows)))

        num_rows = dataset.num_rows

        # Create database instance if not provided
        should_close_db = False
        if self.db is None:
            # Import here to avoid circular import
            from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase

            self.db = WeaviateLegalDocumentsDatabase()
            should_close_db = True
            self.db.__enter__()

        try:
            collection = self.db.get_collection(self.collection_name)
            initial_count = self.db.get_collection_size(collection)
            logger.info(f"Initial number of objects in collection: {initial_count}")

            if not self.config.upsert:
                logger.info("upsert disabled - uploading only new embeddings")
                dataset = self._filter_existing_objects(dataset, collection, num_rows)
            else:
                logger.info(
                    "upsert enabled - uploading all embeddings (automatically updating already uploaded)"
                )

            # If after filtering, there's nothing to do
            if dataset.num_rows == 0:
                logger.info("No new objects to insert, skipping.")
                return 0

            total_batches = math.ceil(dataset.num_rows / self.config.batch_size)
            logger.info(f"Processing {total_batches} batches with size {self.config.batch_size}")

            self._process_batches(dataset, total_batches)

            final_count = self.db.get_collection_size(collection)
            added_count = final_count - initial_count
            logger.info(f"Final number of objects in collection: {final_count}")
            logger.info(f"Added {added_count} new objects")
            
            return added_count

        finally:
            if should_close_db:
                self.db.__exit__(None, None, None)
                self.db = None

    def _filter_existing_objects(self, dataset: Dataset, collection: Any, num_rows: int) -> Dataset:
        """
        Filter out objects that already exist in the collection.

        Args:
            dataset: Dataset to filter
            collection: Weaviate collection to check against
            num_rows: Number of rows in the dataset

        Returns:
            Filtered dataset
        """
        # This is implemented by subclasses
        return dataset

    def _process_batches(self, dataset: Dataset, total_batches: int) -> None:
        """
        Process all batches from the dataset.

        Args:
            dataset: Dataset to process
            total_batches: Total number of batches
        """
        if self.config.processing_proc > 1:
            logger.info(f"Using parallel processing with {self.config.ingest_proc} workers")
            with ThreadPoolExecutor(max_workers=self.config.ingest_proc) as executor:
                futures = []
                for batch_idx in range(total_batches):
                    future = executor.submit(
                        self.process_batch,
                        dataset=dataset,
                        batch_idx=batch_idx,
                    )
                    futures.append(future)

                try:
                    # Process futures in chunks to avoid memory issues
                    chunk_size = self.config.ingest_proc
                    for i in range(0, len(futures), chunk_size):
                        chunk = futures[i : i + chunk_size]
                        for future in as_completed(chunk):
                            try:
                                future.result()
                            except Exception as e:
                                logger.error(f"Batch processing failed: {str(e)}")
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received. Cancelling remaining tasks...")
                    for future in futures:
                        future.cancel()
                    raise
        else:
            logger.info("Using sequential processing")
            for batch_idx in tqdm(
                range(total_batches),
                total=total_batches,
                desc="Uploading batches sequentially",
            ):
                self.process_batch(
                    dataset=dataset,
                    batch_idx=batch_idx,
                )

    @abstractmethod
    def process_batch(
        self,
        dataset: Dataset,
        batch_idx: int,
    ) -> None:
        """
        Process a single batch of the dataset.

        Args:
            dataset: Dataset to process
            batch_idx: Index of the batch to process
        """
        pass


class ChunkIngester(CollectionIngester):
    """Ingester for document chunks collection."""

    def __init__(
        self,
        db: Optional["WeaviateLegalDocumentsDatabase"] = None,
        config: Optional[IngestConfig] = None,
    ):
        """
        Initialize a chunk ingester.

        Args:
            db: Optional database connection to reuse
            config: Optional ingestion configuration
        """
        # Import here to avoid circular import
        from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase

        super().__init__(
            collection_name=WeaviateLegalDocumentsDatabase.DOCUMENT_CHUNKS_COLLECTION,
            db=db,
            config=config,
        )

    def _filter_existing_objects(self, dataset: Dataset, collection: Any, num_rows: int) -> Dataset:
        """
        Filter out chunks that already exist in the collection.

        Args:
            dataset: Dataset to filter
            collection: Weaviate collection to check against
            num_rows: Number of rows in the dataset

        Returns:
            Filtered dataset containing only new chunks
        """
        # Get existing UUIDs
        logger.info("Fetching existing UUIDs from collection...")
        uuids = set(self.db.get_uuids(collection))

        # Pre-generate UUIDs for all items in dataset for comparison
        logger.info("Generating deterministic UUIDs for all chunks to filter...")

        # Use a generator function to avoid memory issues with large datasets
        def get_chunk_uuids():
            for batch_idx in range(0, num_rows, self.config.batch_size):
                batch_end = min(batch_idx + self.config.batch_size, num_rows)
                batch = dataset[batch_idx:batch_end]
                for jid, cid in zip(batch["document_id"], batch["chunk_id"]):
                    yield generate_deterministic_uuid(jid, cid)

        # Filter out chunks that already exist in collection
        to_filter = set(get_chunk_uuids())
        already_exists = to_filter.intersection(uuids)
        logger.info(f"Found {len(already_exists)} existing chunks in database")

        # Filter dataset to include only new chunks
        def filter_chunks(item):
            return generate_deterministic_uuid(item["document_id"], item["chunk_id"]) not in uuids

        filtered_dataset = dataset.filter(
            filter_chunks,
            num_proc=self.config.processing_proc,
            desc="Filtering out already uploaded chunks",
        )

        logger.info(f"Filtered out {num_rows - filtered_dataset.num_rows} existing embeddings")
        return filtered_dataset

    def process_batch(
        self,
        dataset: Dataset,
        batch_idx: int,
    ) -> None:
        """
        Process and insert a batch of embeddings into Weaviate.

        Args:
            dataset: Dataset containing chunks to process
            batch_idx: Index of the batch to process
        """
        batch = dataset[
            batch_idx * self.config.batch_size : (batch_idx + 1) * self.config.batch_size
        ]
        try:
            if not self._validate_batch(batch):
                logger.error("Skipping invalid batch")
                return

            collection = self.db.document_chunks_collection

            # Pre-generate all UUIDs for this batch
            uuids_to_insert = []
            for jid, cid in zip(batch["judgment_id"], batch["chunk_id"]):
                uuid = generate_deterministic_uuid(jid, cid)
                uuids_to_insert.append(uuid)

            # Use simple fixed-size batch
            with collection.batch.fixed_size(batch_size=self.config.batch_size) as batch_op:
                for idx, (jid, cid, text, emb) in enumerate(
                    zip(
                        batch["document_id"],
                        batch["chunk_id"],
                        batch["chunk_text"],
                        batch["embedding"],
                    )
                ):
                    # Generate deterministic UUID for this chunk
                    uuid = uuids_to_insert[idx]

                    # Create chunk properties using schema-defined fields
                    chunk = DocumentChunk(
                        document_id=jid,
                        document_type=DocumentType.JUDGMENT,
                        chunk_id=cid,
                        chunk_text=text,
                        # Add optional fields if available in your batch
                        segment_type=batch.get("segment_type", [None])[0],
                        position=batch.get("position", [None])[0],
                        confidence_score=batch.get("confidence_score", [None])[0],
                        cited_references=batch.get("cited_references", [None])[0],
                        tags=batch.get("tags", [None])[0],
                        parent_segment_id=batch.get("parent_segment_id", [None])[0],
                        section_heading=batch.get("section_heading", [None])[0],
                        start_char_index=batch.get("start_char_index", [None])[0],
                        end_char_index=batch.get("end_char_index", [None])[0],
                    ).dict(exclude_none=True)

                    # Add object to batch with vector and deterministic UUID
                    batch_op.add_object(
                        uuid=uuid,
                        properties=chunk,
                        vector={
                            "base": emb,  # Primary embedding
                            "dev": emb,  # Development embedding
                            "fast": emb,  # Fast embedding
                        },
                    )

            logger.info("Successfully processed batch of chunks")
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise

    def _validate_batch(self, batch: dict[str, list[Any]]) -> bool:
        """
        Validate batch data before processing.

        Args:
            batch: Batch data to validate

        Returns:
            True if batch is valid, False otherwise
        """
        try:
            assert len(batch["document_id"]) > 0, "Batch is empty"
            assert len(batch["document_id"]) == len(batch["chunk_id"]), (
                "Mismatched lengths between document_id and chunk_id"
            )
            assert len(batch["document_id"]) == len(batch["chunk_text"]), (
                "Mismatched lengths between document_id and chunk_text"
            )
            assert len(batch["document_id"]) == len(batch["embedding"]), (
                "Mismatched lengths between document_id and embedding"
            )

            # Validate embedding dimensions
            embedding_shape = np.array(batch["embedding"][0]).shape
            logger.debug(f"Embedding shape: {embedding_shape}")
            assert len(embedding_shape) == 1, f"Invalid embedding shape: {embedding_shape}"

            # Check for null values
            assert all(id_ is not None for id_ in batch["document_id"]), "Found null document_id"
            assert all(chunk is not None for chunk in batch["chunk_text"]), "Found null chunk_text"
            assert all(emb is not None for emb in batch["embedding"]), "Found null embedding"

            return True
        except AssertionError as e:
            logger.error(f"Batch validation failed: {str(e)}")
            logger.debug(f"Batch contents: {batch}")
            return False


class DocumentIngester(CollectionIngester):
    """Ingester for legal documents collection."""

    def __init__(
        self,
        db: Optional["WeaviateLegalDocumentsDatabase"] = None,
        config: Optional[IngestConfig] = None,
    ):
        """
        Initialize a document ingester.

        Args:
            db: Optional database connection to reuse
            config: Optional ingestion configuration
        """
        # Import here to avoid circular import
        from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase

        super().__init__(
            collection_name=WeaviateLegalDocumentsDatabase.LEGAL_DOCUMENTS_COLLECTION,
            db=db,
            config=config,
        )

    def _filter_existing_objects(self, dataset: Dataset, collection: Any, num_rows: int) -> Dataset:
        """
        Filter out documents that already exist in the collection.

        Args:
            dataset: Dataset to filter
            collection: Weaviate collection to check against
            num_rows: Number of rows in the dataset

        Returns:
            Filtered dataset containing only new documents
        """
        # Get existing UUIDs
        logger.info("Fetching existing UUIDs from collection...")
        uuids = set(self.db.get_uuids(collection))

        # Pre-generate UUIDs for all items in dataset for comparison
        logger.info("Generating deterministic UUIDs for all documents to filter...")

        def get_doc_uuids():
            for batch_idx in range(0, num_rows, self.config.batch_size):
                batch_end = min(batch_idx + self.config.batch_size, num_rows)
                batch = dataset[batch_idx:batch_end]
                for jid in batch["document_id"]:
                    yield generate_deterministic_uuid(jid)

        # Filter out documents that already exist in collection
        to_filter = set(get_doc_uuids())
        already_exists = to_filter.intersection(uuids)
        logger.info(f"Found {len(already_exists)} existing documents in database")

        # Filter dataset to include only new documents
        def filter_docs(item):
            return generate_deterministic_uuid(item["document_id"]) not in uuids

        filtered_dataset = dataset.filter(
            filter_docs,
            num_proc=self.config.processing_proc,
            desc="Filtering out already uploaded documents",
        )

        logger.info(f"Filtered out {num_rows - filtered_dataset.num_rows} existing embeddings")
        return filtered_dataset

    def process_batch(
        self,
        dataset: Dataset,
        batch_idx: int,
    ) -> None:
        """
        Process and insert a batch of documents into Weaviate.

        Args:
            dataset: Dataset containing documents to process
            batch_idx: Index of the batch to process
        """
        batch = dataset[
            batch_idx * self.config.batch_size : (batch_idx + 1) * self.config.batch_size
        ]

        try:
            collection = self.db.legal_documents_collection

            # Pre-generate UUIDs for this batch
            uuids_to_insert = []
            for jid in batch["judgment_id"]:
                uuid = generate_deterministic_uuid(jid)
                uuids_to_insert.append(uuid)

            # Use simple fixed-size batch
            with collection.batch.fixed_size(batch_size=self.config.batch_size) as batch_op:
                for i in range(len(batch["document_id"])):
                    # Get deterministic UUID for this document
                    uuid = uuids_to_insert[i]

                    # Get all available properties from the batch that match the schema
                    property_names = set(self.db.legal_documents_properties).intersection(
                        batch.keys()
                    )
                    missing_properties = set(self.db.legal_documents_properties).difference(
                        batch.keys()
                    )
                    logger.warning(
                        f"Found missing properties compared to weaviate schema: {missing_properties}, "
                        f"uploading only {property_names}"
                    )
                    properties = {key: batch[key][i] for key in property_names}
                    properties = process_judgment_dates(properties)

                    # Create a LegalDocument instance with the available properties
                    # doc = LegalDocument(
                    #     document_id=batch["document_id"][i],
                    #     document_type=DocumentType.JUDGMENT,
                    #     # Add core fields
                    #     title=properties.get("title"),
                    #     date_issued=properties.get("date_issued"),
                    #     document_number=properties.get("document_number"),
                    #     language=properties.get("language", "pl"),
                    #     country=properties.get("country", "Poland"),
                    #     full_text=properties.get("full_text"),
                    #     summary=properties.get("summary"),
                    #     # Add nested objects if available
                    #     issuing_body=properties.get("issuing_body"),
                    #     segmentation_info=properties.get("segmentation_info"),
                    #     legal_references=properties.get("legal_references"),
                    #     legal_concepts=properties.get("legal_concepts"),
                    #     outcome=properties.get("outcome"),
                    #     judgment_specific=properties.get("judgment_specific"),
                    #     metadata=properties.get("metadata"),
                    # ).dict(exclude_none=True)

                    # Add object to batch with vector and deterministic UUID
                    batch_op.add_object(
                        uuid=uuid,
                        properties=properties,
                        vector={
                            "base": batch["embedding"][i],  # Primary embedding
                            "dev": batch["embedding"][i],  # Development embedding
                            "fast": batch["embedding"][i],  # Fast embedding
                        },
                    )

            logger.debug("Successfully processed batch of documents")
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise
