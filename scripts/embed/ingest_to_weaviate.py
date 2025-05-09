import gc
import hashlib
import math
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from juddges.config import EmbeddingConfig
from juddges.data.config import IngestConfig
from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase
from juddges.data.ingesters import ChunkIngester, DocumentIngester
from juddges.data.schemas import DocumentChunk, DocumentType, LegalDocument
from juddges.settings import CONFIG_PATH, ROOT_PATH
from juddges.utils.config import resolve_config
from juddges.utils.date_utils import process_judgment_dates
from juddges.utils.misc import parse_true_string

logger.info(f"Environment variables loaded from {ROOT_PATH / '.env'} file")
load_dotenv(ROOT_PATH / ".env", override=True)

logger.info(f"WEAVIATE_HOST: {os.getenv('WEAVIATE_HOST')}")
logger.info(f"WEAVIATE_PORT: {os.getenv('WEAVIATE_PORT')}")
logger.info(f"WEAVIATE_GRPC_PORT: {os.getenv('WEAVIATE_GRPC_PORT')}")

# Default values for normal operation
DEFAULT_INGEST_BATCH_SIZE = 32
DEFAULT_UPSERT = True

# Debug mode configuration
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
if DEBUG:
    # Debug defaults for faster testing
    DEFAULT_INGEST_BATCH_SIZE = 2
    DEFAULT_MAX_DOCUMENTS = 5
    DEFAULT_PROCESSING_PROC = 1
    DEFAULT_INGEST_PROC = 1
    logger.info("Running in DEBUG mode with reduced processing values")
else:
    # Production defaults
    DEFAULT_MAX_DOCUMENTS = None
    DEFAULT_PROCESSING_PROC = 4  # max(1, multiprocessing.cpu_count() - 2)
    DEFAULT_INGEST_PROC = 2  # max(1, int(DEFAULT_PROCESSING_PROC / 2))


def generate_deterministic_uuid(document_id: str, chunk_id: str = None) -> str:
    """
    Generate a deterministic UUID for a document or chunk.

    This ensures that if ingestion is interrupted and restarted, the same object
    will receive the same UUID, allowing for proper deduplication.

    Args:
        document_id: The document ID (judgment_id)
        chunk_id: The chunk ID (if generating UUID for a chunk)

    Returns:
        A deterministic UUID as string
    """
    if chunk_id:
        # For chunks, combine document_id and chunk_id
        key = f"{document_id}_{chunk_id}"
    else:
        # For documents, use only document_id
        key = document_id

    # Create SHA-256 hash and use the first 32 chars (16 bytes) for UUID generation
    hash_obj = hashlib.sha256(key.encode("utf-8"))
    hex_digest = hash_obj.hexdigest()

    # Format as UUID (8-4-4-4-12 format)
    uuid = f"{hex_digest[:8]}-{hex_digest[8:12]}-{hex_digest[12:16]}-{hex_digest[16:20]}-{hex_digest[20:32]}"

    return uuid


@dataclass
class IngestConfig:
    """Configuration for ingestion process."""

    batch_size: int = DEFAULT_INGEST_BATCH_SIZE
    upsert: bool = DEFAULT_UPSERT
    max_documents: Optional[int] = DEFAULT_MAX_DOCUMENTS
    processing_proc: int = DEFAULT_PROCESSING_PROC
    ingest_proc: int = DEFAULT_INGEST_PROC


class CollectionIngester(ABC):
    """Base class for Weaviate collection ingesters."""

    def __init__(
        self,
        collection_name: str,
        db: Optional[WeaviateLegalDocumentsDatabase] = None,
        config: Optional[IngestConfig] = None,
    ):
        self.collection_name = collection_name
        self.db = db
        self.config = config or IngestConfig()

    def ingest(self, dataset: Dataset) -> None:
        """
        Ingest the dataset into Weaviate.

        Args:
            dataset: The dataset to ingest
        """
        # Limit dataset size if max_documents is specified
        if self.config.max_documents is not None:
            logger.info(f"Limiting ingestion to first {self.config.max_documents} documents")
            dataset = dataset.select(range(min(self.config.max_documents, dataset.num_rows)))

        num_rows = dataset.num_rows

        # Create database instance if not provided
        should_close_db = False
        if self.db is None:
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
                return

            total_batches = math.ceil(dataset.num_rows / self.config.batch_size)
            logger.info(f"Processing {total_batches} batches with size {self.config.batch_size}")

            self._process_batches(dataset, total_batches)

            final_count = self.db.get_collection_size(collection)
            logger.info(f"Final number of objects in collection: {final_count}")
            logger.info(f"Added {final_count - initial_count} new objects")

        finally:
            if should_close_db:
                self.db.__exit__(None, None, None)
                self.db = None

    def _filter_existing_objects(self, dataset: Dataset, collection: Any, num_rows: int) -> Dataset:
        """Filter out objects that already exist in the collection."""
        # This is implemented by subclasses
        return dataset

    def _process_batches(self, dataset: Dataset, total_batches: int) -> None:
        """Process all batches from the dataset."""
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
        """Process a single batch of the dataset."""
        pass


class ChunkIngester(CollectionIngester):
    """Ingester for document chunks collection."""

    def __init__(
        self,
        db: Optional[WeaviateLegalDocumentsDatabase] = None,
        config: Optional[IngestConfig] = None,
    ):
        super().__init__(
            collection_name=WeaviateLegalDocumentsDatabase.DOCUMENT_CHUNKS_COLLECTION,
            db=db,
            config=config,
        )

    def _filter_existing_objects(self, dataset: Dataset, collection: Any, num_rows: int) -> Dataset:
        """Filter out chunks that already exist in the collection."""
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
                for jid, cid in zip(batch["judgment_id"], batch["chunk_id"]):
                    yield generate_deterministic_uuid(jid, cid)

        # Filter out chunks that already exist in collection
        to_filter = set(get_chunk_uuids())
        already_exists = to_filter.intersection(uuids)
        logger.info(f"Found {len(already_exists)} existing chunks in database")

        # Filter dataset to include only new chunks
        def filter_chunks(item):
            return generate_deterministic_uuid(item["judgment_id"], item["chunk_id"]) not in uuids

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
        """Process and insert a batch of embeddings into Weaviate."""
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
                        batch["judgment_id"],
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
        """Validate batch data before processing."""
        try:
            assert len(batch["judgment_id"]) > 0, "Batch is empty"
            assert len(batch["judgment_id"]) == len(batch["chunk_id"]), (
                "Mismatched lengths between judgment_id and chunk_id"
            )
            assert len(batch["judgment_id"]) == len(batch["chunk_text"]), (
                "Mismatched lengths between judgment_id and chunk_text"
            )
            assert len(batch["judgment_id"]) == len(batch["embedding"]), (
                "Mismatched lengths between judgment_id and embedding"
            )

            # Validate embedding dimensions
            embedding_shape = np.array(batch["embedding"][0]).shape
            logger.debug(f"Embedding shape: {embedding_shape}")
            assert len(embedding_shape) == 1, f"Invalid embedding shape: {embedding_shape}"

            # Check for null values
            assert all(id_ is not None for id_ in batch["judgment_id"]), "Found null judgment_id"
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
        db: Optional[WeaviateLegalDocumentsDatabase] = None,
        config: Optional[IngestConfig] = None,
    ):
        super().__init__(
            collection_name=WeaviateLegalDocumentsDatabase.LEGAL_DOCUMENTS_COLLECTION,
            db=db,
            config=config,
        )

    def _filter_existing_objects(self, dataset: Dataset, collection: Any, num_rows: int) -> Dataset:
        """Filter out documents that already exist in the collection."""
        # Get existing UUIDs
        logger.info("Fetching existing UUIDs from collection...")
        uuids = set(self.db.get_uuids(collection))

        # Pre-generate UUIDs for all items in dataset for comparison
        logger.info("Generating deterministic UUIDs for all documents to filter...")

        def get_doc_uuids():
            for batch_idx in range(0, num_rows, self.config.batch_size):
                batch_end = min(batch_idx + self.config.batch_size, num_rows)
                batch = dataset[batch_idx:batch_end]
                for jid in batch["judgment_id"]:
                    yield generate_deterministic_uuid(jid)

        # Filter out documents that already exist in collection
        to_filter = set(get_doc_uuids())
        already_exists = to_filter.intersection(uuids)
        logger.info(f"Found {len(already_exists)} existing documents in database")

        # Filter dataset to include only new documents
        def filter_docs(item):
            return generate_deterministic_uuid(item["judgment_id"]) not in uuids

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
        """Process and insert a batch of documents into Weaviate."""
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
                for i in range(len(batch["judgment_id"])):
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
                    doc = LegalDocument(
                        document_id=batch["judgment_id"][i],
                        document_type=DocumentType.JUDGMENT,
                        # Add core fields
                        title=properties.get("title"),
                        date_issued=properties.get("date_issued"),
                        document_number=properties.get("document_number"),
                        language=properties.get("language", "pl"),
                        country=properties.get("country", "Poland"),
                        full_text=properties.get("full_text"),
                        summary=properties.get("summary"),
                        # Add nested objects if available
                        issuing_body=properties.get("issuing_body"),
                        segmentation_info=properties.get("segmentation_info"),
                        legal_references=properties.get("legal_references"),
                        legal_concepts=properties.get("legal_concepts"),
                        outcome=properties.get("outcome"),
                        judgment_specific=properties.get("judgment_specific"),
                        metadata=properties.get("metadata"),
                    ).dict(exclude_none=True)

                    # Add object to batch with vector and deterministic UUID
                    batch_op.add_object(
                        uuid=uuid,
                        properties=doc,
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


class DatasetLoader:
    """Utility class to load datasets for ingestion."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def _check_embeddings_exist(self) -> None:
        """Check if embeddings path exists."""
        if not Path(self.config.output_dir).exists():
            raise ValueError(f"Embeddings directory does not exist: {self.config.output_dir}")
        logger.info(f"Using embeddings from {self.config.output_dir}")

    def load_chunk_dataset(self) -> Dataset:
        """Load chunk embeddings dataset."""
        self._check_embeddings_exist()
        chunks_path = Path(self.config.output_dir) / self.config.CHUNK_EMBEDDINGS_DIR
        logger.info(f"Loading chunk dataset from {chunks_path}")
        return load_dataset("parquet", data_dir=chunks_path, split="train")

    def load_document_dataset(self) -> Dataset:
        """Load document embeddings dataset."""
        self._check_embeddings_exist()
        docs_path = Path(self.config.output_dir) / self.config.AGG_EMBEDDINGS_DIR
        logger.info(f"Loading document dataset from {docs_path}")
        return load_dataset("parquet", data_dir=docs_path, split="train")


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="embedding.yaml")
def main(cfg: DictConfig) -> None:
    """Run data ingestion into Weaviate."""
    # Resolve the configuration
    config = resolve_config(cfg, EmbeddingConfig)
    config = EmbeddingConfig(**config)
    logger.info(f"Configuration: {config}")

    # Handle command line overrides
    max_documents = config.max_documents
    if "max_documents" in cfg:
        max_documents = int(cfg.max_documents)
        logger.info(f"Overriding max_documents from command line: {max_documents}")

    upsert = config.upsert
    if "upsert" in cfg:
        upsert = parse_true_string(cfg.upsert)
        logger.info(f"Overriding upsert from command line: {upsert}")

    ingest_batch_size = config.ingest_batch_size
    if "ingest_batch_size" in cfg:
        ingest_batch_size = int(cfg.ingest_batch_size)
        logger.info(f"Overriding ingest_batch_size from command line: {ingest_batch_size}")

    # Create DatasetLoader and load datasets
    loader = DatasetLoader(config)

    # Configure ingestion
    ingest_config = IngestConfig(
        max_documents=max_documents,
        upsert=upsert,
        batch_size=ingest_batch_size,
    )

    # Initialize database for both ingesters to share
    with WeaviateLegalDocumentsDatabase() as db:
        # Ensure the collections exist
        db.create_collections()

        # Process document data
        try:
            document_dataset = loader.load_document_dataset()
            logger.info(f"Loaded document dataset with {document_dataset.num_rows} rows")

            # Ingest documents
            document_ingester = DocumentIngester(db=db, config=ingest_config)
            document_ingester.ingest(document_dataset)

            # Free up memory
            del document_dataset
            gc.collect()
        except Exception as e:
            logger.error(f"Error processing document data: {e}")
            logger.warning("Continuing with chunk ingestion...")

        # Process chunk data
        try:
            chunk_dataset = loader.load_chunk_dataset()
            logger.info(f"Loaded chunk dataset with {chunk_dataset.num_rows} rows")

            # Ingest chunks
            chunk_ingester = ChunkIngester(db=db, config=ingest_config)
            chunk_ingester.ingest(chunk_dataset)
        except Exception as e:
            logger.error(f"Error processing chunk data: {e}")

    logger.info("Ingestion completed")


if __name__ == "__main__":
    main()
