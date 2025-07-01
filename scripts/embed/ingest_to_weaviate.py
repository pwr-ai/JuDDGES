import gc
import hashlib
import json
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
from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase
from juddges.data.loaders import DATASET_COLUMN_MAPPINGS, remap_row
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

EMBEDDING_KEY = "embedding"


def generate_deterministic_uuid(
    document_id: str, chunk_id: Optional[str] = None
) -> str:
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
        columns_to_ingest: Optional[list[str]] = None,
    ):
        self.collection_name = collection_name
        self.db = db
        self.config = config or IngestConfig()
        self.columns_to_ingest = columns_to_ingest

    def ingest(self, dataset: Dataset, embedding_dataset: Dataset) -> None:
        """
        Ingest the dataset into Weaviate.

        Args:
            dataset: The document dataset
            embedding_dataset: The dataset with the chunk embeddings
        """
        # check if datasets contains "document_id" column
        if "document_id" not in dataset.column_names:
            raise ValueError("Dataset must contain 'document_id' column for ingestion")
        if "document_id" not in embedding_dataset.column_names:
            raise ValueError(
                "Embedding dataset must contain 'document_id' column for ingestion"
            )

        # Get set of valid document IDs from document dataset
        valid_document_ids = set(dataset["document_id"])
        logger.info(
            f"Found {len(valid_document_ids)} valid document IDs in document dataset"
        )

        if self.columns_to_ingest:
            logger.info(f"Selecting columns: {self.columns_to_ingest}")
            document_columns = list(
                set(dataset.column_names).intersection(set(self.columns_to_ingest))
            )
            logger.info(f"Document columns: {document_columns}")
            embedding_columns = list(
                set(embedding_dataset.column_names).intersection(
                    set(self.columns_to_ingest)
                )
            )
            logger.info(f"Embedding columns: {embedding_columns}")
            dataset = dataset.select_columns(document_columns)
            embedding_dataset = embedding_dataset.select_columns(embedding_columns)

        # Limit dataset size if max_documents is specified
        if self.config.max_documents is not None:
            logger.info(
                f"Limiting ingestion to first {self.config.max_documents} documents"
            )
            dataset = dataset.select(
                range(min(self.config.max_documents, dataset.num_rows))
            )

            # Update valid document IDs based on the filtered dataset
            valid_document_ids = set(dataset["document_id"])
            logger.info(
                f"Filtering embedding dataset to match {len(valid_document_ids)} document IDs"
            )

            # Filter embedding dataset to only include embeddings for documents in the dataset
            embedding_dataset = embedding_dataset.filter(
                lambda example: example["document_id"] in valid_document_ids,
                num_proc=self.config.processing_proc,
            )

        # num_rows = embedding_dataset.num_rows

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
                embedding_dataset = self._filter_existing_objects(
                    embedding_dataset, collection, embedding_dataset.num_rows
                )
            else:
                logger.info(
                    "upsert enabled - uploading all embeddings (automatically updating already uploaded)"
                )

            # If after filtering, there's nothing to do
            if embedding_dataset.num_rows == 0:
                logger.info("No new objects to insert, skipping.")
                return

            total_batches = math.ceil(
                embedding_dataset.num_rows / self.config.batch_size
            )
            logger.info(
                f"Processing {total_batches} batches with size {self.config.batch_size}"
            )

            self._process_batches(dataset, embedding_dataset, total_batches)

            final_count = self.db.get_collection_size(collection)
            logger.info(f"Final number of objects in collection: {final_count}")
            logger.info(f"Added {final_count - initial_count} new objects")

        finally:
            if should_close_db:
                self.db.__exit__(None, None, None)
                self.db = None

    def _filter_existing_objects(
        self, dataset: Dataset, collection: Any = None, num_rows: Optional[int] = None
    ) -> Dataset:
        """Filter out objects that already exist in the collection."""
        if self.db is None:
            raise ValueError(
                "Database connection is required for filtering existing objects"
            )

        if collection is None:
            collection = self.db.get_collection(self.collection_name)
        if num_rows is None:
            num_rows = dataset.num_rows

        # Get existing UUIDs from collection
        logger.info("Fetching existing UUIDs from collection...")
        uuids = set(self.db.get_uuids(collection))
        logger.info(f"Found {len(uuids)} existing objects in database")

        # Generate UUIDs for items in dataset to compare
        logger.info("Generating deterministic UUIDs for filtering...")

        def get_dataset_uuids():
            for batch_idx in range(0, num_rows, self.config.batch_size):
                batch_end = min(batch_idx + self.config.batch_size, num_rows)
                batch = dataset[batch_idx:batch_end]

                # Check if this is a chunk dataset (has chunk_id) or document dataset
                if "chunk_id" in batch:
                    # This is a chunk dataset
                    for doc_id, chunk_id in zip(
                        batch["document_id"], batch["chunk_id"]
                    ):
                        yield generate_deterministic_uuid(doc_id, chunk_id)
                else:
                    # This is a document dataset
                    for doc_id in batch["document_id"]:
                        yield generate_deterministic_uuid(doc_id)

        # Get set of UUIDs that would be generated from this dataset
        dataset_uuids = set(get_dataset_uuids())
        already_exists = dataset_uuids.intersection(uuids)
        logger.info(f"Found {len(already_exists)} objects already in database")

        # Filter dataset to include only new objects
        def filter_objects(item):
            if "chunk_id" in item:
                # Chunk filtering
                uuid = generate_deterministic_uuid(
                    item["document_id"], item["chunk_id"]
                )
            else:
                # Document filtering
                uuid = generate_deterministic_uuid(item["document_id"])
            return uuid not in uuids

        filtered_dataset = dataset.filter(
            filter_objects,
            num_proc=self.config.processing_proc,
            desc="Filtering out already uploaded objects",
        )

        logger.info(
            f"Filtered out {num_rows - filtered_dataset.num_rows} existing objects"
        )
        return filtered_dataset

    def _process_batches(
        self, dataset: Dataset, embedding_dataset: Dataset, total_batches: int
    ) -> None:
        """Process all batches from the dataset."""
        if self.config.processing_proc > 1:
            logger.info(
                f"Using parallel processing with {self.config.ingest_proc} workers"
            )
            with ThreadPoolExecutor(max_workers=self.config.ingest_proc) as executor:
                futures = []
                for batch_idx in range(total_batches):
                    future = executor.submit(
                        self.process_batch,
                        dataset=dataset,
                        embedding_dataset=embedding_dataset,
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
                    logger.info(
                        "Keyboard interrupt received. Cancelling remaining tasks..."
                    )
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
                    embedding_dataset=embedding_dataset,
                    batch_idx=batch_idx,
                )

    @abstractmethod
    def process_batch(
        self,
        dataset: Dataset,
        embedding_dataset: Dataset,
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
        columns_to_ingest: Optional[list[str]] = None,
    ):
        super().__init__(
            collection_name=WeaviateLegalDocumentsDatabase.DOCUMENT_CHUNKS_COLLECTION,
            db=db,
            config=config,
            columns_to_ingest=columns_to_ingest,
        )

    def _filter_existing_objects(
        self,
        dataset: Dataset,
        collection: Any,
        num_rows: int,
        document_ids: Optional[set] = None,
    ) -> Dataset:
        """
        Filter out chunks that already exist in the collection and those with document_ids not in the document dataset.

        Args:
            dataset: The dataset containing chunks
            collection: The Weaviate collection
            num_rows: Number of rows in the dataset
            document_ids: Set of valid document IDs from the document dataset

        Returns:
            Filtered dataset containing only new chunks for valid documents
        """
        if self.db is None:
            raise ValueError(
                "Database connection is required for filtering existing chunks"
            )

        # Get existing UUIDs
        logger.info("Fetching existing UUIDs from collection...")
        uuids = set(self.db.get_uuids(collection))

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

        # Filter dataset to include only new chunks and only for documents in our valid set
        def filter_chunks(item):
            # First check if document_id is in the valid set of document IDs
            if document_ids is not None and item["document_id"] not in document_ids:
                return False

            # Then check if the chunk is not already in the database
            return (
                generate_deterministic_uuid(item["document_id"], item["chunk_id"])
                not in uuids
            )

        filtered_dataset = dataset.filter(
            filter_chunks,
            num_proc=self.config.processing_proc,
            desc="Filtering chunks by document ID and removing already uploaded chunks",
        )

        if document_ids is not None:
            removed_invalid_docs = sum(
                1 for doc_id in dataset["document_id"] if doc_id not in document_ids
            )
            logger.info(
                f"Filtered out {removed_invalid_docs} chunks with invalid document IDs"
            )

        logger.info(
            f"Filtered out {num_rows - filtered_dataset.num_rows} chunks (existing or with invalid document IDs)"
        )
        return filtered_dataset

    def process_batch(
        self,
        dataset: Dataset,  # Document dataset
        embedding_dataset: Dataset,  # Chunk embedding dataset
        batch_idx: int,
    ) -> None:
        # Get the batch of chunks
        batch = embedding_dataset[
            batch_idx
            * self.config.batch_size : (batch_idx + 1)
            * self.config.batch_size
        ]

        # Build a lookup for document attributes by document_id
        doc_lookup = {doc_id: idx for idx, doc_id in enumerate(dataset["document_id"])}
        # List of document-level attributes to merge into chunk
        doc_attrs_keys = ["language", "country", "date_issued", "document_type"]

        try:
            if self.db is None:
                raise ValueError("Database connection is required for batch processing")
            collection = self.db.document_chunks_collection
            uuids_to_insert = []
            for doc_id, chunk_id in zip(batch["document_id"], batch["chunk_id"]):
                uuid = generate_deterministic_uuid(doc_id, chunk_id)
                uuids_to_insert.append(uuid)

            with collection.batch.fixed_size(
                batch_size=self.config.batch_size
            ) as batch_op:
                for i in range(len(batch["document_id"])):
                    doc_id = batch["document_id"][i]
                    chunk_id = batch["chunk_id"][i]
                    doc_idx = doc_lookup.get(doc_id)
                    doc_attrs = {
                        k: dataset[k][doc_idx]
                        for k in doc_attrs_keys
                        if doc_idx is not None and k in dataset.column_names
                    }
                    # Serialize JSON fields if needed
                    cited_references = (
                        batch.get("cited_references", [None])[i]
                        if "cited_references" in batch
                        else None
                    )
                    if cited_references is not None and not isinstance(
                        cited_references, str
                    ):
                        cited_references = json.dumps(cited_references)
                    tags = batch.get("tags", [None])[i] if "tags" in batch else None
                    if tags is not None and not isinstance(tags, str):
                        tags = json.dumps(tags)
                    chunk = DocumentChunk(
                        document_id=doc_id,
                        document_type=doc_attrs.get(
                            "document_type", DocumentType.JUDGMENT
                        ),
                        chunk_id=chunk_id,
                        chunk_text=batch["chunk_text"][i],
                        date_issued=doc_attrs.get("date_issued"),
                        publication_date=doc_attrs.get(
                            "date_issued"
                        ),  # Same as date_issued
                        segment_type=None,  # Optional field
                        position=(
                            batch.get("position", [None])[i]
                            if "position" in batch
                            else None
                        ),
                        confidence_score=None,  # Optional field
                        parent_segment_id=None,  # Optional field
                        cited_references=cited_references,
                        tags=tags,
                        language=doc_attrs.get("language"),
                        country=doc_attrs.get("country"),
                    ).dict(exclude_none=True)
                    batch_op.add_object(
                        uuid=uuids_to_insert[i],
                        properties=chunk,
                        vector={
                            "base": batch["embedding"][i],
                            "dev": batch["embedding"][i],
                            "fast": batch["embedding"][i],
                        },
                    )
            logger.info(
                "Successfully processed batch of chunks with merged document attributes"
            )
            gc.collect()
        except Exception as e:
            logger.error(f"Error processing chunk batch: {str(e)}")
            raise

    def _validate_batch(self, batch: dict[str, list[Any]]) -> bool:
        """Validate batch data before processing."""
        try:
            assert len(batch["judgment_id"]) > 0, "Batch is empty"
            assert len(batch["judgment_id"]) == len(
                batch["chunk_id"]
            ), "Mismatched lengths between judgment_id and chunk_id"
            assert len(batch["judgment_id"]) == len(
                batch["chunk_text"]
            ), "Mismatched lengths between judgment_id and chunk_text"
            assert len(batch["judgment_id"]) == len(
                batch["embedding"]
            ), "Mismatched lengths between judgment_id and embedding"

            # Validate embedding dimensions
            embedding_shape = np.array(batch["embedding"][0]).shape
            logger.debug(f"Embedding shape: {embedding_shape}")
            assert (
                len(embedding_shape) == 1
            ), f"Invalid embedding shape: {embedding_shape}"

            # Check for null values
            assert all(
                id_ is not None for id_ in batch["judgment_id"]
            ), "Found null judgment_id"
            assert all(
                chunk is not None for chunk in batch["chunk_text"]
            ), "Found null chunk_text"
            assert all(
                emb is not None for emb in batch["embedding"]
            ), "Found null embedding"

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
        columns_to_ingest: Optional[list[str]] = None,
        default_column_values: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            collection_name=WeaviateLegalDocumentsDatabase.LEGAL_DOCUMENTS_COLLECTION,
            db=db,
            config=config,
            columns_to_ingest=columns_to_ingest,
        )
        self.default_column_values = default_column_values or {}

    def _filter_existing_objects(
        self, dataset: Dataset, collection: Any, num_rows: int
    ) -> Dataset:
        """Filter out documents that already exist in the collection."""
        if self.db is None:
            raise ValueError(
                "Database connection is required for filtering existing documents"
            )

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

        logger.info(
            f"Filtered out {num_rows - filtered_dataset.num_rows} existing embeddings"
        )
        return filtered_dataset

    def process_batch(
        self,
        dataset: Dataset,
        embedding_dataset: Dataset,
        batch_idx: int,
    ) -> None:
        """Process and insert a batch of documents into Weaviate."""
        batch = dataset[
            batch_idx
            * self.config.batch_size : (batch_idx + 1)
            * self.config.batch_size
        ]

        try:
            if self.db is None:
                raise ValueError("Database connection is required for batch processing")
            collection = self.db.legal_documents_collection

            # Pre-generate UUIDs for this batch
            uuids_to_insert = []
            for jid in batch["document_id"]:
                uuid = generate_deterministic_uuid(jid)
                uuids_to_insert.append(uuid)

            # Use simple fixed-size batch
            with collection.batch.fixed_size(
                batch_size=self.config.batch_size
            ) as batch_op:
                for i in range(len(batch["document_id"])):
                    # Get deterministic UUID for this document
                    uuid = uuids_to_insert[i]

                    properties = {}
                    for key in self.columns_to_ingest or []:
                        if key == EMBEDDING_KEY:
                            continue
                        value = batch[key][i]
                        if (
                            value is None
                            or (isinstance(value, float) and np.isnan(value))
                        ) and key in self.default_column_values:
                            value = self.default_column_values[key]
                        properties[key] = value
                    # Add any default columns not present in the batch
                    for key, value in self.default_column_values.items():
                        if key not in properties or properties[key] is None:
                            properties[key] = value

                    properties = process_judgment_dates(properties)

                    # Create a LegalDocument instance with the available properties
                    doc = LegalDocument(
                        document_id=properties.get("document_id") or "",
                        document_type=properties.get("document_type") or "",
                        # Add core fields
                        title=properties.get("title"),
                        date_issued=properties.get("date_issued"),
                        publication_date=properties.get(
                            "date_issued"
                        ),  # Same as date_issued
                        document_number=properties.get("document_number"),
                        language=properties.get("language"),
                        country=properties.get("country"),
                        full_text=properties.get("full_text"),
                        summary=properties.get("summary"),
                        thesis=properties.get("thesis"),
                        issuing_body=properties.get("issuing_body"),
                        legal_references=properties.get("legal_references"),
                        legal_concepts=properties.get("legal_concepts"),
                        outcome=properties.get("outcome"),
                        parties=properties.get("parties"),
                        judgment_specific=properties.get("judgment_specific"),
                        tax_interpretation_specific=properties.get(
                            "tax_interpretation_specific"
                        ),
                        legal_act_specific=properties.get("legal_act_specific"),
                        relationships=properties.get("relationships"),
                        legal_analysis=properties.get("legal_analysis"),
                        structured_content=properties.get("structured_content"),
                        section_embeddings=properties.get("section_embeddings"),
                        metadata=properties.get("metadata"),
                    ).dict(exclude_none=True)

                    # get embedding from embedding_dataset
                    embedding = embedding_dataset.filter(
                        lambda x: x["document_id"] == batch["document_id"][i]
                    )
                    if len(embedding) == 0:
                        logger.warning(
                            f"No embedding found for document {batch['document_id'][i]}"
                        )
                        continue
                    embedding = embedding[0]["embedding"]

                    # Add object to batch with vector and deterministic UUID
                    batch_op.add_object(
                        uuid=uuid,
                        properties=doc,
                        vector={
                            "base": embedding,  # Primary embedding
                            "dev": embedding,  # Development embedding
                            "fast": embedding,  # Fast embedding
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
            raise ValueError(
                f"Embeddings directory does not exist: {self.config.output_dir}"
            )
        logger.info(f"Using embeddings from {self.config.output_dir}")

    def load_chunk_dataset(self) -> Dataset:
        """Load chunk embeddings dataset."""
        self._check_embeddings_exist()
        chunks_path = Path(self.config.output_dir) / self.config.CHUNK_EMBEDDINGS_DIR
        logger.info(f"Loading chunk dataset from {chunks_path}")
        dataset = load_dataset(
            "parquet",
            data_dir=str(chunks_path),
            split="train",
            num_proc=self.config.num_output_shards,
        )
        # Cast to Dataset type for type checker
        return dataset  # type: ignore

    def load_document_embeddings_dataset(self) -> Dataset:
        """Load document embeddings dataset, chunked embeddings are already aggregated."""
        self._check_embeddings_exist()
        docs_path = Path(self.config.output_dir) / self.config.AGG_EMBEDDINGS_DIR
        logger.info(f"Loading document embeddings dataset from {docs_path}")
        return load_dataset(  # type: ignore
            "parquet",
            data_dir=str(docs_path),
            split="train",
            num_proc=self.config.num_output_shards,
        )

    def load_document_dataset(self) -> Dataset:
        """Load document dataset, meaning the dataset with the original text, attributes, etc."""
        logger.info(f"Loading document dataset from {self.config.dataset_name}")
        return load_dataset(  # type: ignore
            self.config.dataset_name,
            split="train",
            num_proc=self.config.num_output_shards,
        )


@hydra.main(
    version_base="1.3", config_path=str(CONFIG_PATH), config_name="embedding.yaml"
)
def main(cfg: DictConfig) -> None:
    """Run data ingestion into Weaviate."""
    # Resolve the configuration
    config = resolve_config(cfg, resolve=True)
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
        logger.info(
            f"Overriding ingest_batch_size from command line: {ingest_batch_size}"
        )

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

        embedding_dataset = loader.load_document_embeddings_dataset()
        logger.info(
            f"Loaded document embeddings dataset with {embedding_dataset.num_rows} rows"
        )
        document_dataset = loader.load_document_dataset()
        logger.info(f"Loaded document dataset with {document_dataset.num_rows} rows")

        # Remap columns if mapping is available
        mapping = DATASET_COLUMN_MAPPINGS.get(config.dataset_name)

        if mapping is not None:
            document_dataset = document_dataset.map(
                lambda row: remap_row(row, mapping), num_proc=config.num_output_shards
            )
            embedding_dataset = embedding_dataset.map(
                lambda row: remap_row(row, mapping), num_proc=config.num_output_shards
            )

        # Ingest documents
        document_ingester = DocumentIngester(
            db=db,
            config=ingest_config,
            columns_to_ingest=(
                list(mapping.values()) + ["embedding"] if mapping else ["embedding"]
            ),
            default_column_values=config.default_column_values,
        )
        document_ingester.ingest(
            dataset=document_dataset, embedding_dataset=embedding_dataset
        )

        # Free up memory
        del embedding_dataset
        gc.collect()

        chunks_embeddings_dataset = loader.load_chunk_dataset()
        if mapping is not None:
            chunks_embeddings_dataset = chunks_embeddings_dataset.map(
                lambda row: remap_row(row, mapping), num_proc=config.num_output_shards
            )
        logger.info(
            f"Loaded chunk dataset with {chunks_embeddings_dataset.num_rows} rows"
        )

        chunk_ingester = ChunkIngester(
            db=db,
            config=ingest_config,
            columns_to_ingest=(
                list(mapping.keys())
                + ["embedding", "chunk_id", "chunk_text", "document_id"]
                if mapping
                else ["embedding", "chunk_id", "chunk_text", "document_id"]
            ),
        )
        chunk_ingester.ingest(
            dataset=document_dataset, embedding_dataset=chunks_embeddings_dataset
        )

    logger.info("Ingestion completed")


if __name__ == "__main__":
    main()
