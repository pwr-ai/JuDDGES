import gc
import math
import multiprocessing
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pformat
from typing import Any, Callable, Dict, List

import hydra
import numpy as np
import polars as pl
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
import weaviate.exceptions
from juddges.config import EmbeddingConfig
from juddges.settings import CONFIG_PATH, ROOT_PATH
from juddges.utils.config import resolve_config
from juddges.utils.date_utils import process_judgment_dates
from juddges.utils.misc import parse_true_string
from weaviate.util import generate_uuid5
from juddges.data.schemas import DocumentChunk, DocumentType, LegalDocument

DEFAULT_INGEST_BATCH_SIZE = 64
DEFAULT_UPSERT = True


class EnhancedWeaviateDB:
    """Enhanced Weaviate database client for document ingestion."""

    LEGAL_DOCUMENTS_COLLECTION = "legal_documents"
    DOCUMENT_CHUNKS_COLLECTION = "document_chunks"

    def __init__(self, host: str, port: str, grpc_port: str, api_key: str | None = None):
        """Initialize the Weaviate client.
        
        Args:
            host: Weaviate host
            port: Weaviate HTTP port
            grpc_port: Weaviate gRPC port
            api_key: Optional API key for authentication
        """
        # Configure timeouts for better reliability
        timeout_config = Timeout(
            init=30,    # 30 seconds for initialization
            query=60,   # 60 seconds for queries
            insert=120  # 120 seconds for inserts
        )

        # Use connect_to_custom for more control over connection parameters
        self.client = weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=False,  # Set to True if using HTTPS
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=False,  # Set to True if using secure gRPC
            auth_credentials=Auth.api_key(api_key) if api_key else None,
            additional_config=AdditionalConfig(timeout=timeout_config)
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()

    def get_collection(self, collection_name: str) -> weaviate.collections.Collection:
        """Get a collection by name."""
        return self.client.collections.get(collection_name)

    def get_collection_size(self, collection: weaviate.collections.Collection) -> int:
        """Get the number of objects in a collection."""
        try:
            return collection.aggregate.over_all().total_count
        except Exception as e:
            logger.error(f"Error getting collection size: {str(e)}")
            raise

    def get_uuids(self, collection: weaviate.collections.Collection) -> List[str]:
        """Get all UUIDs from a collection."""
        try:
            response = collection.query.fetch_objects(include_vector=False).objects
            return [obj.uuid for obj in response]
        except Exception as e:
            logger.error(f"Error getting UUIDs: {str(e)}")
            raise

    @property
    def legal_documents_collection(self) -> weaviate.collections.Collection:
        return self.get_collection(self.LEGAL_DOCUMENTS_COLLECTION)

    @property
    def document_chunks_collection(self) -> weaviate.collections.Collection:
        return self.get_collection(self.DOCUMENT_CHUNKS_COLLECTION)

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


def do_ingest(
    dataset: Dataset,
    collection_name: str,
    process_batch_func: Callable[[EnhancedWeaviateDB, Dataset, int, int], None],
    batch_size: int,
    upsert: bool,
) -> None:
    num_rows = dataset.num_rows

    with EnhancedWeaviateDB(WV_HOST, WV_PORT, WV_GRPC_PORT, WV_API_KEY) as db:
        collection = db.get_collection(collection_name)
        initial_count = db.get_collection_size(collection)
        logger.info(f"Initial number of objects in collection: {initial_count}")

        if not upsert:
            logger.info("upsert disabled - uploading only new embeddings")
            uuids = set(db.get_uuids(collection))
            logger.info(f"Found {len(uuids)} existing UUIDs in database")

            dataset = dataset.filter(
                lambda item: item["uuid"] not in uuids,
                num_proc=PROCESSING_PROC,
                desc="Filtering out already uploaded embeddings",
            )
            logger.info(f"Filtered out {num_rows - dataset.num_rows} existing embeddings")
        else:
            logger.info(
                "upsert enabled - uploading all embeddings (automatically updating already uploaded)"
            )

        total_batches = math.ceil(num_rows / batch_size)
        logger.info(f"Processing {total_batches} batches with size {batch_size}")

        if PROCESSING_PROC > 1:
            logger.info(f"Using parallel processing with {INGEST_PROC} workers")
            with ThreadPoolExecutor(max_workers=INGEST_PROC) as executor:
                futures = []
                for batch_idx in range(total_batches):
                    future = executor.submit(
                        process_batch_func,
                        db=db,
                        dataset=dataset,
                        batch_idx=batch_idx,
                        batch_size=batch_size,
                    )
                    futures.append(future)
                
                try:
                    # Process futures in chunks to avoid memory issues
                    chunk_size = INGEST_PROC
                    for i in range(0, len(futures), chunk_size):
                        chunk = futures[i:i + chunk_size]
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
                process_batch_func(
                    db=db,
                    dataset=dataset,
                    batch_idx=batch_idx,
                    batch_size=batch_size,
                )

        final_count = db.get_collection_size(collection)
        logger.info(f"Final number of objects in collection: {final_count}")
        logger.info(f"Added {final_count - initial_count} new objects")


def process_batch_of_chunks(
    db: EnhancedWeaviateDB,
    dataset: Dataset,
    batch_idx: int,
    batch_size: int,
) -> None:
    """Process and insert a batch of embeddings into Weaviate."""
    batch = dataset[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    try:
        if not validate_batch_of_chunks(batch):
            logger.error("Skipping invalid batch")
            return

        collection = db.document_chunks_collection
        
        # Use simple fixed-size batch
        with collection.batch.fixed_size(batch_size=batch_size) as batch_op:
            for jid, cid, text, emb in zip(
                batch["judgment_id"],
                batch["chunk_id"],
                batch["chunk_text"],
                batch["embedding"],
            ):
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

                # Add object to batch with vector
                batch_op.add_object(
                    properties=chunk,
                    vector={
                        "base": emb,  # Primary embedding
                        "dev": emb,   # Development embedding
                        "fast": emb,  # Fast embedding
                    }
                )

        logger.info(f"Successfully processed batch of chunks")
        gc.collect()

    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise


def process_batch_of_documents(
    db: EnhancedWeaviateDB,
    dataset: Dataset,
    batch_idx: int,
    batch_size: int,
) -> None:
    """Process and insert a batch of documents into Weaviate."""
    batch = dataset[batch_idx * batch_size : (batch_idx + 1) * batch_size]

    try:
        collection = db.legal_documents_collection
        
        # Use simple fixed-size batch
        with collection.batch.fixed_size(batch_size=batch_size) as batch_op:
            for i in range(len(batch["judgment_id"])):
                # Get all available properties from the batch that match the schema
                property_names = set(db.legal_documents_properties).intersection(batch.keys())
                missing_properties = set(db.legal_documents_properties).difference(batch.keys())
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
                
                # Add object to batch with vector
                batch_op.add_object(
                    properties=doc,
                    vector={
                        "base": batch["embedding"][i],  # Primary embedding
                        "dev": batch["embedding"][i],   # Development embedding
                        "fast": batch["embedding"][i],  # Fast embedding
                    }
                )

        logger.debug(f"Successfully processed batch of documents")
        gc.collect()

    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise


def validate_batch_of_chunks(batch: dict[str, list[Any]]) -> bool:
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


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="embedding.yaml")
def main(cfg: DictConfig) -> None:
    cfg_dict = resolve_config(cfg)

    ingest_batch_size = cfg_dict.pop("ingest_batch_size", DEFAULT_INGEST_BATCH_SIZE)
    upsert = parse_true_string(cfg_dict.pop("upsert", DEFAULT_UPSERT))
    logger.debug(f"Using batch size: {ingest_batch_size}")
    logger.debug(f"Using upsert: {upsert}")

    logger.info(f"config:\n{pformat(cfg_dict)}")
    config = EmbeddingConfig(**cfg_dict)

    _check_embeddings_exist(config)
    logger.info(f"Starting ingestion of {config.output_dir} to Weaviate...")

    logger.info("Loading datasets...")
    chunk_ds = load_dataset(
        "parquet",
        data_dir=config.chunk_embeddings_dir,
        num_proc=PROCESSING_PROC,
    )

    ds_polars = pl.scan_parquet(f"hf://datasets/{config.dataset_name}/data/*.parquet")
    agg_ds_polars = pl.scan_parquet(f"{config.agg_embeddings_dir}/*.parquet")

    logger.info("Preparing aggregated dataset (it may take a few minutes)...")
    # TODO: Now we use temporary parquet file to avoid OOM conversion to hf.Dataset
    # the pipeline might be rewriten to be intire in polars instead
    with tempfile.NamedTemporaryFile(suffix=".parquet") as temp_file:
        # Stream query result directly to parquet file
        agg_ds_polars.join(ds_polars, on="judgment_id", how="left").sink_parquet(temp_file.name)

        # Load the temporary parquet file using HF datasets
        agg_ds = load_dataset(
            "parquet",
            data_files=temp_file.name,
            num_proc=PROCESSING_PROC,
        )["train"]

    logger.info("Done!")
    chunk_ds = chunk_ds["train"]
    
    # Run ingestion
    do_ingest(
        dataset=chunk_ds,
        collection_name=EnhancedWeaviateDB.DOCUMENT_CHUNKS_COLLECTION,
        process_batch_func=process_batch_of_chunks,
        batch_size=ingest_batch_size,
        upsert=True,
    )
    del chunk_ds
    gc.collect()

    do_ingest(
        dataset=agg_ds,
        collection_name=EnhancedWeaviateDB.LEGAL_DOCUMENTS_COLLECTION,
        process_batch_func=process_batch_of_documents,
        batch_size=ingest_batch_size,
        upsert=upsert,
    )


def _check_embeddings_exist(config: EmbeddingConfig) -> None:
    assert (
        config.agg_embeddings_dir.exists()
    ), f"Embeddings directory {config.agg_embeddings_dir} does not exist"
    assert (
        config.chunk_embeddings_dir.exists()
    ), f"Embeddings directory {config.chunk_embeddings_dir} does not exist"


if __name__ == "__main__":
    # Configure logger to include timestamps and line numbers
    LOG_FILE = ROOT_PATH / "weaviate_ingestion.log"
    logger.add(
        LOG_FILE,
        rotation="100 MB",
        level="INFO",  # Default to INFO level
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )
    logger.debug(f"Saving logs to {LOG_FILE}")
    logger.debug(f"Using .env file at: {ROOT_PATH / '.env'}")
    load_dotenv(ROOT_PATH / ".env", override=True)

    # Get Weaviate connection details from environment
    WV_HOST = os.environ["WV_HOST"]
    WV_PORT = os.environ["WV_PORT"]
    WV_GRPC_PORT = os.environ["WV_GRPC_PORT"]
    WV_API_KEY = os.environ.get("WV_API_KEY")  # Make API key optional
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # For text2vec-openai if used

    # Additional headers for model providers if needed
    HEADERS = {}
    if OPENAI_API_KEY:
        HEADERS["X-OpenAI-Api-Key"] = OPENAI_API_KEY

    logger.debug(f"WV_HOST: {WV_HOST}")
    logger.debug(f"WV_PORT: {WV_PORT}")
    logger.debug(f"WV_GRPC_PORT: {WV_GRPC_PORT}")
    logger.debug(f"Using API key: {'Yes' if WV_API_KEY else 'No'}")
    logger.debug(f"Using OpenAI integration: {'Yes' if OPENAI_API_KEY else 'No'}")

    PROCESSING_PROC = int(os.getenv("PROCESSING_PROC", multiprocessing.cpu_count() - 2))
    INGEST_PROC = int(os.getenv("INGEST_PROC", int(PROCESSING_PROC / 2)))

    logger.debug(f"Using {PROCESSING_PROC} processes for embedding ingestion")
    logger.debug(f"Using {INGEST_PROC} workers for parallel ingestion")
    logger.debug(f"Connecting to Weaviate at {WV_HOST}:{WV_PORT} (gRPC: {WV_GRPC_PORT})")

    main()
