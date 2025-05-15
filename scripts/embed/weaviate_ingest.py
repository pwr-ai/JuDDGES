#!/usr/bin/env python3
"""
Main entry point for Weaviate ingestion.

This script orchestrates the ingestion of document and chunk embeddings
into Weaviate collections.
"""

import gc
import os
from pprint import pformat

import hydra
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from embed.weaviate.config import (
    DEBUG,
    DEFAULT_INGEST_BATCH_SIZE,
    DEFAULT_MAX_DOCUMENTS,
    DEFAULT_PROCESSING_PROC,
    DEFAULT_UPSERT,
    IngestConfig,
)
from embed.weaviate.ingesters import ChunkIngester, DocumentIngester
from embed.weaviate.loaders import DatasetLoader
from juddges.config import EmbeddingConfig
from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase
from juddges.settings import CONFIG_PATH, ROOT_PATH
from juddges.utils.config import resolve_config
from juddges.utils.misc import parse_true_string


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="embedding.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the Weaviate ingestion script.

    Args:
        cfg: Hydra configuration object
    """
    if DEBUG:
        logger.info("=== Running in DEBUG mode ===")
        logger.info(f"Default batch size: {DEFAULT_INGEST_BATCH_SIZE}")
        logger.info(f"Default max documents: {DEFAULT_MAX_DOCUMENTS}")
        logger.info(f"Default processing proc: {DEFAULT_PROCESSING_PROC}")

    cfg_dict = resolve_config(cfg)

    # Extract ingestion configuration
    ingest_config = IngestConfig(
        batch_size=cfg_dict.pop("ingest_batch_size", DEFAULT_INGEST_BATCH_SIZE),
        upsert=parse_true_string(cfg_dict.pop("upsert", DEFAULT_UPSERT)),
        max_documents=cfg_dict.pop("max_documents", DEFAULT_MAX_DOCUMENTS),
        processing_proc=DEFAULT_PROCESSING_PROC,
        ingest_proc=int(os.getenv("INGEST_PROC", DEFAULT_PROCESSING_PROC // 2)),
    )

    logger.info(f"Using batch size: {ingest_config.batch_size}")
    logger.info(f"Using upsert: {ingest_config.upsert}")
    logger.info(f"Max documents to ingest: {ingest_config.max_documents}")

    logger.info(f"config:\n{pformat(cfg_dict)}")
    config = EmbeddingConfig(**cfg_dict)

    logger.info(f"Starting ingestion of {config.output_dir} to Weaviate...")

    # Set up Weaviate schema before ingestion
    with WeaviateLegalDocumentsDatabase() as db:
        db.create_collections()

    # Load datasets
    dataset_loader = DatasetLoader(config)

    try:
        # Load and ingest chunks
        logger.info("Loading and ingesting chunk embeddings...")
        chunk_ds = dataset_loader.load_chunk_dataset()

        # Filter chunks if max_documents is set
        if ingest_config.max_documents is not None:
            doc_ds = dataset_loader.load_document_dataset()
            doc_ids = set(
                doc_ds.select(range(min(ingest_config.max_documents, doc_ds.num_rows)))[
                    "judgment_id"
                ]
            )
            chunk_ds = chunk_ds.filter(lambda x: x["judgment_id"] in doc_ids)
            logger.info(
                f"Selected {len(doc_ids)} documents and {chunk_ds.num_rows} chunks for ingestion"
            )
            del doc_ds
            gc.collect()

        # Create a shared database connection for both ingesters
        with WeaviateLegalDocumentsDatabase() as db:
            # Ingest chunks
            logger.info("Starting ingestion of chunk embeddings to Weaviate...")
            chunk_ingester = ChunkIngester(db=db, config=ingest_config)
            chunk_ingester.ingest(chunk_ds)

            # Free memory
            del chunk_ds
            gc.collect()
            logger.info("Chunk ingestion complete and memory freed.")

            # Load and ingest documents
            logger.info("Loading and ingesting document embeddings...")
            doc_ds = dataset_loader.load_document_dataset()

            # Ingest documents
            logger.info("Starting ingestion of aggregated document embeddings to Weaviate...")
            doc_ingester = DocumentIngester(db=db, config=ingest_config)
            doc_ingester.ingest(doc_ds)
            logger.info("Document ingestion complete.")

    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise

    logger.info("Ingestion process completed successfully!")


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
    WV_API_KEY = os.environ["WV_API_KEY"]

    logger.debug(f"WV_HOST: {WV_HOST}")
    logger.debug(f"WV_PORT: {WV_PORT}")
    logger.debug(f"WV_GRPC_PORT: {WV_GRPC_PORT}")
    logger.debug(f"Using API key: {'Yes' if WV_API_KEY else 'No'}")

    logger.debug(f"Connecting to Weaviate at {WV_HOST}:{WV_PORT} (gRPC: {WV_GRPC_PORT})")

    main()
