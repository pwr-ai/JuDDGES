import gc
import math
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pformat
from typing import Any, Callable

import hydra
import numpy as np
import polars as pl
from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from grpc import stream_stream_rpc_method_handler
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from juddges.config import EmbeddingConfig
from juddges.data.weaviate_db import WeaviateJudgmentsDatabase
from juddges.settings import CONFIG_PATH, ROOT_PATH
from juddges.utils.config import resolve_config
from juddges.utils.date_utils import process_judgment_dates
from weaviate.util import generate_uuid5


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="embedding.yaml")
def main(cfg: DictConfig):
    cfg_dict = resolve_config(cfg)
    logger.info(f"config:\n{pformat(cfg_dict)}")
    config = EmbeddingConfig(**cfg_dict)

    _check_embeddings_exist(config)
    logger.info(f"Igesting {config.output_dir} to Weaviate...")

    ds = load_dataset(config.dataset_name, num_proc=NUM_PROC)
    chunk_ds = load_dataset("parquet", data_dir=config.chunk_embeddings_dir, num_proc=NUM_PROC)
    agg_ds = load_dataset("parquet", data_dir=config.agg_embeddings_dir, num_proc=NUM_PROC)

    ds = ds["train"]
    chunk_ds = chunk_ds["train"]
    agg_ds = agg_ds["train"]

    breakpoint()

    agg_ds = concatenate_datasets(
        [ds, agg_ds.rename_column("judgment_id", "agg_judgment_id")], axis=1
    )
    assert (
        agg_ds.select_columns(["agg_judgment_id", "judgment_id"])
        .to_polars()
        .with_columns(pl.col("agg_judgment_id") == pl.col("judgment_id"))
        .all()
    ), "Script assumes same order of judgments in embeddings and original dataset"
    agg_ds = agg_ds.remove_columns(["agg_judgment_id"])

    do_ingest(
        chunk_ds,
        WeaviateJudgmentsDatabase.JUDGMENT_CHUNKS_COLLECTION,
        process_batch_of_chunks,
        BATCH_SIZE,
        upsert=True,
    )

    do_ingest(
        agg_ds,
        WeaviateJudgmentsDatabase.JUDGMENTS_COLLECTION,
        process_batch_of_documents,
        BATCH_SIZE,
        upsert=True,
    )


def do_ingest(
    dataset: Dataset,
    collection: stream_stream_rpc_method_handler,
    process_batch_func: Callable,
    batch_size: int,
    upsert: bool,
) -> None:
    with WeaviateJudgmentsDatabase(WV_HOST, WV_PORT, WV_GRPC_PORT, WV_API_KEY) as db:
        collection = db.get_collection(collection)
        initial_count = db.get_collection_size(collection)
        logger.info(f"Initial number of objects in collection: {initial_count}")

        if not upsert:
            logger.info("upsert disabled - uploading only new embeddings")
            uuids = set(db.get_uuids(collection))
            logger.info(f"Found {len(uuids)} existing UUIDs in database")

            original_len = dataset.num_rows
            dataset = dataset.filter(
                lambda item: item["uuid"] not in uuids,
                num_proc=NUM_PROC,
                desc="Filtering out already uploaded embeddings",
            )
            logger.info(f"Filtered out {original_len - dataset.num_rows} existing embeddings")
        else:
            logger.info(
                "upsert enabled - uploading all embeddings (automatically updating already uploaded)"
            )

        total_batches = math.ceil(dataset.num_rows / batch_size)
        logger.info(f"Processing {total_batches} batches with size {batch_size}")

        if NUM_PROC > 1:
            logger.info(f"Using parallel processing with {MAX_WORKERS} workers")
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [
                    executor.submit(process_batch_func, db, batch)
                    for batch in tqdm(
                        dataset.iter(batch_size=batch_size),
                        total=total_batches,
                        desc="Uploading batches in parallel",
                    )
                ]
                for future in as_completed(futures):
                    try:
                        future.result()  # Raise exceptions if any
                    except Exception as e:
                        logger.error(f"Batch processing failed: {str(e)}")
        else:
            logger.info("Using sequential processing")
            for batch in tqdm(
                dataset.iter(batch_size=batch_size),
                total=total_batches,
                desc="Uploading batches sequentially",
            ):
                process_batch_func(db, batch)

        final_count = len(db.get_uuids(db.judgment_chunks_collection))
        logger.info(f"Final number of objects in collection: {final_count}")
        logger.info(f"Added {final_count - initial_count} new objects")


def process_batch_of_chunks(db: WeaviateJudgmentsDatabase, batch: dict[str, list[Any]]) -> None:
    """Process and insert a batch of embeddings into Weaviate."""
    try:
        if not validate_batch_of_chunks(batch):
            logger.error("Skipping invalid batch")
            return

        objects = [
            {
                "properties": {
                    "judgment_id": jid,
                    "chunk_id": cid,
                    "chunk_text": text,
                },
                "uuid": generate_uuid5(f"{jid}_chunk_{cid}"),
                "vector": emb,
            }
            for jid, cid, text, emb in zip(
                batch["judgment_id"],
                batch["chunk_id"],
                batch["chunk_text"],
                batch["embedding"],
            )
        ]

        if objects:
            db.insert_batch(collection=db.judgment_chunks_collection, objects=objects)
            logger.info(f"Successfully inserted {len(objects)} objects")

        # Help garbage collection
        del objects
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


def process_batch_of_documents(db: WeaviateJudgmentsDatabase, batch: dict[str, list[Any]]) -> None:
    """Process and insert a batch of documents into Weaviate."""
    try:
        records = []
        for i in range(len(batch["judgment_id"])):
            judgment_id = batch["judgment_id"][i]

            # Skip if no embedding exists
            if judgment_id not in batch:
                logger.warning(f"No embedding found for judgment_id: {judgment_id}")
                continue

            properties = {
                key: batch[key][i] for key in batch.keys() if key in db.judgments_properties
            }
            properties = process_judgment_dates(properties)

            records.append(
                {
                    "properties": properties,
                    "uuid": generate_uuid5(judgment_id),
                    "vector": batch["embedding"][i].tolist(),
                }
            )

        if records:
            db.insert_batch(
                collection=db.judgments_collection,
                objects=records,
            )
            logger.info(f"Successfully inserted {len(records)} documents")
        else:
            logger.warning("No records to insert")
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise


def _check_embeddings_exist(config: EmbeddingConfig):
    assert (
        config.agg_embeddings_dir.exists()
    ), f"Embeddings directory {config.agg_embeddings_dir} does not exist"
    assert (
        config.chunk_embeddings_dir.exists()
    ), f"Embeddings directory {config.chunk_embeddings_dir} does not exist"


if __name__ == "__main__":
    # Configure logger to include timestamps and line numbers
    logger.add(
        "weaviate_ingestion.log",
        rotation="100 MB",
        level="INFO",  # Default to INFO level
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )

    logger.debug(f"Using .env file at: {ROOT_PATH / '.env'}")
    load_dotenv(ROOT_PATH / ".env", override=True)

    WV_HOST = os.environ["WV_HOST"]
    WV_PORT = os.environ["WV_PORT"]
    WV_GRPC_PORT = os.environ["WV_GRPC_PORT"]
    WV_API_KEY = os.environ["WV_API_KEY"]

    logger.debug(f"WV_HOST: {WV_HOST}")
    logger.debug(f"WV_PORT: {WV_PORT}")
    logger.debug(f"WV_GRPC_PORT: {WV_GRPC_PORT}")

    BATCH_SIZE = 64
    NUM_PROC = multiprocessing.cpu_count() - 2
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", int(NUM_PROC / 2)))

    logger.debug(f"Using batch size: {BATCH_SIZE}")
    logger.debug(f"Using {NUM_PROC} processes for embedding ingestion")
    logger.debug(f"Using {MAX_WORKERS} workers for parallel ingestion")
    logger.debug(f"Connecting to Weaviate at {WV_HOST}:{WV_PORT} (gRPC: {WV_GRPC_PORT})")

    main()
