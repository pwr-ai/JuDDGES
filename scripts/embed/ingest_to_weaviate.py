import gc
import math
import multiprocessing
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pformat
from typing import Any, Callable

import hydra
import numpy as np
import polars as pl
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

import weaviate.exceptions
from juddges.config import EmbeddingConfig
from juddges.data.weaviate_db import WeaviateJudgmentsDatabase
from juddges.settings import CONFIG_PATH, ROOT_PATH
from juddges.utils.config import resolve_config
from juddges.utils.date_utils import process_judgment_dates
from juddges.utils.misc import parse_true_string
from weaviate.util import generate_uuid5

DEFAULT_INGEST_BATCH_SIZE = 64
DEFAULT_UPSERT = True


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
    do_ingest(
        dataset=chunk_ds,
        collection_name=WeaviateJudgmentsDatabase.JUDGMENT_CHUNKS_COLLECTION,
        process_batch_func=process_batch_of_chunks,
        batch_size=ingest_batch_size,
        upsert=True,
    )
    del chunk_ds
    gc.collect()

    do_ingest(
        dataset=agg_ds,
        collection_name=WeaviateJudgmentsDatabase.JUDGMENTS_COLLECTION,
        process_batch_func=process_batch_of_documents,
        batch_size=ingest_batch_size,
        upsert=upsert,
    )


def do_ingest(
    dataset: Dataset,
    collection_name: str,
    process_batch_func: Callable[[WeaviateJudgmentsDatabase, Dataset, int, int], None],
    batch_size: int,
    upsert: bool,
) -> None:
    num_rows = dataset.num_rows

    with WeaviateJudgmentsDatabase(WV_HOST, WV_PORT, WV_GRPC_PORT, WV_API_KEY) as db:
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
                try:
                    futures = [
                        executor.submit(
                            process_batch_func,
                            db=db,
                            dataset=dataset,
                            batch_idx=batch_idx,
                            batch_size=batch_size,
                        )
                        for batch_idx in range(total_batches)
                    ]
                    for future in as_completed(futures):
                        try:
                            future.result()  # Raise exceptions if any
                        except weaviate.exceptions.WeaviateBaseError as e:
                            logger.error(f"Batch processing failed: {str(e)}")
                        except (ValueError, AssertionError) as e:
                            logger.error(f"Batch validation failed: {str(e)}")
                        except Exception as e:
                            logger.error(f"Unexpected error during batch processing: {str(e)}")
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received. Cancelling remaining tasks...")
                    executor.shutdown(wait=False, cancel_futures=True)
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
    db: WeaviateJudgmentsDatabase,
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

    except (
        weaviate.exceptions.WeaviateConnectionError,
        weaviate.exceptions.WeaviateBatchError,
        weaviate.exceptions.WeaviateQueryError,
        weaviate.exceptions.WeaviateClosedClientError,
        ValueError,
    ) as e:
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


def process_batch_of_documents(
    db: WeaviateJudgmentsDatabase,
    dataset: Dataset,
    batch_idx: int,
    batch_size: int,
) -> None:
    """Process and insert a batch of documents into Weaviate."""
    batch = dataset[batch_idx * batch_size : (batch_idx + 1) * batch_size]

    try:
        records = []
        for i in range(len(batch["judgment_id"])):
            property_names = set(db.judgments_properties).intersection(batch.keys())
            missing_properties = set(db.judgments_properties).difference(batch.keys())
            logger.warning(
                f"Found missing properties compared to weaviate schema: {missing_properties}, "
                f"uploading only {property_names}"
            )
            properties = {key: batch[key][i] for key in property_names}
            properties = process_judgment_dates(properties)

            records.append(
                {
                    "properties": properties,
                    "uuid": generate_uuid5(batch["judgment_id"][i]),
                    "vector": batch["embedding"][i],
                }
            )

        if records:
            db.insert_batch(
                collection=db.judgments_collection,
                objects=records,
            )
            logger.debug(f"Successfully inserted {len(records)} documents")
            del records
            gc.collect()
        else:
            logger.warning("No records to insert")
    except (
        weaviate.exceptions.WeaviateConnectionError,
        weaviate.exceptions.WeaviateBatchError,
        weaviate.exceptions.WeaviateQueryError,
        weaviate.exceptions.WeaviateClosedClientError,
        ValueError,
    ) as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise


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

    WV_HOST = os.environ["WV_HOST"]
    WV_PORT = os.environ["WV_PORT"]
    WV_GRPC_PORT = os.environ["WV_GRPC_PORT"]
    WV_API_KEY = os.environ["WV_API_KEY"]

    logger.debug(f"WV_HOST: {WV_HOST}")
    logger.debug(f"WV_PORT: {WV_PORT}")
    logger.debug(f"WV_GRPC_PORT: {WV_GRPC_PORT}")

    PROCESSING_PROC = int(os.getenv("PROCESSING_PROC", multiprocessing.cpu_count() - 2))
    INGEST_PROC = int(os.getenv("INGEST_PROC", int(PROCESSING_PROC / 2)))

    logger.debug(f"Using {PROCESSING_PROC} processes for embedding ingestion")
    logger.debug(f"Using {INGEST_PROC} workers for parallel ingestion")
    logger.debug(f"Connecting to Weaviate at {WV_HOST}:{WV_PORT} (gRPC: {WV_GRPC_PORT})")

    main()
