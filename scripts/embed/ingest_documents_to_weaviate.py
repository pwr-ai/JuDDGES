import math
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from tqdm.auto import tqdm
from weaviate.util import generate_uuid5

from juddges.data.weaviate_db import WeaviateJudgmentsDatabase
from juddges.settings import ROOT_PATH
from juddges.utils.date_utils import process_judgment_dates

# Configure logger
logger.add(
    "weaviate_ingestion.log",
    rotation="100 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
)

load_dotenv(dotenv_path=ROOT_PATH / ".env", override=True)

WV_HOST = os.environ["WV_HOST"]
WV_PORT = os.environ["WV_PORT"]
WV_GRPC_PORT = os.environ["WV_GRPC_PORT"]
WV_API_KEY = os.environ["WV_API_KEY"]

BATCH_SIZE = 16
NUM_PROC = multiprocessing.cpu_count() - 2
MAX_WORKERS = int(os.getenv("MAX_WORKERS", int(NUM_PROC / 2)))

logger.debug(f"Using batch size: {BATCH_SIZE}")
logger.debug(f"Using {MAX_WORKERS} workers for parallel ingestion")
logger.debug(f"Connecting to Weaviate at {WV_HOST}:{WV_PORT} (gRPC: {WV_GRPC_PORT})")


def process_batch(db, batch):
    """Process and insert a batch of documents into Weaviate."""
    try:
        records = []
        for i in range(len(batch["judgment_id"])):
            properties = {key: batch[key][i] for key in batch.keys()}
            # Process dates to RFC3339 format
            properties = process_judgment_dates(properties)
            records.append(
                {
                    "properties": properties,
                    "uuid": generate_uuid5(batch["judgment_id"][i]),
                }
            )

        db.insert_batch(
            collection=db.judgments_collection,
            objects=records,
        )
        logger.info(f"Successfully inserted {len(records)} documents")
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise


def main(
    dataset_name: Path = typer.Option("JuDDGES/pl-court-raw"),
    batch_size: int = typer.Option(BATCH_SIZE),
    debug: bool = typer.Option(False, help="Enable debug logging"),
) -> None:
    if debug:
        logger.remove()
        logger.add(
            "weaviate_ingestion.log",
            rotation="100 MB",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
        )
        logger.info("Debug logging enabled")

    try:
        dataset = load_dataset(str(dataset_name))["train"]
        logger.info(f"Dataset loaded with columns: {dataset.column_names}")
        total_batches = math.ceil(len(dataset) / batch_size)

        with WeaviateJudgmentsDatabase(
            WV_HOST, WV_PORT, WV_GRPC_PORT, WV_API_KEY
        ) as db:
            initial_count = len(db.get_uuids(db.judgments_collection))
            logger.info(f"Initial number of documents in collection: {initial_count}")

            if MAX_WORKERS > 1:
                logger.info(f"Using parallel processing with {MAX_WORKERS} workers")
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [
                        executor.submit(process_batch, db, batch)
                        for batch in tqdm(
                            dataset.iter(batch_size=batch_size),
                            total=total_batches,
                            desc="Uploading batches in parallel",
                        )
                    ]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Batch processing failed: {str(e)}")
            else:
                logger.info("Using sequential processing")
                for batch in tqdm(
                    dataset.iter(batch_size=batch_size),
                    total=total_batches,
                    desc="Uploading batches sequentially",
                ):
                    process_batch(db, batch)

            final_count = len(db.get_uuids(db.judgments_collection))
            logger.info(f"Final number of documents in collection: {final_count}")
            logger.info(f"Added {final_count - initial_count} new documents")

    except Exception as e:
        logger.error(f"Script failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    typer.run(main)
