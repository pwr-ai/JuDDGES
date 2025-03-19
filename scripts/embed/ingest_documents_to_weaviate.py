import math
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import typer
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from tqdm.auto import tqdm

from juddges.data.weaviate_db import WeaviateJudgmentsDatabase
from juddges.settings import ROOT_PATH
from juddges.utils.date_utils import process_judgment_dates
from weaviate.util import generate_uuid5

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

BATCH_SIZE = 32
NUM_PROC = multiprocessing.cpu_count() - 2
MAX_WORKERS = int(os.getenv("MAX_WORKERS", int(NUM_PROC / 2)))

logger.debug(f"Using batch size: {BATCH_SIZE}")
logger.debug(f"Using {MAX_WORKERS} workers for parallel ingestion")
logger.debug(f"Connecting to Weaviate at {WV_HOST}:{WV_PORT} (gRPC: {WV_GRPC_PORT})")


def get_batch_embeddings(judgment_ids: list[str], embeddings_dict: dict) -> dict:
    """Extract only needed embeddings for the current batch."""
    return {
        judgment_id: embeddings_dict[judgment_id]
        for judgment_id in judgment_ids
        if judgment_id in embeddings_dict
    }


def process_batch(db, batch, batch_embeddings):
    """Process and insert a batch of documents into Weaviate."""
    try:
        records = []
        for i in range(len(batch["judgment_id"])):
            judgment_id = batch["judgment_id"][i]

            # Skip if no embedding exists
            if judgment_id not in batch_embeddings:
                logger.warning(f"No embedding found for judgment_id: {judgment_id}")
                continue

            properties = {key: batch[key][i] for key in batch.keys()}
            properties = process_judgment_dates(properties)

            records.append(
                {
                    "properties": properties,
                    "uuid": generate_uuid5(judgment_id),
                    "vector": batch_embeddings[judgment_id].tolist(),
                }
            )

        if records:  # Only insert if we have valid records
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
    embeddings_dir: Path = typer.Option(...),
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
        # Load embeddings
        target_file = embeddings_dir.parent / "agg_embeddings.pt"
        logger.info(f"Loading embeddings from {target_file}")
        embeddings_dict = torch.load(target_file)
        logger.info(f"Loaded embeddings for {len(embeddings_dict)} documents")

        dataset = load_dataset(str(dataset_name))["train"]
        logger.info(f"Dataset loaded with columns: {dataset.column_names}")
        total_batches = math.ceil(len(dataset) / batch_size)

        with WeaviateJudgmentsDatabase(WV_HOST, WV_PORT, WV_GRPC_PORT, WV_API_KEY) as db:
            initial_count = len(db.get_uuids(db.judgments_collection))
            logger.info(f"Initial number of documents in collection: {initial_count}")

            if MAX_WORKERS > 1:
                logger.info(f"Using parallel processing with {MAX_WORKERS} workers")
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = []
                    for batch in tqdm(
                        dataset.iter(batch_size=batch_size),
                        total=total_batches,
                        desc="Uploading batches in parallel",
                    ):
                        # Get embeddings only for current batch
                        batch_embeddings = get_batch_embeddings(
                            batch["judgment_id"], embeddings_dict
                        )
                        futures.append(executor.submit(process_batch, db, batch, batch_embeddings))

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
                    # Get embeddings only for current batch
                    batch_embeddings = get_batch_embeddings(batch["judgment_id"], embeddings_dict)
                    process_batch(db, batch, batch_embeddings)

            final_count = len(db.get_uuids(db.judgments_collection))
            logger.info(f"Final number of documents in collection: {final_count}")
            logger.info(f"Added {final_count - initial_count} new documents")

    except Exception as e:
        logger.error(f"Script failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    typer.run(main)
