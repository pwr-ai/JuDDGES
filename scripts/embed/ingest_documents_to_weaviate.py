import math
import multiprocessing
import os
import asyncio
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
NUM_PROC = int(os.getenv("NUM_PROC", multiprocessing.cpu_count() - 2))
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


async def process_batch(db, batch, batch_embeddings, properties_list):
    """Process and insert a batch of documents into Weaviate."""
    try:
        records = []
        for i in range(len(batch["judgment_id"])):
            judgment_id = batch["judgment_id"][i]

            # Skip if no embedding exists
            if judgment_id not in batch_embeddings:
                logger.warning(f"No embedding found for judgment_id: {judgment_id}")
                continue

            properties = {
                key: batch[key][i] for key in batch.keys() if key in properties_list
            }
            properties = process_judgment_dates(properties)

            records.append(
                {
                    "properties": properties,
                    "uuid": generate_uuid5(judgment_id),
                    "vector": batch_embeddings[judgment_id].tolist(),
                }
            )

        if records:  # Only insert if we have valid records
            await db.async_insert_batch(
                collection=db.judgments_collection,
                objects=records,
            )
            logger.info(f"Successfully inserted {len(records)} documents")
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise


async def get_collection_size(collection):
    """Get the number of documents in a collection."""
    count = 0
    async for _ in collection.iterator(return_properties=[]):
        count += 1
    return count


async def main_async(
    dataset_name: str,
    embeddings_dir: Path,
    batch_size: int,
    debug: bool,
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
        embeddings_dict = torch.load(target_file, weights_only=True)
        logger.info(f"Loaded embeddings for {len(embeddings_dict)} documents")

        dataset = load_dataset(str(dataset_name))["train"]
        logger.info(f"Dataset loaded with columns: {dataset.column_names}")
        total_batches = math.ceil(len(dataset) / batch_size)

        async with WeaviateJudgmentsDatabase(WV_HOST, WV_PORT, WV_GRPC_PORT, WV_API_KEY) as db:
            logger.info("Checking number of documents in collection...")
            initial_count = await get_collection_size(db.judgments_collection)
            logger.info(f"Initial number of documents in collection: {initial_count}")

            # Get properties asynchronously
            properties_list = await db.judgments_properties
            
            if set(properties_list) != set(dataset.column_names):
                logger.warning(
                    "Dataset columns do not match judgment properties, ignoring extra columns: "
                    f"{set(dataset.column_names) - set(properties_list)}"
                )

            if MAX_WORKERS > 1:
                logger.info(f"Using parallel processing with {MAX_WORKERS} workers")
                batch_tasks = []
                
                for batch_idx, batch in enumerate(tqdm(
                    dataset.iter(batch_size=batch_size),
                    total=total_batches,
                    desc="Preparing batch tasks",
                )):
                    # Get embeddings only for current batch
                    batch_embeddings = get_batch_embeddings(
                        batch["judgment_id"], embeddings_dict
                    )
                    batch_tasks.append(process_batch(db, batch, batch_embeddings, properties_list))
                    
                    # Process in smaller groups to avoid memory issues
                    if len(batch_tasks) >= MAX_WORKERS or batch_idx == total_batches - 1:
                        await asyncio.gather(*batch_tasks)
                        batch_tasks = []
            else:
                logger.info("Using sequential processing")
                for batch in tqdm(
                    dataset.iter(batch_size=batch_size),
                    total=total_batches,
                    desc="Uploading batches sequentially",
                ):
                    # Get embeddings only for current batch
                    batch_embeddings = get_batch_embeddings(batch["judgment_id"], embeddings_dict)
                    await process_batch(db, batch, batch_embeddings, properties_list)

            final_count = await get_collection_size(db.judgments_collection)
            logger.info(f"Final number of documents in collection: {final_count}")
            logger.info(f"Added {final_count - initial_count} new documents")

    except Exception as e:
        logger.error(f"Script failed: {str(e)}", exc_info=True)
        raise


def main(
    dataset_name: Path = typer.Option("JuDDGES/pl-court-raw"),
    embeddings_dir: Path = typer.Option(...),
    batch_size: int = typer.Option(BATCH_SIZE),
    debug: bool = typer.Option(False, help="Enable debug logging"),
) -> None:
    """Run the main async function."""
    asyncio.run(main_async(
        dataset_name=str(dataset_name),
        embeddings_dir=embeddings_dir,
        batch_size=batch_size,
        debug=debug,
    ))


if __name__ == "__main__":
    typer.run(main)
