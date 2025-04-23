import gc
import math
import multiprocessing
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import typer
from datasets import load_from_disk
from dotenv import load_dotenv
from loguru import logger
from tqdm.auto import tqdm

from juddges.data.weaviate_db import WeaviateJudgmentsDatabase
from juddges.settings import ROOT_PATH
from weaviate.util import generate_uuid5

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


def validate_batch(batch):
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


async def process_batch(db, batch):
    """Process and insert a batch of embeddings into Weaviate."""
    try:
        if not validate_batch(batch):
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
            await db.async_insert_batch(collection=db.judgment_chunks_collection, objects=objects)
            logger.info(f"Successfully inserted {len(objects)} objects")

        # Help garbage collection
        del objects
        gc.collect()

    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise


async def main_async(
    embeddings_dir: Path,
    batch_size: int,
    upsert: bool,
    max_embeddings: int,
    debug: bool,
) -> None:
    if debug:
        logger.remove()  # Remove default handler
        logger.add(
            "weaviate_ingestion.log",
            rotation="100 MB",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
        )
        logger.info("Debug logging enabled")

    try:
        logger.warning(
            "The script will upload local embeddings to the database, "
            "make sure they are the same as in the inference module of the database."
        )

        # Validate input directory
        embeddings_path = Path(embeddings_dir)
        assert embeddings_path.exists(), f"Embeddings directory not found: {embeddings_dir}"

        embs = load_from_disk(str(embeddings_dir))
        if "judgement_id" in embs.column_names:
            logger.warning(
                "judgement_id column found in dataset, renaming to judgment_id to avoid conflicts"
            )
            embs = embs.rename_columns({"judgement_id": "judgment_id"})
        logger.info(f"Loaded dataset with {len(embs)} embeddings")
        logger.info(f"Dataset columns: {embs.column_names}")

        if max_embeddings is not None:
            logger.info(f"Limiting number of embeddings to {max_embeddings}")
            embs = embs.select(range(min(max_embeddings, len(embs))))

        embs = embs.map(
            lambda item: {
                "uuid": WeaviateJudgmentsDatabase.uuid_from_judgment_chunk_id(
                    judgment_id=item["judgment_id"],
                    chunk_id=item["chunk_id"],
                )
            },
            num_proc=NUM_PROC,
            desc="Generating UUIDs",
        )

        async with WeaviateJudgmentsDatabase(WV_HOST, WV_PORT, WV_GRPC_PORT, WV_API_KEY) as db:
            uuids = await db.async_get_uuids(db.judgment_chunks_collection)
            initial_count = len(uuids)
            logger.info(f"Initial number of objects in collection: {initial_count}")

            if not upsert:
                logger.info("upsert disabled - uploading only new embeddings")
                uuids_set = set(uuids)
                logger.info(f"Found {len(uuids_set)} existing UUIDs in database")

                original_len = len(embs)
                embs = embs.filter(
                    lambda item: item["uuid"] not in uuids_set,
                    num_proc=NUM_PROC,
                    desc="Filtering out already uploaded embeddings",
                )
                logger.info(f"Filtered out {original_len - len(embs)} existing embeddings")
            else:
                logger.info(
                    "upsert enabled - uploading all embeddings (automatically updating already uploaded)"
                )

            total_batches = math.ceil(len(embs) / batch_size)
            logger.info(f"Processing {total_batches} batches with size {batch_size}")

            if MAX_WORKERS > 1:
                logger.info(f"Using parallel processing with {MAX_WORKERS} workers")
                batch_tasks = []
                
                for batch_idx, batch in enumerate(tqdm(
                    embs.iter(batch_size=batch_size),
                    total=total_batches,
                    desc="Preparing batch tasks",
                )):
                    batch_tasks.append(process_batch(db, batch))
                    
                    # Process in smaller groups to avoid memory issues
                    if len(batch_tasks) >= MAX_WORKERS or batch_idx == total_batches - 1:
                        await asyncio.gather(*batch_tasks)
                        batch_tasks = []
                        # Help with garbage collection
                        gc.collect()
            else:
                logger.info("Using sequential processing")
                for batch in tqdm(
                    embs.iter(batch_size=batch_size),
                    total=total_batches,
                    desc="Uploading batches sequentially",
                ):
                    await process_batch(db, batch)
                    # Clean up batch after processing
                    del batch
                    gc.collect()

            final_uuids = await db.async_get_uuids(db.judgment_chunks_collection)
            final_count = len(final_uuids)
            logger.info(f"Final number of objects in collection: {final_count}")
            logger.info(f"Added {final_count - initial_count} new objects")

    except Exception as e:
        logger.error(f"Script failed: {str(e)}", exc_info=True)
        raise


def main(
    embeddings_dir: Path = typer.Option(
        "data/embeddings/pl-court-raw/mmlw-roberta-large/all_embeddings"
    ),
    batch_size: int = typer.Option(BATCH_SIZE),
    upsert: bool = typer.Option(False),
    max_embeddings: int = typer.Option(None, help="Maximum number of embeddings to process"),
    debug: bool = typer.Option(False, help="Enable debug logging"),
) -> None:
    """Run the main async function."""
    asyncio.run(main_async(
        embeddings_dir=embeddings_dir,
        batch_size=batch_size,
        upsert=upsert,
        max_embeddings=max_embeddings,
        debug=debug,
    ))


if __name__ == "__main__":
    typer.run(main)
