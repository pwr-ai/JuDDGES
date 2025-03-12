import math
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import typer
from datasets import load_from_disk
from dotenv import load_dotenv
from loguru import logger
from tqdm.auto import tqdm
from weaviate.util import generate_uuid5

from juddges.data.weaviate_db import WeaviateJudgementsDatabase
from juddges.settings import ROOT_PATH

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
        assert len(batch["judgement_id"]) > 0, "Batch is empty"
        assert len(batch["judgement_id"]) == len(
            batch["chunk_id"]
        ), "Mismatched lengths between judgement_id and chunk_id"
        assert len(batch["judgement_id"]) == len(
            batch["chunk_text"]
        ), "Mismatched lengths between judgement_id and chunk_text"
        assert len(batch["judgement_id"]) == len(
            batch["embedding"]
        ), "Mismatched lengths between judgement_id and embedding"

        # Validate embedding dimensions
        embedding_shape = np.array(batch["embedding"][0]).shape
        logger.debug(f"Embedding shape: {embedding_shape}")
        assert len(embedding_shape) == 1, f"Invalid embedding shape: {embedding_shape}"

        # Check for null values
        assert all(
            id_ is not None for id_ in batch["judgement_id"]
        ), "Found null judgement_id"
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


def process_batch(db, batch):
    """Process and insert a batch of embeddings into Weaviate."""
    logger.debug(f"Processing batch of size {len(batch['judgement_id'])}")

    if not validate_batch(batch):
        logger.error("Skipping invalid batch")
        return

    try:
        objects = [
            {
                "properties": {
                    "judgement_id": batch["judgement_id"][i],
                    "chunk_id": batch["chunk_id"][i],
                    "chunk_text": batch["chunk_text"][i],
                },
                "uuid": generate_uuid5(
                    f"{batch['judgement_id'][i]}_chunk_{batch['chunk_id'][i]}"
                ),
                "vector": batch["embedding"][i],
            }
            for i in range(len(batch["judgement_id"]))
        ]

        logger.info(f"Created {len(objects)} objects for insertion")
        logger.debug(
            f"Sample object: {str(objects[0])[:15] + '...' if objects else 'No objects'}"
        )

        if len(objects) > 0:
            try:
                db.insert_batch(
                    collection=db.judgement_chunks_collection,
                    objects=objects,
                )
                logger.info(
                    f"Successfully inserted {len(objects)} objects into {db.JUDGMENT_CHUNKS_COLLECTION}"
                )
            except Exception as e:
                logger.error(f"Failed to insert batch: {str(e)}")
                logger.debug(f"Failed batch objects: {objects}")
                raise
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise


def main(
    embeddings_dir: Path = typer.Option(
        "data/embeddings/pl-court-raw/mmlw-roberta-large/all_embeddings"
    ),
    batch_size: int = typer.Option(BATCH_SIZE),
    upsert: bool = typer.Option(False),
    max_embeddings: int = typer.Option(
        None, help="Maximum number of embeddings to process"
    ),
    debug: bool = typer.Option(False, help="Enable debug logging"),
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
        assert (
            embeddings_path.exists()
        ), f"Embeddings directory not found: {embeddings_dir}"

        embs = load_from_disk(str(embeddings_dir))
        logger.info(f"Loaded dataset with {len(embs)} embeddings")
        logger.info(f"Dataset columns: {embs.column_names}")

        if max_embeddings is not None:
            logger.info(f"Limiting number of embeddings to {max_embeddings}")
            embs = embs.select(range(min(max_embeddings, len(embs))))

        embs = embs.map(
            lambda item: {
                "uuid": WeaviateJudgementsDatabase.uuid_from_judgement_chunk_id(
                    judgement_id=item["judgement_id"], chunk_id=item["chunk_id"]
                )
            },
            num_proc=NUM_PROC,
            desc="Generating UUIDs",
        )

        with WeaviateJudgementsDatabase(
            WV_HOST, WV_PORT, WV_GRPC_PORT, WV_API_KEY
        ) as db:
            initial_count = len(db.get_uuids(db.judgement_chunks_collection))
            logger.info(f"Initial number of objects in collection: {initial_count}")

            if not upsert:
                logger.info("upsert disabled - uploading only new embeddings")
                uuids = set(db.get_uuids(db.judgement_chunks_collection))
                logger.info(f"Found {len(uuids)} existing UUIDs in database")

                original_len = len(embs)
                embs = embs.filter(
                    lambda item: item["uuid"] not in uuids,
                    num_proc=NUM_PROC,
                    desc="Filtering out already uploaded embeddings",
                )
                logger.info(
                    f"Filtered out {original_len - len(embs)} existing embeddings"
                )
            else:
                logger.info(
                    "upsert enabled - uploading all embeddings (automatically updating already uploaded)"
                )

            total_batches = math.ceil(len(embs) / batch_size)
            logger.info(f"Processing {total_batches} batches with size {batch_size}")

            if NUM_PROC > 1:
                logger.info(f"Using parallel processing with {NUM_PROC} workers")
                with ThreadPoolExecutor(max_workers=NUM_PROC) as executor:
                    futures = [
                        executor.submit(process_batch, db, batch)
                        for batch in tqdm(
                            embs.iter(batch_size=batch_size),
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
                    embs.iter(batch_size=batch_size),
                    total=total_batches,
                    desc="Uploading batches sequentially",
                ):
                    process_batch(db, batch)

            final_count = len(db.get_uuids(db.judgement_chunks_collection))
            logger.info(f"Final number of objects in collection: {final_count}")
            logger.info(f"Added {final_count - initial_count} new objects")

    except Exception as e:
        logger.error(f"Script failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    typer.run(main)
