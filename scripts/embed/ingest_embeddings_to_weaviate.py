import math
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from datasets import load_from_disk
from dotenv import load_dotenv
from loguru import logger
from tqdm.auto import tqdm
from weaviate.util import generate_uuid5

from juddges.data.weaviate_db import WeaviateJudgementsDatabase

load_dotenv()
WV_HOST = os.environ["WV_HOST"]
WV_PORT = os.getenv("WV_PORT", "8080")
WV_GRPC_PORT = os.getenv("WV_GRPC_PORT", "50051")
WV_API_KEY = os.getenv("WV_API_KEY", None)

BATCH_SIZE = 64
logger.info(f"Using batch size: {BATCH_SIZE}")
NUM_PROC = multiprocessing.cpu_count() - 2
logger.info(f"Using {NUM_PROC} processes for embedding ingestion")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", int(NUM_PROC / 2)))
logger.info(f"Using {MAX_WORKERS} workers for parallel ingestion")
logger.info(f"Connecting to Weaviate at {WV_HOST}:{WV_PORT} (gRPC: {WV_GRPC_PORT})")


def process_batch(db, batch):
    objects = [
        {
            "properties": {
                "judgment_id": batch["_id"][i],
                "chunk_id": batch["chunk_id"][i],
                "chunk_text": batch["text_chunk"][i],
            },
            "uuid": generate_uuid5(f"{batch['_id'][i]}_chunk_{batch['chunk_id'][i]}"),
            "vector": batch["embedding"][i],
        }
        for i in range(len(batch["_id"]))
    ]
    db.insert_batch(
        collection=db.judgement_chunks_collection,
        objects=objects,
    )


def main(
    embeddings_dir: Path = typer.Option(
        "data/embeddings/pl-court-raw/mmlw-roberta-large/all_embeddings"
    ),
    batch_size: int = typer.Option(BATCH_SIZE),
    upsert: bool = typer.Option(False),
    max_embeddings: int = typer.Option(
        None, help="Maximum number of embeddings to process"
    ),
    cpu_count: int = typer.Option(
        NUM_PROC,
        help="Number of CPUs to use for parallel ingestion, defaults to NUM_PROC",
    ),
) -> None:
    logger.warning(
        "The script will upload local embeddings to the database, "
        "make sure they are the same as in the inference module of the database."
    )
    embs = load_from_disk(str(embeddings_dir))["train"]

    if max_embeddings is not None:
        logger.info(f"Limiting number of embeddings to {max_embeddings}")
        embs = embs.select(range(min(max_embeddings, len(embs))))

    embs = embs.map(
        lambda item: {
            "uuid": WeaviateJudgementsDatabase.uuid_from_judgement_chunk_id(
                judgement_id=item["_id"], chunk_id=item["chunk_id"]
            )
        },
        num_proc=NUM_PROC,
        desc="Generating UUIDs",
    )
    with WeaviateJudgementsDatabase(WV_HOST, WV_PORT, WV_GRPC_PORT, WV_API_KEY) as db:
        if not upsert:
            logger.info("upsert disabled - uploading only new embeddings")
            uuids = set(db.get_uuids(db.judgement_chunks_collection))
            embs = embs.filter(
                lambda item: item["uuid"] not in uuids,
                num_proc=NUM_PROC,
                desc="Filtering out already uploaded embeddings",
            )
        else:
            logger.info(
                "upsert enabled - uploading all embeddings (automatically updating already uploaded)"
            )

        if cpu_count is None:
            cpu_count = NUM_PROC

        if cpu_count > 0:
            with ThreadPoolExecutor(max_workers=cpu_count) as executor:
                futures = [
                    executor.submit(process_batch, db, batch)
                    for batch in tqdm(
                        embs.iter(batch_size=batch_size),
                        total=math.ceil(len(embs) / batch_size),
                        desc="Uploading batches in parallel",
                    )
                ]
                for future in as_completed(futures):
                    future.result()  # Raise exceptions if any
        else:
            for batch in tqdm(
                embs.iter(batch_size=batch_size),
                total=math.ceil(len(embs) / batch_size),
                desc="Uploading batches",
            ):
                process_batch(db, batch)


if __name__ == "__main__":
    typer.run(main)
