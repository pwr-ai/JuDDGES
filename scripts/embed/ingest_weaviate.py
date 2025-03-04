import math
import os
from pathlib import Path

import typer
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from tqdm.auto import tqdm

from juddges.data.weaviate_db import WeaviateJudgementsDatabase
from weaviate.util import generate_uuid5

load_dotenv()
WV_HOST = os.getenv("WV_HOST", "localhost")
WV_PORT = os.getenv("WV_PORT", "8080")
WV_GRPC_PORT = os.getenv("WV_GRPC_PORT", "50051")
WV_API_KEY = os.getenv("WV_API_KEY", None)

BATCH_SIZE = 64
NUM_PROC = int(os.getenv("NUM_PROC", 1))

logger.info(f"Connecting to Weaviate at {WV_HOST}:{WV_PORT} (gRPC: {WV_GRPC_PORT})")


def main(
    embeddings_dir: Path = typer.Option(...),
    batch_size: int = typer.Option(BATCH_SIZE),
    upsert: bool = typer.Option(False),
) -> None:
    logger.warning(
        "The script will upload local embeddings to the database, "
        "make sure they are the same as in the inference module of the database."
    )
    embs = load_dataset(str(embeddings_dir))["train"]
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
            embs = embs.filter(lambda item: item["uuid"] not in uuids)
        else:
            logger.info(
                "upsert enabled - uploading all embeddings (automatically updating already uploaded)"
            )

        for batch in tqdm(
            embs.iter(batch_size=batch_size),
            total=math.ceil(len(embs) / batch_size),
            desc="Uploading batches",
        ):
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


if __name__ == "__main__":
    typer.run(main)
