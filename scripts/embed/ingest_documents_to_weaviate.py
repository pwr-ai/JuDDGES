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
WV_HOST = os.environ["WV_HOST"]
WV_PORT = os.getenv("WV_PORT", "8080")
WV_GRPC_PORT = os.getenv("WV_GRPC_PORT", "50051")
WV_API_KEY = os.getenv("WV_API_KEY", None)

BATCH_SIZE = 64
NUM_PROC = int(os.getenv("NUM_PROC", 1))

logger.info(f"Connecting to Weaviate at {WV_HOST}:{WV_PORT} (gRPC: {WV_GRPC_PORT})")


def main(
    dataset_name: Path = typer.Option(...),
    batch_size: int = typer.Option(BATCH_SIZE),
) -> None:
    dataset = load_dataset(str(dataset_name))["train"]
    with WeaviateJudgementsDatabase(WV_HOST, WV_PORT, WV_GRPC_PORT, WV_API_KEY) as db:
        for batch in tqdm(
            dataset.iter(batch_size=batch_size),
            total=math.ceil(len(dataset) / batch_size),
            desc="Uploading batches",
        ):
            records = [
                {
                    "properties": {key: batch[key][i] for key in batch.keys()},
                    "uuid": generate_uuid5(batch["judgement_id"][i]),
                }
                for i in range(len(batch["judgement_id"]))
            ]
            db.insert_batch(
                collection=db.judgements_collection,
                objects=records,
            )


if __name__ == "__main__":
    typer.run(main)
