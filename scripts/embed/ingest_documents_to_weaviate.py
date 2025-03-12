import math
import os
from pathlib import Path

import typer
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from tqdm.auto import tqdm
from weaviate.util import generate_uuid5

from juddges.data.weaviate_db import WeaviateJudgmentsDatabase
from juddges.settings import ROOT_PATH
from juddges.utils.date_utils import process_judgement_dates

load_dotenv(dotenv_path=ROOT_PATH / ".env", override=True)

WV_HOST = os.environ["WV_HOST"]
WV_PORT = os.environ["WV_PORT"]
WV_GRPC_PORT = os.environ["WV_GRPC_PORT"]
WV_API_KEY = os.environ["WV_API_KEY"]

BATCH_SIZE = 64
NUM_PROC = int(os.getenv("NUM_PROC", 1))

logger.info(f"Connecting to Weaviate at {WV_HOST}:{WV_PORT} (gRPC: {WV_GRPC_PORT})")


def main(
    dataset_name: Path = typer.Option(...),
    batch_size: int = typer.Option(BATCH_SIZE),
) -> None:
    dataset = load_dataset(str(dataset_name))["train"]
    with WeaviateJudgmentsDatabase(WV_HOST, WV_PORT, WV_GRPC_PORT, WV_API_KEY) as db:
        for batch in tqdm(
            dataset.iter(batch_size=batch_size),
            total=math.ceil(len(dataset) / batch_size),
            desc="Uploading batches",
        ):
            records = []
            for i in range(len(batch["judgement_id"])):
                properties = {key: batch[key][i] for key in batch.keys()}
                # Process dates to RFC3339 format
                properties = process_judgement_dates(properties)
                records.append(
                    {
                        "properties": properties,
                        "uuid": generate_uuid5(batch["judgement_id"][i]),
                    }
                )

            db.insert_batch(
                collection=db.judgements_collection,
                objects=records,
            )


if __name__ == "__main__":
    typer.run(main)
