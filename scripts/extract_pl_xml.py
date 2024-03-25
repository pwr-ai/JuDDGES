import math
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Any

import typer
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from loguru import logger
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, ConfigurationError
from tenacity import (
    wait_random_exponential,
    retry_if_exception_type,
    stop_after_attempt,
    retry,
    retry_if_exception_message,
    retry_all,
)
from tqdm import tqdm

from juddges.preprocessing.pl_court_parser import SimplePlJudgementsParser

BATCH_SIZE = 100
INGEST_JOBS = 6

load_dotenv("secrets.env", verbose=True)


def main(
    dataset_dir: Path = typer.Option(..., help="Path to the dataset directory"),
    target_dir: Path = typer.Option(..., help="Path to the target directory"),
    num_proc: Optional[int] = typer.Option(None, help="Number of processes to use"),
    ingest: bool = typer.Option(False, help="Ingest the dataset to MongoDB"),
    mongo_uri: Optional[str] = typer.Option(None, envvar="MONGO_URI"),
    mongo_batch_size: int = typer.Option(BATCH_SIZE),
    ingest_jobs: int = typer.Option(INGEST_JOBS),
) -> None:
    target_dir.parent.mkdir(exist_ok=True, parents=True)
    ds = load_dataset("parquet", name="pl_judgements", data_dir=dataset_dir)
    num_shards = len(ds["train"].info.splits["train"].shard_lengths)
    parser = SimplePlJudgementsParser()
    ds = (
        ds["train"]
        .select_columns(
            ["_id", "date", "type", "excerpt", "content"]
        )  # leave only most important columns
        .filter(lambda x: x["content"] is not None)
        .map(parser, input_columns="content", num_proc=num_proc)
    )
    ds.save_to_disk(target_dir, num_shards=num_shards)

    if ingest:
        assert mongo_uri is not None
        ds = ds.with_format(columns=["_id"] + parser.schema)
        _ingest_dataset(ds, mongo_uri, mongo_batch_size, ingest_jobs)


def _ingest_dataset(dataset: Dataset, mongo_uri: str, batch_size: int, num_jobs: int) -> None:
    """Uploads the dataset to MongoDB."""
    num_batches = math.ceil(dataset.num_rows / batch_size)

    worker = IngestWorker(mongo_uri)
    with Pool(num_jobs) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    worker,
                    dataset.iter(batch_size=batch_size),
                ),
                total=num_batches,
                desc="Ingesting",
            )
        )


class IngestWorker:
    def __init__(self, mongo_uri: str):
        self.mongo_uri = mongo_uri

    @retry(
        wait=wait_random_exponential(multiplier=1, min=4, max=30),
        retry=retry_all(
            retry_if_exception_type(ConfigurationError),
            retry_if_exception_message(match="DNS operation timed out"),
        ),
        stop=stop_after_attempt(5),
    )
    def __call__(self, batch: dict[str, list[Any]]) -> None:
        client: MongoClient[dict[str, Any]] = MongoClient(self.mongo_uri)
        collection = client["juddges"]["judgements"]

        ids = batch.pop("_id")
        ingest_batch = [
            UpdateOne({"_id": ids[i]}, {"$set": {col: batch[col][i] for col in batch.keys()}})
            for i in range(len(ids))
        ]

        try:
            collection.bulk_write(ingest_batch, ordered=False)
        except BulkWriteError as err:
            logger.error(err)


if __name__ == "__main__":
    typer.run(main)
