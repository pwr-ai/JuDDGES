from pathlib import Path
from typing import Any, Mapping, Sequence

import typer
from dotenv import load_dotenv
from loguru import logger
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.command_cursor import CommandCursor

from juddges.data.utils import path_safe_udate, save_jsonl

load_dotenv("secrets.env", verbose=True)


def agg_sample(collection: Collection[Any], size: int) -> CommandCursor[Any]:
    """Aggregate a sample of records from a MongoDB collection.

    Args:
        collection (Collection): MongoDB collection
        size (itn): Number of records to sample

    Returns:
        CommandCursor: MongoDB cursor with sampled records
    """
    logger.info(f"Sampling {size} records...")
    agg: Sequence[Mapping[str, Any]] = [
        {"$match": {"text": {"$exists": True}}},
        {"$sample": {"size": size}},
    ]
    return collection.aggregate(agg)


def main(
    size: int = typer.Option(help="Number of records to sample"),
    # FIXME `seed` mongo do not support seed in sampling
    # seed: int = typer.Option(default=None, help="Random seed"),
    mongo_uri: str = typer.Option(..., envvar="MONGO_URI"),
    out: str = typer.Option(default=None, help="Output file path"),
) -> None:
    if out is None:
        from juddges.settings import SAMPLE_DATA_PATH

        out = SAMPLE_DATA_PATH / f"judgements_sample{size}_{path_safe_udate()}.jsonl"
        logger.warning("Output file path not provided, using the default `out`: ./")

    logger.info("Connecting to MongoDB...")
    client: MongoClient[Any] = MongoClient(mongo_uri)

    logger.info("Fetching the `judgements` collection...")
    collection = client["juddges"]["judgements"]

    sample_records = agg_sample(collection, size=size)

    out_path = Path(out)
    if ".jsonl" in out_path.suffixes or ".jsonlines" in out_path.suffixes:
        logger.info(f"Saving sample to {out_path}...")
        save_jsonl(sample_records, out_path)
    else:
        raise NotImplementedError("Only JSONL output is supported")


if __name__ == "__main__":
    typer.run(main)
