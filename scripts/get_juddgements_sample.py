import typer
from pathlib import Path
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.command_cursor import CommandCursor
from dotenv import load_dotenv
from loguru import logger

from juddges.data.utils import save_jsonl, path_safe_udate

load_dotenv("secrets.env", verbose=True)


def agg_sample(collection: Collection, size: int) -> CommandCursor:
    """Aggregate a sample of records from a MongoDB collection.

    Args:
        collection (Collection): MongoDB collection
        size (itn): Number of records to sample

    Returns:
        CommandCursor: MongoDB cursor with sampled records
    """
    logger.info(f"Sampling {size} records...")
    agg = [
        {
            "$match": {
                "text": {
                    "$exists": True
                }
            }
        },
        {
            "$sample": {
                "size": size
            }
        },
    ]
    return collection.aggregate(agg)


def main(
    size: int = typer.Option(help="Number of records to sample"),
    # FIXME `seed` mongo do not support seed in sampling
    seed: int = typer.Option(default=None, help="Random seed"),
    mongo_uri: str = typer.Option(..., envvar="MONGO_URI"),
    out: str = typer.Option(default=None, help="Output file path"),
):
    if out is None:
        # FIXME: data dir from settings when available, fix logger info
        # from juddges.settings import PL_JUDGEMENTS_PATH
        # out = PL_JUDGEMENTS_PATH / f"judgements_sample{size}_{path_safe_udate()}.jsonl"
        logger.warning("Output file path not provided, using the default `out`: ./")
        out = f"judgements_sample{size}_{path_safe_udate()}.jsonl"

    logger.info("Connecting to MongoDB...")
    client = MongoClient(mongo_uri)

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
