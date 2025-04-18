from pathlib import Path

import pandas as pd
import pymongo
import typer
from loguru import logger
from tqdm import tqdm

from juddges.utils.logging import setup_loguru

setup_loguru(extra={"script": __file__})


def main(
    db_uri: str = typer.Option(..., envvar="DB_URI"),
    log_file: Path = typer.Option(None, help="Log file to save the logs to."),
) -> None:
    setup_loguru(extra={"script": __file__}, log_file=log_file)
    logger.info("Running drop_duplicated_document_pages.py with args:\n" + str(locals()))

    client = pymongo.MongoClient(db_uri)
    db = client["nsa"]
    pages_col = db["document_pages"]

    logger.info("Checking for duplicates. This may take a while...")
    cursor = pages_col.aggregate(
        [{"$group": {"_id": "$page", "count": {"$sum": 1}}}, {"$match": {"count": {"$gt": 1}}}]
    )
    duplicates = pd.DataFrame(list(cursor))

    if len(duplicates) == 0:
        logger.info("No duplicates found.")
        return

    # ask user for confirmation
    logger.info(f"Found {len(duplicates)} duplicates.")
    if not typer.confirm("Do you want to delete them?"):
        logger.info("Aborted.")
        return

    # delete duplicates
    logger.info("Deleting duplicates...")
    for page in tqdm(duplicates["_id"]):
        pages_col.delete_many({"page": page})


typer.run(main)
