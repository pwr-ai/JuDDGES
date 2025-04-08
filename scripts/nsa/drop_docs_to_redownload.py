from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pymongo
import typer
from loguru import logger
from pymongo.collection import Collection

from juddges.utils.logging import setup_loguru

setup_loguru(extra={"script": __file__})


def main(
    db_uri: str = typer.Option(..., envvar="DB_URI"),
    redownload_days_back: int = typer.Option(
        720, help="Days back to redownload pages from. Defaults to 720 (2 years)."
    ),
    log_file: Path = typer.Option(None, help="Log file to save the logs to."),
) -> None:
    setup_loguru(extra={"script": __file__}, log_file=log_file)
    logger.info("Running drop_docs_to_redownload.py with args:\n" + str(locals()))

    client = pymongo.MongoClient(db_uri)
    db = client["nsa"]
    dates_col = db["dates"]
    docs_col = db["document_pages"]

    docs_to_redownload = get_docs_to_redownload(dates_col, docs_col, redownload_days_back)
    logger.info(f"Found {len(docs_to_redownload)} documents to redownload.")

    if len(docs_to_redownload) == 0:
        logger.info("No documents to redownload found.")
        return

    if not typer.confirm("Do you want to delete them?"):
        logger.info("Aborted.")
        raise typer.Abort()

    result = docs_col.delete_many({"doc_id": {"$in": docs_to_redownload}})
    logger.info(f"Deleted {result.deleted_count} documents")


def get_docs_to_redownload(
    dates_col: Collection, docs_col: Collection, days_back: int
) -> list[str]:
    df = pd.DataFrame(dates_col.find())
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= (datetime.now() - timedelta(days=days_back))]
    df = df.explode("document_ids")
    df = df[~df["document_ids"].isna()]
    done_docs = {doc["doc_id"] for doc in docs_col.find({}, {"_id": 0, "doc_id": 1})}
    return [doc_id for doc_id in df["document_ids"].tolist() if doc_id in done_docs]


if __name__ == "__main__":
    typer.run(main)
