from pathlib import Path
import pandas as pd
import pymongo
from tqdm import tqdm
import typer

from loguru import logger
from juddges.utils.logging import setup_loguru

setup_loguru(extra={"script": __file__})


def main(
    db_uri: str = typer.Option(..., envvar="DB_URI"),
    log_file: Path = typer.Option(None, help="Log file to save the logs to."),
) -> None:
    setup_loguru(extra={"script": __file__}, log_file=log_file)
    logger.info("Running drop_dates_with_duplicated_documents.py with args:\n" + str(locals()))

    client = pymongo.MongoClient(db_uri)
    db = client["nsa"]
    dates_col = db["dates"]

    df = pd.DataFrame(dates_col.find().sort("date"))
    df = df.explode("document_ids")
    df = df[~df["document_ids"].isna()]
    duplicated = df[df.duplicated(subset="document_ids", keep=False)]
    dates_to_drop = duplicated["date"].unique()

    logger.info(f"Found {len(dates_to_drop)} dates with duplicated documents.")

    if len(dates_to_drop) == 0:
        logger.info("No dates with duplicated documents found.")
        return

    if not typer.confirm("Do you want to delete them?"):
        logger.info("Aborted.")
        raise typer.Abort()

    for date in tqdm(dates_to_drop):
        dates_col.delete_many({"date": date})


if __name__ == "__main__":
    typer.run(main)
