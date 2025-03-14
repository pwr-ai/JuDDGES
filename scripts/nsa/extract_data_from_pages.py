from pathlib import Path
import pandas as pd
import pymongo
import typer
from tqdm import tqdm
import pyarrow.parquet as pq
from loguru import logger


from juddges.settings import NSA_DATA_PATH
from juddges.data.nsa.extractor import NSADataExtractor
from juddges.utils.logging import setup_loguru

setup_loguru(extra={"script": __file__})

OUTPUT_PATH = NSA_DATA_PATH / "dataset"
N_JOBS = 10


def main(
    db_uri: str = typer.Option(..., envvar="DB_URI"),
    n_jobs: int = typer.Option(N_JOBS),
    log_file: Path = typer.Option(None, help="Log file to save the logs to."),
) -> None:
    setup_loguru(extra={"script": __file__}, log_file=log_file)
    logger.info("Running extract_data_from_pages.py with args:\n" + str(locals()))

    client = pymongo.MongoClient(db_uri, compressors="snappy")
    db = client["nsa"]
    docs_col = db["document_pages"]

    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    # Check if OUTPUT_PATH is not empty and prompt for confirmation before removing files
    if any(OUTPUT_PATH.iterdir()):
        confirm = typer.confirm(f"Output directory {OUTPUT_PATH} is not empty. Remove all files?")
        if confirm:
            for file in OUTPUT_PATH.iterdir():
                if file.is_file():
                    file.unlink()
            logger.info(f"Removed all files from {OUTPUT_PATH}")
        else:
            typer.Abort("Output directory is not empty. Exiting...")

    extractor = NSADataExtractor()
    for i, df in tqdm(enumerate(fetch_documents(docs_col))):
        data = extractor.extract_data_from_pages_to_pyarrow(df["page"], df["doc_id"], n_jobs=n_jobs)
        pq.write_table(data, OUTPUT_PATH / f"data_{i}.parquet")
        del df
        del data


def fetch_documents(collection, batch_size=5000, chunk_size=50000):
    buffer = []
    cursor = collection.find().batch_size(batch_size)
    for doc in tqdm(cursor, desc="Fetching documents"):
        doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
        buffer.append(doc)
        if len(buffer) >= chunk_size:
            yield pd.DataFrame(buffer)
            buffer = []
    if buffer:
        yield pd.DataFrame(buffer)


if __name__ == "__main__":
    typer.run(main)
