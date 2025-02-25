from pathlib import Path
import pandas as pd
import pymongo
import typer
from tqdm import tqdm
from loguru import logger
from juddges.utils.logging import setup_loguru

from juddges.settings import NSA_DATA_PATH


def main(
    db_uri: str = typer.Option(..., envvar="DB_URI"),
    log_file: Path = typer.Option(None, help="Log file to save the logs to."),
) -> None:
    setup_loguru(extra={"script": __file__}, log_file=log_file)
    logger.info("Running save_pages_from_db_to_file.py with args:\n" + str(locals()))

    client = pymongo.MongoClient(db_uri)
    db = client["nsa"]
    docs_col = db["document_pages"]

    docs_output_path = NSA_DATA_PATH / "pages" / "pages.parquet"
    logger.info(f"Saving document pages in Parquet format to {docs_output_path}...")
    write_to_parquet_in_chunks(docs_output_path, docs_col)


def fetch_documents(collection, batch_size=5000):
    cursor = collection.find().batch_size(batch_size)
    for doc in cursor:
        doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
        yield doc


def write_to_parquet_in_chunks(file_path, collection, batch_size=5000, chunk_size=50000):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = []
    chunk_index = 0

    for doc in tqdm(fetch_documents(collection, batch_size)):
        buffer.append(doc)
        if len(buffer) >= chunk_size:
            df = pd.DataFrame(buffer)
            chunk_file = file_path.parent / f"{file_path.stem}_chunk_{chunk_index}.parquet"
            df.to_parquet(chunk_file, engine="pyarrow", compression="snappy")
            del df
            buffer = []
            chunk_index += 1

    if buffer:
        df = pd.DataFrame(buffer)
        chunk_file = file_path.parent / f"{file_path.stem}_chunk_{chunk_index}.parquet"
        df.to_parquet(chunk_file, engine="pyarrow", compression="snappy")


typer.run(main)
