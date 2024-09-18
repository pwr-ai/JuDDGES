import pandas as pd
import pymongo
import typer
import urllib3
from tqdm import tqdm

from juddges.settings import NSA_DATA_PATH

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DB_URI = "mongodb://localhost:27017/"


def fetch_documents(collection, batch_size=5000):
    cursor = collection.find().batch_size(batch_size)
    for doc in cursor:
        doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
        yield doc


def write_to_parquet_in_chunks(file_path, collection, batch_size=5000, chunk_size=100000):
    file_path.mkdir(parents=True, exist_ok=True)
    buffer = []
    chunk_index = 0

    for doc in tqdm(fetch_documents(collection, batch_size)):
        buffer.append(doc)
        if len(buffer) >= chunk_size:
            df = pd.DataFrame(buffer)
            chunk_file = file_path.parent / f"{file_path.stem}_chunk_{chunk_index}.parquet"
            df.to_parquet(chunk_file, engine="pyarrow", compression="snappy")
            buffer = []
            chunk_index += 1

    if buffer:
        df = pd.DataFrame(buffer)
        chunk_file = file_path.parent / f"{file_path.stem}_chunk_{chunk_index}.parquet"
        df.to_parquet(chunk_file, engine="pyarrow", compression="snappy")


def main(
    db_uri: str = typer.Option(DB_URI),
) -> None:
    client = pymongo.MongoClient(db_uri)
    db = client["nsa"]
    docs_col = db["document_pages"]
    errors_col = db["document_pages_errors"]

    # Save document pages in Parquet format
    docs_output_path = NSA_DATA_PATH / "pages" / "pages.parquet"
    write_to_parquet_in_chunks(docs_output_path, docs_col)

    # Save document errors in Parquet format
    errors_output_path = NSA_DATA_PATH / "errors" / "errors.parquet"
    write_to_parquet_in_chunks(errors_output_path, errors_col)


typer.run(main)
