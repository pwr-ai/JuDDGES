import json

import pandas as pd
import pymongo
import typer
import urllib3
from tqdm import tqdm

from juddges.settings import NSA_DATA_PATH

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DB_URI = "mongodb://localhost:27017/"


def fetch_documents(collection, batch_size=1000):
    cursor = collection.find().batch_size(batch_size)
    for doc in cursor:
        doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
        yield doc


def write_to_jsonl_in_chunks(file_path, collection, batch_size=1000):
    with open(file_path, "w") as file:
        for doc in tqdm(fetch_documents(collection, batch_size)):
            file.write(json.dumps(doc) + "\n")  # Write each document as a JSON line


def main(
    db_uri: str = typer.Option(DB_URI),
) -> None:
    client = pymongo.MongoClient(db_uri)
    db = client["nsa"]
    docs_col = db["document_pages"]
    errors_col = db["document_pages_errors"]

    NSA_DATA_PATH.mkdir(parents=True, exist_ok=True)

    # Save document pages in JSONL format
    docs_output_path = NSA_DATA_PATH / "pages.jsonl"
    write_to_jsonl_in_chunks(docs_output_path, docs_col)

    # Save document errors in JSONL format
    errors_output_path = NSA_DATA_PATH / "errors.jsonl"
    write_to_jsonl_in_chunks(errors_output_path, errors_col)


typer.run(main)
