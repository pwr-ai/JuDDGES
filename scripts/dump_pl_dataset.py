import sys
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from dotenv import load_dotenv
from loguru import logger
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from tqdm import tqdm, trange
from pyarrow.parquet import ParquetDataset

BATCH_SIZE = 100
CHUNK_SIZE = 50_000

load_dotenv("secrets.env", verbose=True)


def main(
    mongo_uri: str = typer.Option(..., envvar="MONGO_URI"),
    batch_size: int = typer.Option(BATCH_SIZE),
    chunk_size: int = typer.Option(CHUNK_SIZE),
    file_name: Path = typer.Option(..., exists=False),
    filter_empty_content: bool = typer.Option(False),
) -> None:
    file_name.parent.mkdir(exist_ok=True, parents=True)

    client: MongoClient[dict[str, Any]] = MongoClient(mongo_uri, server_api=ServerApi("1"))
    collection = client["juddges"]["judgements"]
    client.admin.command("ping")

    if filter_empty_content:
        query = {"content": {"$ne": True}}
    else:
        query = {}

    num_docs = collection.count_documents(query)

    dumped_data = file_name.parent.glob("*.parquet")
    start_offset = 0
    if dumped_data:
        logger.warning(f"Found {len(list(dumped_data))} files in {file_name.parent}")
        if typer.confirm("Do you want to continue previous dump?"):
            dataset = ParquetDataset(file_name.parent)
            start_offset = sum(p.count_rows() for p in dataset.fragments)
        else:
            logger.error("Delete data to start a new data dump")
            sys.exit(1)

    logger.info(f"Starting from {start_offset}-th document, batch no. {start_offset // chunk_size}")
    for offset in trange(start_offset, num_docs, chunk_size, desc="Chunks"):
        docs = list(
            tqdm(
                collection.find(query, batch_size=batch_size).skip(offset).limit(chunk_size),
                total=chunk_size,
                leave=False,
                desc="Documents in chunk",
            )
        )
        i = offset // chunk_size
        dumped_f_name = save_docs(docs, file_name, i)
        logger.info(f"Dumped {i}-th batch of documents to {dumped_f_name}")


def save_docs(docs: list[dict[str, Any]], file_name: Path, i: int | None) -> Path:
    if i is not None:
        file_name = file_name.with_name(f"{file_name.stem}_{i:02d}{file_name.suffix}")
    pd.DataFrame(docs).to_parquet(file_name)
    return file_name


if __name__ == "__main__":
    typer.run(main)
