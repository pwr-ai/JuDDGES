import math
import multiprocessing
from typing import Optional, Any

import typer
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from juddges.data.database import get_mongo_collection, BatchedDatabaseCursor, BatchDatabaseUpdate
from juddges.preprocessing.pl_court_parser import SimplePlJudgementsParser

BATCH_SIZE = 100
INGEST_JOBS = 6

load_dotenv()


def main(
    mongo_uri: str = typer.Option(None, envvar="MONGO_URI"),
    batch_size: int = typer.Option(BATCH_SIZE),
    n_jobs: Optional[int] = typer.Option(None, help="Number of processes to use"),
) -> None:
    # find rows which have non-empty content field
    query = {"content": {"$ne": None}}
    collection = get_mongo_collection()
    num_docs_to_update = collection.count_documents(query)
    logger.info(f"There are {num_docs_to_update} documents to update")

    # fetch all ids at once to avoid cursor timeout
    cursor = collection.find(query, {"content": 1}, batch_size=batch_size)
    batched_cursor = BatchedDatabaseCursor(cursor=cursor, batch_size=batch_size, prefetch=False)

    parse_doc = ParseDoc()
    parse_doc_and_update_db = BatchDatabaseUpdate(mongo_uri, parse_doc)

    with multiprocessing.Pool(n_jobs) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    parse_doc_and_update_db,
                    batched_cursor,
                ),
                total=math.ceil(num_docs_to_update / batch_size),
            )
        )


class ParseDoc:
    def __init__(self) -> None:
        self.parser = SimplePlJudgementsParser()

    def __call__(self, doc: dict[str, Any]) -> dict[str, Any]:
        return self.parser(doc["content"])


if __name__ == "__main__":
    typer.run(main)
