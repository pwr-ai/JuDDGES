import math
import multiprocessing
from typing import Any, Optional

import typer
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from juddges.data.database import BatchDatabaseUpdate, BatchedDatabaseCursor, get_mongo_collection
from juddges.preprocessing.pl_court_parser import SimplePlJudgementsParser

BATCH_SIZE = 100
load_dotenv()


def main(
    mongo_uri: str = typer.Option(None, envvar="MONGO_URI"),
    batch_size: int = typer.Option(BATCH_SIZE),
    n_jobs: int = typer.Option(1, help="Number of processes to use"),
    last_update_from: Optional[str] = typer.Option(None, help="Format: YYYY-MM-DD"),
) -> None:
    query: dict[str, Any] = {"$and": [{"content": {"$ne": None}}, {"text": {"$exists": False}}]}

    if last_update_from is not None:
        # get all rows which were last updated after the given date
        query = {
            "$and": [
                query,
                {"lastUpdate": {"$gte": last_update_from}},
            ]
        }

    collection = get_mongo_collection()
    logger.info("Counting documents to update...")
    num_docs_to_update = collection.count_documents(query)
    logger.info(f"There are {num_docs_to_update} documents to update")

    if num_docs_to_update == 0:
        return

    # fetch all ids at once to avoid cursor timeout
    cursor = collection.find(query, {"content": 1}, batch_size=batch_size)
    batched_cursor = BatchedDatabaseCursor(cursor=cursor, batch_size=batch_size, prefetch=False)

    parse_doc = ParseDoc()
    parse_doc_and_update_db = BatchDatabaseUpdate(mongo_uri, parse_doc)

    num_batches = math.ceil(num_docs_to_update / batch_size)
    if n_jobs > 1:
        with multiprocessing.Pool(n_jobs) as pool:
            list(
                tqdm(
                    pool.imap_unordered(
                        parse_doc_and_update_db,
                        batched_cursor,
                    ),
                    total=num_batches,
                    desc="Extracting documents",
                )
            )
    else:
        for doc in tqdm(batched_cursor, total=num_docs_to_update, desc="Extracting documents"):
            parse_doc_and_update_db(doc)


class ParseDoc:
    def __init__(self) -> None:
        self.parser = SimplePlJudgementsParser()

    def __call__(self, doc: dict[str, Any]) -> dict[str, Any]:
        return self.parser(doc["content"])


if __name__ == "__main__":
    typer.run(main)
