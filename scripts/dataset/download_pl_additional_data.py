import math
import multiprocessing
from enum import Enum
from typing import Any, Optional

import typer
from dotenv import load_dotenv
from loguru import logger
from pymongo.results import BulkWriteResult
from requests import ConnectionError, HTTPError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

from juddges.data.database import BatchDatabaseUpdate, BatchedDatabaseCursor, get_mongo_collection
from juddges.data.pl_court_api import DataNotFoundError, PolishCourtAPI

N_JOBS = 6
BATCH_SIZE = 100

load_dotenv()


class DataType(Enum):
    CONTENT = "content"
    DETAILS = "details"


def main(
    mongo_uri: str = typer.Option(..., envvar="MONGO_URI"),
    data_type: DataType = typer.Option(DataType.CONTENT),
    batch_size: int = typer.Option(BATCH_SIZE),
    n_jobs: int = typer.Option(N_JOBS),
    last_update_from: Optional[str] = typer.Option(None, help="Format: YYYY-MM-DD"),
    dry: bool = typer.Option(False),
) -> None:
    api = PolishCourtAPI()

    # checks whether database misses any field present in the API schema
    # (it tests for non-existing fields, not field with null value)
    query: dict[str, Any] = {
        "$or": [{field: {"$exists": False}} for field in api.schema[data_type.value]]
    }
    # if last_update_from is provided, it filters documents by last update date
    if last_update_from is not None:
        query = {"$and": [{"lastUpdate": {"$gte": last_update_from}}, query]}

    collection = get_mongo_collection()
    num_docs_to_update = collection.count_documents(query)
    logger.info(f"Found {num_docs_to_update} documents to update")

    if dry:
        return

    # fetch all ids at once to avoid cursor timeout
    cursor = collection.find(query, {"_id": 1}, batch_size=batch_size).limit(200)
    batched_cursor = BatchedDatabaseCursor(cursor=cursor, batch_size=batch_size, prefetch=True)
    batches = list(batched_cursor)

    download_data = AdditionalDataDownloader(data_type)
    download_data_and_update_db = BatchDatabaseUpdate(mongo_uri, download_data)

    num_batches = math.ceil(num_docs_to_update / batch_size)
    if n_jobs == 1:
        with tqdm(batches, total=num_batches, desc="Downloading and updating database") as pbar:
            for batch in pbar:
                res = download_data_and_update_db(batch)
                pbar.set_postfix(bulk_write_results_summary(res))
    else:
        with multiprocessing.Pool(n_jobs) as pool:
            with tqdm(
                pool.imap_unordered(
                    download_data_and_update_db,
                    batches,
                ),
                total=num_batches,
                desc="Downloading and updating database",
            ) as pbar:
                for res in pbar:
                    pbar.set_postfix(bulk_write_results_summary(res))

    assert collection.count_documents(query) == 0


class AdditionalDataDownloader:
    def __init__(self, data_type: DataType):
        self.data_type = data_type
        self.api = PolishCourtAPI()

    @retry(
        wait=wait_random_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type((HTTPError, ConnectionError)),
        stop=stop_after_attempt(5),
    )
    def __call__(self, doc: dict[str, Any]) -> dict[str, Any]:
        doc_id = str(doc["_id"])
        try:
            if self.data_type == DataType.CONTENT:
                return self.api.get_content(doc_id)
            elif self.data_type == DataType.DETAILS:
                return self.api.get_cleaned_details(doc_id)
            else:
                raise ValueError(f"Invalid field: {self.data_type.value}")
        except DataNotFoundError as err:
            logger.warning(err)
            return dict.fromkeys(self.api.schema[self.data_type.value], None)


def bulk_write_results_summary(res: BulkWriteResult) -> dict[str, int]:
    return {
        "matched_for_update": res.matched_count,
        "modified": res.modified_count,
        "upserted": res.upserted_count,
        "deleted": res.deleted_count,
    }


if __name__ == "__main__":
    typer.run(main)
