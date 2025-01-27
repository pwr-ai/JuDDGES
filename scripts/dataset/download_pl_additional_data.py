import math
import multiprocessing
from enum import Enum
from typing import Any, Optional

import typer
from dotenv import load_dotenv
from loguru import logger
from requests import ConnectionError, HTTPError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
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

    query: dict[str, Any] = {}
    if last_update_from is not None:
        # get all rows which were last updated after the given date
        query = {"lastUpdate": {"$gte": last_update_from}}
    else:
        # find rows which are missing at least one field
        query = {"$or": [{field: {"$exists": False}} for field in api.schema[data_type.value]]}

    collection = get_mongo_collection()
    num_docs_to_update = collection.count_documents(query)
    logger.info(f"There are {num_docs_to_update} documents to update")

    if dry:
        return

    # fetch all ids at once to avoid cursor timeout
    cursor = collection.find(query, {"_id": 1}, batch_size=batch_size)
    batched_cursor = BatchedDatabaseCursor(cursor=cursor, batch_size=batch_size, prefetch=True)
    batches = list(batched_cursor)

    download_data = AdditionalDataDownloader(data_type)
    download_data_and_update_db = BatchDatabaseUpdate(mongo_uri, download_data)

    with multiprocessing.Pool(n_jobs) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    download_data_and_update_db,
                    batches,
                ),
                total=math.ceil(num_docs_to_update / batch_size),
            )
        )

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


if __name__ == "__main__":
    typer.run(main)
