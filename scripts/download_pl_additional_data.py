import math
from enum import Enum
from typing import Any

import typer
from dotenv import load_dotenv
from loguru import logger
from mpire.pool import WorkerPool
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError
from requests import HTTPError, ConnectionError
from tenacity import retry, wait_random_exponential, retry_if_exception_type, stop_after_attempt
from tqdm import tqdm

from juddges.data.models import get_mongo_collection
from juddges.data.pl_court_api import PolishCourtAPI, DataNotFoundError

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
) -> None:
    collection = get_mongo_collection()

    api = PolishCourtAPI()

    # find rows which are missing at least one field
    query = {"$or": [{field: {"$exists": False}} for field in api.schema[data_type.value]]}

    num_docs_to_update = collection.count_documents(query)
    logger.info(f"There are {num_docs_to_update} documents to update")

    # fetch all ids at once to avoid cursor timeout
    cursor = collection.find(query, {"_id": 1}, batch_size=batch_size)
    docs_to_update: list[str] = []
    for doc in tqdm(cursor, total=num_docs_to_update, desc="Fetching doc list"):
        docs_to_update.append(str(doc["_id"]))

    batched_docs_to_update = (
        docs_to_update[i : i + batch_size] for i in range(0, len(docs_to_update), batch_size)
    )

    download_update = DownloadDataAndUpdateDatabase(mongo_uri, data_type)
    with WorkerPool(n_jobs=n_jobs) as pool:
        pool.map_unordered(
            download_update,
            batched_docs_to_update,
            progress_bar=True,
            iterable_len=math.ceil(num_docs_to_update / batch_size),
        )


class DownloadDataAndUpdateDatabase:
    def __init__(self, mongo_uri: str, data_type: DataType):
        self.mongo_uri = mongo_uri
        self.data_type = data_type
        self.api = PolishCourtAPI()

    def __call__(self, *doc_ids: str) -> None:
        data_batch: list[UpdateOne] = []

        for d_id in doc_ids:
            fetched_data = self._download_data(d_id)
            data_batch.append(UpdateOne({"_id": d_id}, {"$set": fetched_data}))

        collection = get_mongo_collection(mongo_uri=self.mongo_uri)

        try:
            collection.bulk_write(data_batch)
        except BulkWriteError as err:
            logger.error(err)

    @retry(
        wait=wait_random_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type((HTTPError, ConnectionError)),
        stop=stop_after_attempt(5),
    )
    def _download_data(self, doc_id: str) -> dict[str, Any]:
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
