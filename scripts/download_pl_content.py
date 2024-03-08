import math
from typing import Generator, Any

import typer
from dotenv import load_dotenv
from loguru import logger
from mpire.pool import WorkerPool
from pymongo import MongoClient, UpdateOne
from pymongo.cursor import Cursor
from pymongo.errors import BulkWriteError
from pymongo.server_api import ServerApi
from requests import HTTPError
from tenacity import retry, wait_random_exponential, retry_if_exception_type, stop_after_attempt

from juddges.data.pl_court_api import PolishCourtAPI

N_JOBS = 8
BATCH_SIZE = 100

load_dotenv("secrets.env", verbose=True)


def main(
    mongo_uri: str = typer.Option(..., envvar="MONGO_URI"),
    batch_size: int = typer.Option(BATCH_SIZE),
    n_jobs: int = typer.Option(N_JOBS),
) -> None:
    client: MongoClient[dict[str, Any]] = MongoClient(mongo_uri, server_api=ServerApi("1"))
    collection = client["juddges"]["judgements"]
    client.admin.command("ping")

    query = {"content": {"$exists": False}}
    num_docs_without_content = collection.count_documents(query)
    logger.info(f"There are {num_docs_without_content} documents without content")

    cursor = collection.find(query, batch_size=batch_size)

    docs_to_update = yield_batches(cursor, batch_size)
    download_content = ContentDownloader(mongo_uri)
    with WorkerPool(n_jobs=n_jobs) as pool:
        pool.map_unordered(
            download_content,
            docs_to_update,
            progress_bar=True,
            iterable_len=math.ceil(num_docs_without_content / batch_size),
        )


class ContentDownloader:
    def __init__(self, mongo_uri: str):
        self.mongo_uri = mongo_uri

    def __call__(self, *doc_ids: str) -> None:
        data_batch: list[UpdateOne] = []

        for d_id in doc_ids:
            content = self._download_content(d_id)
            data_batch.append(UpdateOne({"_id": d_id}, {"$set": {"content": content}}))

        client: MongoClient[dict[str, Any]] = MongoClient(self.mongo_uri)
        collection = client["juddges"]["judgements"]

        try:
            collection.bulk_write(data_batch)
        except BulkWriteError as err:
            logger.error(err)

    @retry(
        wait=wait_random_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type(HTTPError),
        stop=stop_after_attempt(5),
    )
    def _download_content(self, doc_id: str) -> str | None:
        api = PolishCourtAPI()
        try:
            return api.get_content(doc_id)
        except HTTPError as err:
            if err.response.status_code == 404:
                logger.warning("Found no content for judgement {id}", id=doc_id)
                return None
            else:
                raise


def yield_batches(
    cursor: Cursor[dict[str, Any]], batch_size: int
) -> Generator[list[str], None, None]:
    """Generates batches of data from pymongo.Cursor.
    Credit: https://stackoverflow.com/a/61809417
    """

    batch: list[str] = []
    for i, row in enumerate(cursor):
        if i % batch_size == 0 and i > 0:
            yield batch
            del batch[:]
        batch.append(str(row["_id"]))
    yield batch


if __name__ == "__main__":
    typer.run(main)
