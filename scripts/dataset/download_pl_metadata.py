from typing import Optional

import typer
from dotenv import load_dotenv
from loguru import logger
from mpire.pool import WorkerPool
from pymongo.errors import BulkWriteError
from requests import HTTPError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from juddges.data.database import get_mongo_collection
from juddges.data.pl_court_api import PolishCourtAPI

N_JOBS = 8
BATCH_SIZE = 1_000

load_dotenv()


def main(
    mongo_uri: str = typer.Option(..., envvar="MONGO_URI"),
    batch_size: int = typer.Option(BATCH_SIZE),
    n_jobs: int = typer.Option(N_JOBS),
    limit: Optional[int] = typer.Option(None),
) -> None:
    api = PolishCourtAPI()
    total_judgements = api.get_number_of_judgements()
    logger.info(f"Total judgements found: {total_judgements}")

    if limit is not None:
        total_judgements = min(total_judgements, limit)

    offsets = list(range(0, total_judgements, batch_size))
    download_metadata = MetadataDownloader(mongo_uri, batch_size)
    with WorkerPool(n_jobs=n_jobs) as pool:
        pool.map_unordered(download_metadata, offsets, progress_bar=True)


class MetadataDownloader:
    def __init__(self, mongo_uri: str, batch_size: int):
        self.mongo_uri = mongo_uri
        self.batch_size = batch_size

    @retry(
        wait=wait_random_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type(HTTPError),
        stop=stop_after_attempt(5),
    )
    def __call__(self, offset: int) -> None:
        collection = get_mongo_collection(mongo_uri=self.mongo_uri)

        params = {
            "sort": "date-asc",
            "limit": self.batch_size,
            "offset": offset,
        }
        api = PolishCourtAPI()
        judgements = api.get_judgements(params)

        for item in judgements:
            item["_id"] = item.pop("id")

        # ignore when come across duplicated documents
        try:
            collection.insert_many(judgements, ordered=False)
        except BulkWriteError as err:
            if any(x["code"] != 11000 for x in err.details["writeErrors"]):
                raise
            else:
                logger.warning(
                    "found {num_duplicated} judgements duplicated",
                    num_duplicated=len(err.details["writeErrors"]),
                )


if __name__ == "__main__":
    typer.run(main)
