from typing import Any, Optional

import typer
from dotenv import load_dotenv
from loguru import logger
from mpire.pool import WorkerPool
from pymongo import ReplaceOne
from requests import HTTPError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

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
    date_from: Optional[str] = typer.Option(None, help="Format: YYYY-MM-DD"),
    date_to: Optional[str] = typer.Option(None, help="Format: YYYY-MM-DD"),
    last_update_from: Optional[str] = typer.Option(None, help="Format: YYYY-MM-DD"),
    last_update_to: Optional[str] = typer.Option(None, help="Format: YYYY-MM-DD"),
    dry: bool = typer.Option(False),
) -> None:
    params = {}
    if date_from is not None:
        params["dateFrom"] = date_from
    if date_to is not None:
        params["dateTo"] = date_to
    if last_update_from is not None:
        params["lastUpdateFrom"] = last_update_from
    if last_update_to is not None:
        params["lastUpdateTo"] = last_update_to

    api = PolishCourtAPI()
    total_judgments = api.get_number_of_judgements(params=params)
    logger.info(f"Total judgements found: {total_judgments:_}")

    if limit is not None:
        total_judgments = min(total_judgments, limit)

    logger.info(
        f"Downloading {total_judgments:_} judgements: (batch_size={batch_size}, n_jobs={n_jobs}, params={params})"
    )
    if dry:
        return

    offsets = list(range(0, total_judgments, batch_size))
    download_metadata = MetadataDownloader(
        mongo_uri=mongo_uri,
        batch_size=batch_size,
        params=params,
    )
    with WorkerPool(n_jobs=n_jobs) as pool:
        pool.map_unordered(download_metadata, offsets, progress_bar=True)


class MetadataDownloader:
    def __init__(
        self,
        mongo_uri: str,
        batch_size: int,
        params: dict[str, Any],
    ):
        self.mongo_uri = mongo_uri
        self.batch_size = batch_size
        self.params = params

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

        assert not any(k in params.keys() for k in self.params.keys())
        params |= self.params

        api = PolishCourtAPI()
        judgments = api.get_judgments(params)

        for item in judgments:
            item["_id"] = item.pop("id")

        # update existing documents or insert new ones
        write_ops = [ReplaceOne({"_id": item["_id"]}, item, upsert=True) for item in judgments]
        result = collection.bulk_write(write_ops, ordered=False)
        logger.info(
            "Write results: {upserted} upserted, {modified} modified",
            upserted=result.upserted_count,
            modified=result.modified_count,
        )


if __name__ == "__main__":
    typer.run(main)
