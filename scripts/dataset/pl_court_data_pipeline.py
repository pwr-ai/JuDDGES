import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pymongo
from dotenv import load_dotenv
from huggingface_hub import DatasetCard, DatasetCardData
from loguru import logger
from requests import HTTPError

from juddges.data.database import MongoInterface
from juddges.data.pl_court_api import DataNotFoundError, PolishCourtAPI
from juddges.preprocessing.pl_court_parser import SimplePlJudgementsParser
from juddges.settings import PL_JUDGEMENTS_PATH, PL_JUDGEMENTS_PATH_RAW
from juddges.utils.hf import get_parquet_num_rows, push_dataset_dir_to_hub
from juddges.utils.pipeline import RetryOnException, get_recent_successful_flow_date
from prefect import flow, runtime, task, unmapped
from prefect.task_runners import ThreadPoolTaskRunner
from prefect.tasks import exponential_backoff

load_dotenv()

MAX_CONCURRENT_WORKERS = 10
BATCH_SIZE = 50
COURT_ID_2_NAME_FILE = "data/datasets/pl/court_id_2_name.csv"
MONGO_URI = os.environ["MONGO_URI"]
MONGO_DB_NAME = os.environ["MONGO_DB_NAME"]
MONGO_COLLECTION_NAME = "pl-court"

REPO_ID = "JuDDGES/pl-court-raw"
SHARD_SIZE = 25_000
DATASET_CARD_TEMPLATE = PL_JUDGEMENTS_PATH / "readme/raw/README.md"
DATASET_CARD_TEMPLATE_ASSETS = PL_JUDGEMENTS_PATH / "readme/raw/README_files"


@flow(task_runner=ThreadPoolTaskRunner(max_workers=MAX_CONCURRENT_WORKERS), log_prints=True)
def update_pl_court_data(
    date_from: str | None = None,
    batch_size: int = BATCH_SIZE,
) -> None:
    latest_successful_flow_date = get_recent_successful_flow_date()
    if latest_successful_flow_date is not None:
        date_from = latest_successful_flow_date
        logger.info(f"Using date_from: {date_from} from latest successful flow")
    elif date_from is None:
        raise ValueError(
            "You have to either provide a date_from param or have some documents in the database"
        )
    else:
        logger.info(f"Using date_from: {date_from} from param")

    n_judgements = fetch_number_of_judgements(date_from)
    logger.info(f"Found {n_judgements} judgements")

    court_department_id_mapper = MapCourtDepartmentIds2Names(mapping_file=COURT_ID_2_NAME_FILE)
    doc_parser = ParseDoc()

    for offset in range(0, n_judgements, batch_size):
        judgments = fetch_judgment_metadata(
            offset=offset,
            batch_size=batch_size,
            date_from=date_from,
        )
        processed_judgements = fetch_judgment_data.map(
            judgments,
            unmapped(court_department_id_mapper),
            unmapped(doc_parser),
        ).result()
        save_or_update_judgments_in_db(judgements=processed_judgements)

    dump_dataset(
        file_name=PL_JUDGEMENTS_PATH_RAW / "pl_court_data.parquet",
        shard_size=SHARD_SIZE,
    )

    push_dataset_to_hub()


class MapCourtDepartmentIds2Names:
    def __init__(self, mapping_file: Path | str) -> None:
        self.mapping: dict[tuple[int, int], dict[str, str]] = (
            pd.read_csv(mapping_file).set_index(["court_id", "dep_id"]).to_dict("index")
        )

    @task(name="map_court_id_2_court_name")
    def __call__(self, doc: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.mapping[(int(doc["courtId"]), int(doc["departmentId"]))]
        except KeyError:
            logger.warning(
                "Missing mapping for (courtId, departmentID): "
                f"{doc['courtId'], doc['departmentId']}"
            )
            return {"court_name": None, "department_name": None}


class ParseDoc:
    def __init__(self) -> None:
        self.parser = SimplePlJudgementsParser()

    @task(name="parse_xml_content")
    def __call__(self, doc: dict[str, Any]) -> dict[str, Any]:
        if doc["content"] is not None:
            return self.parser(doc["content"])
        else:
            return {}


@task(name="fetch_judgment_data")
def fetch_judgment_data(
    judgment: dict[str, Any],
    court_department_id_mapper: MapCourtDepartmentIds2Names,
    doc_parser: ParseDoc,
) -> dict[str, Any]:
    judgment |= court_department_id_mapper(doc=judgment)
    judgment |= fetch_judgment_details(judgment_id=judgment["id"])
    judgment |= fetch_judgment_content(judgment_id=judgment["id"])
    judgment |= doc_parser(doc=judgment)
    return judgment


@task(
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
    retry_jitter_factor=0.5,
    log_prints=True,
)
def get_most_recent_last_update_date() -> datetime | None:
    with MongoInterface(
        uri=MONGO_URI,
        db_name=MONGO_DB_NAME,
        collection_name=MONGO_COLLECTION_NAME,
    ) as db:
        latest_doc = db.collection.find_one(sort=[("lastUpdate", -1)])

    if latest_doc and "lastUpdate" in latest_doc:
        last_update = latest_doc["lastUpdate"]
        if not isinstance(last_update, datetime):
            try:
                last_update = datetime.fromisoformat(last_update)
            except (ValueError, TypeError):
                logger.warning(f"Could not parse lastUpdate: {last_update}")
                return None

        return last_update

    logger.info("No previous documents found, using default date")
    return None


@task(
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
    retry_jitter_factor=0.5,
    log_prints=True,
)
def fetch_number_of_judgements(date_from: str) -> int:
    api = PolishCourtAPI()
    params = {
        "lastUpdateFrom": date_from,
    }
    return api.get_number_of_judgements(params=params)


@task(
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
    retry_jitter_factor=0.5,
)
def fetch_judgment_metadata(offset: int, batch_size: int, date_from: str) -> list[dict[str, Any]]:
    api = PolishCourtAPI()
    params = {
        "lastUpdateFrom": date_from,
        "sort": "date-asc",
        "limit": batch_size,
        "offset": offset,
    }
    return api.get_judgements(params=params)


@task(
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
    retry_jitter_factor=0.5,
)
def fetch_judgment_content(judgment_id: str) -> dict[str, Any]:
    api = PolishCourtAPI()
    try:
        return api.get_content(id=judgment_id)
    except DataNotFoundError as e:
        logger.error(f"Error fetching judgment content for {judgment_id}: {e}")
        return {"content": None}


@task(
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
    retry_jitter_factor=0.5,
)
def fetch_judgment_details(judgment_id: str) -> dict[str, Any]:
    api = PolishCourtAPI()
    try:
        return api.get_cleaned_details(id=judgment_id)
    except DataNotFoundError as e:
        logger.error(f"Error fetching judgment details for {judgment_id}: {e}")
        return dict.fromkeys(api.schema["details"], None)


@task(
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=60),
    retry_jitter_factor=1.0,
    retry_condition_fn=RetryOnException(pymongo.errors.ConnectionFailure),
    log_prints=True,
)
def save_or_update_judgments_in_db(judgements: list[dict[str, Any]]) -> None:
    for judgement in judgements:
        try:
            judgement["_id"] = judgement.pop("@id")
        except KeyError:
            judgement["_id"] = judgement.pop("id")

    with MongoInterface(
        uri=MONGO_URI,
        db_name=MONGO_DB_NAME,
        collection_name=MONGO_COLLECTION_NAME,
        batch_size=BATCH_SIZE,
    ) as db:
        db.update_documents(judgements)


@task(
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=60),
    retry_jitter_factor=1.0,
    retry_condition_fn=RetryOnException(pymongo.errors.ConnectionFailure),
    log_prints=True,
)
def dump_dataset(file_name: Path, shard_size: int) -> None:
    query = {"content": {"$ne": None}}
    with MongoInterface(
        uri=MONGO_URI,
        db_name=MONGO_DB_NAME,
        collection_name=MONGO_COLLECTION_NAME,
        batch_size=BATCH_SIZE,
    ) as db:
        db.dump_collection(file_name=file_name, shard_size=shard_size, filter_query=query)


@task
def generate_dataset_card() -> None:
    cmd = [
        "jupyter",
        "nbconvert",
        "--no-input",
        "--to",
        "markdown",
        "--execute",
        "nbs/Dataset Cards/01_Dataset_Description_Raw.ipynb",
        "--output-dir",
        "data/datasets/pl/readme/raw",
        "--output",
        "README",
    ]
    subprocess.run(cmd, check=True)


@task(
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=60),
    retry_condition_fn=RetryOnException(HTTPError),
)
def push_dataset_to_hub() -> None:
    commit_message = f"Dataset update {runtime.flow_run.scheduled_start_time.to_date_string()}"

    num_rows = get_parquet_num_rows(PL_JUDGEMENTS_PATH_RAW)
    assert 100_000 < num_rows < 1_000_000

    card_data = DatasetCardData(
        language="pl",
        multilinguality="monolingual",
        size_categories="100K<n<1M",
        source_datasets=["original"],
        pretty_name="Polish Court Judgments Raw",
        tags=["polish court"],
    )
    card = DatasetCard.from_template(
        card_data,
        template_path=DATASET_CARD_TEMPLATE,
    )

    push_dataset_dir_to_hub(
        dataset_path=PL_JUDGEMENTS_PATH_RAW,
        card=card,
        card_assets=DATASET_CARD_TEMPLATE_ASSETS,
        repo_id=REPO_ID,
        commit_message=commit_message,
    )


if __name__ == "__main__":
    update_pl_court_data.serve(
        name="pl-court-data",
        parameters={"date_from": "2025-02-01"},
        cron="0 18 * * 5",
    )
