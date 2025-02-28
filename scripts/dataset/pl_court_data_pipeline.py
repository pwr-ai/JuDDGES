import asyncio
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from dotenv import dotenv_values, load_dotenv
from huggingface_hub import DatasetCard, DatasetCardData, HfApi
from loguru import logger
from prefect import flow, get_client, runtime, task, unmapped
from prefect.client.schemas.filters import FlowRunFilter
from prefect.client.schemas.sorting import FlowRunSort
from prefect.task_runners import ThreadPoolTaskRunner
from tqdm import tqdm, trange

from juddges.data.database import BatchDatabaseUpdate, get_mongo_collection
from juddges.data.pl_court_api import PolishCourtAPI
from juddges.preprocessing.pl_court_parser import SimplePlJudgementsParser
from juddges.settings import PL_JUDGEMENTS_PATH, PL_JUDGEMENTS_PATH_RAW

load_dotenv()

MAX_CONCURRENT_WORKERS = 10
BATCH_SIZE = 50
COURT_ID_2_NAME_FILE = "data/datasets/pl/court_id_2_name.csv"
MONGO_URI = dotenv_values()["MONGO_URI"]
COLLECTION_NAME = "pl-court"

REPO_ID = "JuDDGES/pl-court-raw"
MAX_SHARD_SIZE = "4GB"
CHUNK_SIZE = 25_000
DATASET_CARD_TEMPLATE = PL_JUDGEMENTS_PATH / "readme/raw/README.md"
DATASET_CARD_TEMPLATE_ASSETS = PL_JUDGEMENTS_PATH / "readme/raw/README_files"


@flow(task_runner=ThreadPoolTaskRunner(max_workers=MAX_CONCURRENT_WORKERS), log_prints=True)
def update_pl_court_data(date_from: str | None = None, batch_size: int = BATCH_SIZE) -> None:
    latest_successful_flow_date = _get_recent_successful_flow_date()
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
    database_writer = SaveOrUpdateJudgements(mongo_uri=MONGO_URI)

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
        database_writer(judgements=processed_judgements)

    dump_dataset(
        chunk_size=CHUNK_SIZE,
        file_name=PL_JUDGEMENTS_PATH_RAW / "pl_court_data.parquet",
        start_offset=0,
        num_docs=n_judgements,
    )

    # push_dataset_to_hub()


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
        return self.parser(doc["content"])


@task(name="fetch_judgment_data")
def fetch_judgment_data(
    judgment: dict[str, Any],
    court_department_id_mapper: MapCourtDepartmentIds2Names,
    doc_parser: ParseDoc,
) -> dict[str, Any]:
    judgement = fetch_judgment_content(judgment_id=judgment["id"])
    judgement |= fetch_judgment_details(judgment_id=judgment["id"])
    judgement |= court_department_id_mapper(doc=judgement)
    judgement |= doc_parser(doc=judgement)
    return judgement


@task(retries=3, retry_delay_seconds=10, log_prints=True)
def get_most_recent_last_update_date() -> datetime | None:
    collection = get_mongo_collection(collection_name=COLLECTION_NAME)
    latest_doc = collection.find_one(sort=[("lastUpdate", -1)])

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


@task(retries=3, retry_delay_seconds=10, log_prints=True)
def fetch_number_of_judgements(date_from: str) -> int:
    api = PolishCourtAPI()
    params = {
        "lastUpdateFrom": date_from,
    }
    return api.get_number_of_judgements(params=params)


@task(retries=3, retry_delay_seconds=10)
def fetch_judgment_metadata(offset: int, batch_size: int, date_from: str) -> list[dict[str, Any]]:
    api = PolishCourtAPI()
    params = {
        "lastUpdateFrom": date_from,
        "sort": "date-asc",
        "limit": batch_size,
        "offset": offset,
    }
    return api.get_judgements(params=params)


@task(retries=3, retry_delay_seconds=10)
def fetch_judgment_content(judgment_id: str) -> dict[str, Any]:
    api = PolishCourtAPI()
    return api.get_content(id=judgment_id)


@task(retries=3, retry_delay_seconds=10)
def fetch_judgment_details(judgment_id: str) -> dict[str, Any]:
    api = PolishCourtAPI()
    return api.get_details(id=judgment_id)


class SaveOrUpdateJudgements:
    def __init__(self, mongo_uri: str) -> None:
        self.update_db = BatchDatabaseUpdate(mongo_uri=mongo_uri)

    @task(name="store_judgements", log_prints=True)
    def __call__(self, judgements: list[dict[str, Any]]) -> None:
        for judgement in judgements:
            judgement["_id"] = judgement.pop("@id")
        self.update_db(judgements)


@task
def dump_dataset(chunk_size: int, file_name: Path, start_offset: int, num_docs: int) -> None:
    collection = get_mongo_collection(collection_name=COLLECTION_NAME)
    query = {"content": {"$ne": None}}
    for offset in trange(start_offset, num_docs, chunk_size, desc="Chunks"):
        docs = list(
            tqdm(
                collection.find(query, {"embedding": 0}, batch_size=BATCH_SIZE)
                .skip(offset)
                .limit(chunk_size),
                total=chunk_size,
                leave=False,
                desc="Documents in chunk",
            )
        )
        i = offset // chunk_size
        dumped_f_name = _save_docs(docs, file_name, i)
        logger.info(f"Dumped {i}-th batch of documents to {dumped_f_name}")


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


@task
def push_dataset_to_hub() -> None:
    commit_message = f"Dataset update {runtime.flow_run.start_time.to_date_string()}"
    ds = load_dataset("parquet", name="pl_judgements", data_dir=PL_JUDGEMENTS_PATH_RAW)

    num_rows = ds["train"].num_rows
    logger.info(f"Loaded dataset size: {num_rows}")

    ds.push_to_hub(
        repo_id=REPO_ID,
        max_shard_size=MAX_SHARD_SIZE,
        commit_message=commit_message,
    )

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
    card.push_to_hub(
        repo_id=REPO_ID,
        commit_message=commit_message,
    )

    api = HfApi()
    api.upload_folder(
        folder_path=DATASET_CARD_TEMPLATE_ASSETS,
        path_in_repo=DATASET_CARD_TEMPLATE_ASSETS.name,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=commit_message,
    )


@push_dataset_to_hub.on_completion
def push_dataset_to_hub_completion() -> None:
    shutil.rmtree(PL_JUDGEMENTS_PATH_RAW)


def _save_docs(docs: list[dict[str, Any]], file_name: Path, i: int | None) -> Path:
    if i is not None:
        file_name = file_name.with_name(f"{file_name.stem}_{i:02d}{file_name.suffix}")
    pd.DataFrame(docs).to_parquet(file_name)
    return file_name


def _get_recent_successful_flow_date() -> str | None:
    async def _get_recent_successful_flow_date() -> str | None:
        async with get_client() as client:
            flow_runs = await client.read_flow_runs(
                flow_run_filter=FlowRunFilter(
                    name={"type": ["update_pl_court_data"]},
                    state={"type": {"any_": ["COMPLETED"]}},
                ),
                sort=FlowRunSort.END_TIME_DESC,
                limit=1,
            )
        if len(flow_runs) == 0:
            return None
        return flow_runs[0].start_time.to_date_string()

    return asyncio.run(_get_recent_successful_flow_date())


if __name__ == "__main__":
    update_pl_court_data.serve(
        name="pl-court-data",
        parameters={"date_from": "2025-02-01"},
        cron="0 18 * * 5",
    )
