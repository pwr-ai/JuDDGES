import math
import multiprocessing
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import typer
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from juddges.data.database import BatchDatabaseUpdate, BatchedDatabaseCursor, get_mongo_collection
from juddges.settings import PL_COURT_DEP_ID_2_NAME

BATCH_SIZE = 100

load_dotenv()


def main(
    mongo_uri: str = typer.Option(..., envvar="MONGO_URI"),
    court_id_map_file: Path = typer.Option(
        PL_COURT_DEP_ID_2_NAME,
        help="Path to the court_id_map files",
    ),
    batch_size: int = typer.Option(BATCH_SIZE, help="Batch size for fetching documents"),
    n_jobs: int = typer.Option(1, help="Number of parallel jobs"),
    last_update_from: Optional[str] = typer.Option(None, help="Format: YYYY-MM-DD"),
) -> None:
    # fail early if file is invalid:
    court_dep_id_mapper = CourtId2CourtName(court_id_map_file)

    query: dict[str, Any] = {
        "$or": [{"court_name": {"$exists": False}}, {"department_name": {"$exists": False}}]
    }

    if last_update_from is not None:
        # get all rows which were last updated after the given date
        query = {
            "$and": [
                query,
                {"lastUpdate": {"$gte": last_update_from}},
            ]
        }

    collection = get_mongo_collection()
    num_docs_to_update = collection.count_documents(query)
    cursor = collection.find(
        query,
        {"_id": 1, "courtId": 1, "departmentId": 1},
        batch_size=batch_size,
    )
    batched_cursor = BatchedDatabaseCursor(cursor=cursor, batch_size=batch_size, prefetch=True)
    map_and_update_db = BatchDatabaseUpdate(mongo_uri=mongo_uri, update_func=court_dep_id_mapper)

    with multiprocessing.Pool(n_jobs) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    map_and_update_db,
                    batched_cursor,
                ),
                total=math.ceil(num_docs_to_update / batch_size),
            )
        )

    assert collection.count_documents(query) == 0


class CourtId2CourtName:
    def __init__(self, mapping_file: Path) -> None:
        self.mapping: dict[tuple[int, int], dict[str, str]] = (
            pd.read_csv(mapping_file).set_index(["court_id", "dep_id"]).to_dict("index")
        )

    def __call__(self, doc: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.mapping[(int(doc["courtId"]), int(doc["departmentId"]))]
        except KeyError:
            logger.warning(
                "Missing mapping for (courtId, departmentID): "
                f"{doc['courtId'], doc['departmentId']}"
            )
            return {"court_name": None, "department_name": None}


if __name__ == "__main__":
    typer.run(main)
