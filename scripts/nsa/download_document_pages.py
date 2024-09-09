import random
from datetime import datetime

import pandas as pd
import pymongo
import typer
import urllib3
from loguru import logger
from mpire import WorkerPool
from random_user_agent.user_agent import UserAgent

from juddges.data.nsa.scraper import NSAScraper
from juddges.settings import NSA_DATA_PATH

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DB_URI = "mongodb://localhost:27017/"


def main(
    n_jobs: int = typer.Option(50),
    proxy_address: str = typer.Option(...),
    db_uri: str = typer.Option(DB_URI),
) -> None:
    client = pymongo.MongoClient(db_uri)
    db = client["nsa"]
    docs_col = db["document_pages"]
    errors_col = db["document_pages_errors"]

    done = docs_col.find().distinct("doc_id")
    logger.info(f"Found {len(done)} done dates in the database.")

    docs_ids_to_download = get_docs_ids_to_download()

    random.shuffle(docs_ids_to_download)
    dates = filter_done(docs_ids_to_download, done)

    user_agents = UserAgent(limit=1000).get_user_agents()
    user_agents = [ua["user_agent"].encode("utf-8").decode("utf-8") for ua in user_agents]

    buffer = []
    success = 0
    pushed_to_db = 0
    error = 0
    with WorkerPool(n_jobs=n_jobs, shared_objects=(proxy_address, user_agents)) as pool:
        for result in pool.imap_unordered(
            process_doc_id,
            dates,
            progress_bar=True,
            progress_bar_options={"smoothing": 0},
            chunk_size=5,
        ):
            assert len(result) == 1
            if "error" in result:
                result["error"]["time_added"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                errors_col.insert_one(result["error"])
                error += 1
            elif "success" in result:
                result["success"]["time_added"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                buffer.append(result["success"])
                success += 1
                if len(buffer) >= 100:
                    docs_col.insert_many(buffer)
                    pushed_to_db += len(buffer)
                    buffer = []
            else:
                raise ValueError(f"Invalid result: {result}")
            logger.info(f"Success: {success}, Pushed to DB: {pushed_to_db}, Error: {error}")

    if buffer:
        docs_col.insert_many(buffer)
        pushed_to_db += len(buffer)
        logger.info(f"Success: {success}, Pushed to DB: {pushed_to_db}, Error: {error}")

    logger.info("Finished scraping pages")
    logger.info("Saving to file")
    NSA_DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_path = NSA_DATA_PATH / "pages.json"
    data = pd.DataFrame(docs_col.find().sort("date"))
    data["_id"] = data["_id"].astype(str)
    data.to_json(output_path, orient="records", indent=4)


def get_docs_ids_to_download() -> list[str]:
    df = pd.read_json(NSA_DATA_PATH / "documents.json")
    df = df.explode("document_ids")
    df = df[~df["document_ids"].isna()]
    duplicated = df[df.duplicated(subset="document_ids", keep=False)].sort_values("document_ids")
    assert (
        len(duplicated) == 0
    ), "Found duplicated document_ids. Please run drop_dates_with_duplicated_documents.py and scrap_documents_list.py"
    return df["document_ids"].tolist()


def filter_done(document_ids: list[str], done: list[str]) -> list[str]:
    done_docs = set(done)
    return [document_id for document_id in document_ids if document_id not in done_docs]


def process_doc_id(
    shared_objects: tuple[str, list[str]], doc_id: str
) -> dict[str, list[dict[str, str]] | dict[str, str]]:
    proxy_address, user_agents = shared_objects
    proxy = {"http": proxy_address, "https": proxy_address}
    nsa_scraper = NSAScraper(
        user_agent=random.choice(user_agents),
        proxy_config=proxy,
    )
    try:
        page = nsa_scraper.get_page_for_doc(doc_id)
    except Exception as e:
        error_message = f"Failed to scrape page for doc {doc_id}: {e}"
        logger.error(f"{error_message}; Error type: {type(e)}")
        return {"error": {"doc_id": doc_id, "error": error_message, "error_type": type(e).__name__}}

    success = {
        "doc_id": doc_id,
        "page": page,
    }
    return {"success": success}


if __name__ == "__main__":
    typer.run(main)
