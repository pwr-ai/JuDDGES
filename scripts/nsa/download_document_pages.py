import random
from datetime import datetime
from pathlib import Path
import pandas as pd
import pymongo
import typer
import urllib3
from loguru import logger
from mpire import WorkerPool
from pymongo.collection import Collection
from random_user_agent.user_agent import UserAgent
from tqdm import tqdm

from juddges.data.nsa.scraper import NSAScraper
from juddges.settings import NSA_DATA_PATH
from juddges.utils.logging import setup_loguru

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

RETRY_COUNT = 2


def main(
    n_jobs: int = typer.Option(25),
    proxy_address: str = typer.Option(...),
    db_uri: str = typer.Option(..., envvar="DB_URI"),
    log_file: Path = typer.Option(None, help="Log file to save the logs to."),
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_loguru(extra={"script": __file__}, log_file=log_file)
    logger.info("Running download_document_pages.py with args:\n" + str(locals()))

    client = pymongo.MongoClient(db_uri)
    db = client["nsa"]
    docs_col = db["document_pages"]
    errors_col = db["document_pages_errors"]

    # Due to potential errors do it twice
    for _ in range(RETRY_COUNT):
        to_download = get_filtered_docs_is_to_download(docs_col)
        logger.info(f"Downloading {len(to_download)} pages")
        download_pages(docs_col, errors_col, n_jobs, proxy_address, to_download)


def download_pages(
    docs_col: Collection,
    errors_col: Collection,
    n_jobs: int,
    proxy_address: str,
    to_download: list[str],
) -> None:
    user_agents = random.choices(UserAgent(limit=100_000).get_user_agents(), k=1000)
    user_agents = [ua["user_agent"].encode("utf-8").decode("utf-8") for ua in user_agents]
    buffer = []
    success = 0
    pushed_to_db = 0
    error = 0
    with WorkerPool(n_jobs=n_jobs, shared_objects=(proxy_address, user_agents)) as pool:
        for result in pool.imap_unordered(
            process_doc_id,
            to_download,
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


def get_filtered_docs_is_to_download(docs_col: Collection) -> list[str]:
    docs_ids_to_download = get_docs_ids_to_download()
    random.shuffle(docs_ids_to_download)
    to_download = filter_done(docs_ids_to_download, docs_col)
    return to_download


def get_docs_ids_to_download() -> list[str]:
    df = pd.read_json(NSA_DATA_PATH / "documents.json")
    df = df.explode("document_ids")
    df = df[~df["document_ids"].isna()]
    duplicated = df[df.duplicated(subset="document_ids", keep=False)].sort_values("document_ids")
    assert (
        len(duplicated) == 0
    ), "Found duplicated document_ids. Please run drop_dates_with_duplicated_documents.py and scrap_documents_list.py"
    return df["document_ids"].tolist()


def filter_done(document_ids: list[str], docs_col: Collection) -> list[str]:
    done = [
        x["doc_id"]
        for x in tqdm(docs_col.find({}, {"_id": 0, "doc_id": 1}), desc="Loading done pages")
    ]

    logger.info(f"Found {len(done)} done pages in the database.")
    logger.info(f"Progress: {len(done)}/{len(document_ids)}")
    logger.info(f"Progress (%): {len(done)/len(document_ids):.1%}")

    done_docs = set(done)
    assert len(done) == len(done_docs)
    return [document_id for document_id in document_ids if document_id not in done_docs]


def process_doc_id(
    shared_objects: tuple[str, list[str]], doc_id: str
) -> dict[str, list[dict[str, str]] | dict[str, str]]:
    proxy_address, user_agents = shared_objects
    proxy = {"http": proxy_address, "https": proxy_address}
    nsa_scraper = NSAScraper(
        user_agent=random.choice(user_agents),
        proxy_config=proxy,
        wait=False,
    )
    try:
        page = nsa_scraper.get_page_for_doc(doc_id)
        nsa_scraper.close()
    except Exception as e:
        nsa_scraper.close()
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
