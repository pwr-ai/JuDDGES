import random
from datetime import datetime, timedelta

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

START_DATE = "1981-01-01"
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def main(
    n_jobs: int = typer.Option(30),
    proxy_address: str = typer.Option(...),
    db_uri: str = typer.Option(DB_URI),
    start_date: str = typer.Option(START_DATE),
    end_date: str = typer.Option(END_DATE),
) -> None:
    client = pymongo.MongoClient(db_uri)
    db = client["nsa"]
    dates_col = db["dates"]
    errors_col = db["errors"]

    done = dates_col.find().distinct("date")
    logger.info(f"Found {len(done)} done dates in the database.")

    dates = generate_dates(start_date, end_date)

    random.shuffle(dates)
    dates = filter_done_dates(dates, done)

    success = 0
    error = 0
    with WorkerPool(n_jobs=n_jobs, shared_objects=proxy_address) as pool:
        for result in pool.imap_unordered(
            process_date,
            dates,
            progress_bar=True,
            progress_bar_options={"smoothing": 0},
            chunk_size=1,
        ):
            assert len(result) == 1
            if "error" in result:
                result["error"]["time_added"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                errors_col.insert_one(result["error"])
                error += 1
            elif "success" in result:
                for r in result["success"]:
                    r["time_added"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dates_col.insert_many(result["success"])
                success += 1
            else:
                raise ValueError(f"Invalid result: {result}")
            logger.info(f"Success: {success}, Error: {error}")

    logger.info("Finished scraping documents")
    logger.info("Saving to file")
    NSA_DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_path = NSA_DATA_PATH / "documents.json"
    data = pd.DataFrame(dates_col.find().sort("date"))
    data["_id"] = data["_id"].astype(str)
    data.to_json(output_path, orient="records", indent=4)


def generate_dates(start_date: str, end_date: str) -> list[str]:
    date_format = "%Y-%m-%d"
    start = datetime.strptime(start_date, date_format)
    end = datetime.strptime(end_date, date_format)

    date_list = []
    current_date = start
    while current_date <= end:
        date_list.append(current_date.strftime(date_format))
        current_date += timedelta(days=1)

    return date_list


def filter_done_dates(dates: list[str], done: list[str]) -> list[str]:
    done_dates = set(done)
    return [date for date in dates if date not in done_dates]


def process_date(proxy_address: str, date: str) -> dict[str, list[dict[str, str]] | dict[str, str]]:
    proxy = {"http": proxy_address, "https": proxy_address}
    nsa_scraper = NSAScraper(
        user_agent=UserAgent(limit=1000).get_random_user_agent().encode("utf-8").decode("utf-8"),
        proxy_config=proxy,
    )
    try:
        documents = nsa_scraper.search_documents_for_date(date)
    except Exception as e:
        error_message = f"Failed to scrape documents for date {date}: {e}"
        logger.error(f"Failed to scrape documents for date {date}: {e}; Error type: {type(e)}")
        return {"error": {"date": date, "error": error_message, "error_type": type(e).__name__}}
    if documents:
        success = []
        for page_id, document_ids in documents.items():
            page_success = "FOUND" if document_ids is not None else "ERROR: Redirected"
            success.append(
                {
                    "date": date,
                    "page_id": page_id,
                    "success": page_success,
                    "document_ids": document_ids,
                }
            )
    else:
        success = [
            {
                "date": date,
                "page_id": None,
                "success": "NO_DOCUMENTS",
            }
        ]
    return {"success": success}


if __name__ == "__main__":
    typer.run(main)
