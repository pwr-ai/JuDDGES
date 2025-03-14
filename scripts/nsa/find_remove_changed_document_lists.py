from pathlib import Path
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pymongo
from pymongo.collection import Collection
import typer
import urllib3
from loguru import logger
from mpire import WorkerPool
from random_user_agent.user_agent import UserAgent

from juddges.data.nsa.scraper import NSAScraper
from juddges.data.nsa.utils import generate_dates
from juddges.utils.logging import setup_loguru

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

START_DATE = "1981-01-01"
END_DATE = (datetime.now(ZoneInfo("Europe/Warsaw")) - timedelta(days=14)).strftime("%Y-%m-%d")


def main(
    n_jobs: int = typer.Option(30),
    proxy_address: str = typer.Option(...),
    db_uri: str = typer.Option(..., envvar="DB_URI"),
    start_date: str = typer.Option(START_DATE, help="Start date for scraping (YYYY-MM-DD)."),
    end_date: str = typer.Option(
        END_DATE, help="End date for scraping (YYYY-MM-DD). Defaults to two weeks ago in Poland."
    ),
    min_interval_between_checks: int = typer.Option(
        30, help="Minimum interval between checks in days."
    ),
    num_elements_to_check: int = typer.Option(3, help="Number of elements to check for each date."),
    max_checks_per_date: int = typer.Option(
        3, help="Maximum number of checks per date if the number of documents is the same."
    ),
    log_file: Path = typer.Option(None, help="Log file to save the logs to."),
) -> None:
    setup_loguru(extra={"script": __file__}, log_file=log_file)
    logger.info("Running find_changes_in_document_list.py with args:\n" + str(locals()))

    client = pymongo.MongoClient(db_uri)
    db = client["nsa"]
    dates_col = db["dates"]
    dates_num_docs = db["dates_num_docs"]

    skip_check = get_dates_to_skip(
        dates_num_docs, min_interval_between_checks, max_checks_per_date, num_elements_to_check
    )

    dates = generate_dates(start_date, end_date)
    dates = [date for date in dates if date not in skip_check]
    logger.info(f"Checking {len(dates)} dates. Skipping {len(skip_check)} dates.")

    random.shuffle(dates)

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
                result["error"]["check_date"] = datetime.now()
                error += 1
            elif "success" in result:
                r = result["success"]
                r["check_date"] = datetime.now().strftime("%Y-%m-%d")
                dates_num_docs.insert_one(r)
                success += 1
            else:
                raise ValueError(f"Invalid result: {result}")
            logger.info(f"Success: {success}, Error: {error}")

    logger.info("Finished checking dates")

    dates_to_remove = get_dates_to_remove(dates_num_docs, num_elements_to_check)
    logger.info(f"Removing {len(dates_to_remove)} dates.")

    if dates_to_remove:
        logger.info(f"Removing {len(dates_to_remove)} dates from the database.")
        result = dates_col.delete_many({"date": {"$in": list(dates_to_remove)}})
        logger.info(f"Deleted {result.deleted_count} documents from dates collection.")
    else:
        logger.info("No dates to remove from the database.")


def get_dates_to_skip(
    dates_num_docs: Collection,
    min_interval_between_checks: int,
    max_checks_per_date: int,
    num_elements_to_check: int,
) -> set[str]:
    num_docs = get_num_docs_aggregated(dates_num_docs)
    mask_len = num_docs["num_docs"].apply(len) >= max_checks_per_date
    mask_same_tail = num_docs["num_docs"].apply(lambda x: len(set(x[-num_elements_to_check:])) == 1)
    mask_time = num_docs["check_date"].apply(
        lambda x: x[-1] > datetime.now() - timedelta(days=min_interval_between_checks)
    )
    skip_check = set(num_docs[mask_len & mask_same_tail].date) | set(num_docs[mask_time].date)
    return skip_check


def get_dates_to_remove(dates_num_docs: Collection, num_elements_to_check: int) -> set[str]:
    num_docs = get_num_docs_aggregated(dates_num_docs)
    mask_tail_not_same = num_docs["num_docs"].apply(
        lambda x: len(set(x[-num_elements_to_check:])) != 1
    )
    dates_to_remove = set(num_docs[mask_tail_not_same].date)
    return dates_to_remove


def get_num_docs_aggregated(dates_num_docs: Collection) -> pd.DataFrame:
    num_docs = pd.DataFrame(dates_num_docs.find())
    num_docs["check_date"] = pd.to_datetime(num_docs["check_date"])
    num_docs = num_docs.sort_values("check_date", ascending=True)
    num_docs = num_docs.groupby("date").agg({"num_docs": list, "check_date": list})
    num_docs = num_docs.reset_index()
    return num_docs


def process_date(proxy_address: str, date: str) -> dict[str, list[dict[str, str]] | dict[str, str]]:
    proxy = {"http": proxy_address, "https": proxy_address}
    nsa_scraper = NSAScraper(
        user_agent=UserAgent(limit=1000).get_random_user_agent().encode("utf-8").decode("utf-8"),
        proxy_config=proxy,
    )
    try:
        num_docs = nsa_scraper.get_num_docs_for_date(date)
        return {"success": {"date": date, "num_docs": num_docs}}
    except Exception as e:
        error_message = f"Failed to scrape documents for date {date}: {e}"
        logger.error(f"Failed to scrape documents for date {date}: {e}; Error type: {type(e)}")
        return {"error": {"date": date, "error": error_message, "error_type": type(e).__name__}}


if __name__ == "__main__":
    typer.run(main)
