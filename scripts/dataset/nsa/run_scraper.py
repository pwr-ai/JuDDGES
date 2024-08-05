from datetime import datetime, timedelta

import pymongo
import typer
from datasets import tqdm
from loguru import logger

from juddges.data.nsa.scraper import NSAScraper

DB_URI = "mongodb://localhost:27017/"

START_DATE = "1981-01-01"
END_DATE = (datetime.now()).strftime("%Y-%m-%d")


def main(mongo_db_uri: str = typer.Option(DB_URI)) -> None:
    client = pymongo.MongoClient(mongo_db_uri)
    db = client["nsa"]
    dates_col = db["dates"]

    done = []
    for record in dates_col.aggregate([{"$group": {"_id": {"date": "$date"}}}]):
        done.append(record["_id"]["date"])
    logger.info(f"Found {len(done)} done dates in the database.")

    dates = generate_dates(START_DATE, END_DATE)

    dates = list(reversed(dates))
    dates = filter_done_dates(dates, done)

    result_log = ""
    with tqdm(total=len(dates), desc="Searching for documents") as pbar:
        for date in dates:
            pbar.set_postfix({"Current date": date, "Last date": result_log})
            nsa_scraper = NSAScraper()
            documents = nsa_scraper.search_documents(date, date)
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
                num_documents = sum(map(lambda x: len(x) if x else 0, documents.values()))
                result_log = f"Found {num_documents} documents on {len(documents)} pages."

            else:
                success = [
                    {
                        "date": date,
                        "page_id": None,
                        "success": "NO_DOCUMENTS",
                    }
                ]
                result_log = "No documents found."
            dates_col.insert_many(success)
            pbar.update()
            logger.info(pbar)


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


def filter_done_dates(dates: list[str], done: list[str]):
    done_dates = set(done)
    return [date for date in dates if date not in done_dates]


typer.run(main)
