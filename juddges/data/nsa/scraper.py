import re

import mechanicalsoup
import typer
from bs4 import BeautifulSoup
from mpire import WorkerPool
from tqdm import trange
from datetime import datetime, timedelta
import pymongo
from loguru import logger

from juddges.utils.tor import TorClient

DB_URI = "mongodb://localhost:27017/"

START_DATE = "1981-01-01"
END_DATE = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")


def main(
    socket_port_start: int = typer.Option(9000),
    config_port_start: int = typer.Option(9900),
    n_jobs: int = typer.Option(1),
) -> None:
    client = pymongo.MongoClient(DB_URI)
    db = client["nsa"]
    dates_col = db["dates"]

    done = []
    for record in dates_col.aggregate(
        [{"$group": {"_id": {"start_date": "$start_date", "end_date": "$end_date"}}}]
    ):
        done.append((record["_id"]["start_date"], record["_id"]["end_date"]))

    logger.info(f"Found {len(done)} done dates in the database.")

    dates = generate_dates(START_DATE, END_DATE)

    start_end_dates = list(reversed(list(zip(dates, dates[1:]))))
    start_end_dates = filter_done_dates(start_end_dates, done)

    worker_torclient_assign = [
        TorClient("", socket_port_start + i, config_port_start + i) for i in range(n_jobs)
    ]
    with WorkerPool(
        n_jobs=n_jobs,
        pass_worker_id=True,
        shared_objects=worker_torclient_assign,
        start_method="threading",
    ) as pool:
        for result in pool.map(
            process_date_range, start_end_dates, progress_bar=True, chunk_size=1
        ):
            dates_col.insert_many(result)


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


def filter_done_dates(dates: list[tuple[str, str]], done: list[tuple[str, str]]):
    done_dates = set(map(tuple, done))
    return [date for date in dates if date not in done_dates]


def process_date_range(
    worker_id: int, worker_torclient_assign: list[TorClient], start_date, end_date
):
    tor_client = worker_torclient_assign[worker_id]
    nsa_scraper = NSAScraper(tor_client)
    documents = nsa_scraper.search_documents(start_date, end_date)
    if documents:
        success = []
        for page_id, document_ids in documents.items():
            page_success = "FOUND" if document_ids is not None else "ERROR: Redirected"
            success.append(
                {
                    "start_date": start_date,
                    "end_date": end_date,
                    "page_id": page_id,
                    "success": page_success,
                    "document_ids": document_ids,
                }
            )
    else:
        success = [
            {
                "start_date": start_date,
                "end_date": end_date,
                "page_id": None,
                "success": "NO_DOCUMENTS",
            }
        ]
    return success


class NSAScraper:
    def __init__(self, tor_client: TorClient | None = None) -> None:
        self.browser = mechanicalsoup.StatefulBrowser(user_agent="MechanicalSoup")
        if tor_client:
            self.browser.session.proxies = tor_client.proxy_config

    def search_documents(self, start_date, end_date):
        response = self.browser.open("https://orzeczenia.nsa.gov.pl/cbo")
        if response.status_code != 200:
            raise Exception(f"Failed to open the website. Status code: {response.status_code}")

        self.browser.select_form()
        # browser["symbole"] = "648"
        self.browser["odDaty"] = start_date
        self.browser["doDaty"] = end_date
        self.browser.submit_selected()
        if self.any_documents_found(self.browser):
            documents = self.retrieve_documents(self.browser)
            num_documents = sum(map(lambda x: len(x) if x else 0, documents.values()))
            print(f"Found {num_documents} documents on {len(documents)} pages.")
            return documents
        else:
            print("No documents found")
            return None

    def any_documents_found(self, browser: mechanicalsoup.StatefulBrowser) -> bool:
        warning_text = "Nie znaleziono orzeczeń spełniających podany warunek!"
        return warning_text not in browser.page.text

    def retrieve_documents(
        self, browser: mechanicalsoup.StatefulBrowser
    ) -> dict[int, list[str] | None]:
        page_links = browser.links(url_regex="^/cbo/find\?p=")
        if not page_links:
            last_page = 1
        else:
            last_page_link = page_links[-2]
            last_page = int(last_page_link.text)

        documents: dict[int, list[str] | None] = {}
        for page_id in trange(1, last_page + 1, disable=last_page == 1):
            browser.open(f"https://orzeczenia.nsa.gov.pl/cbo/find?p={page_id}")

            if browser.url.endswith(f"{page_id}"):
                page_documents = self.find_documents_on_page(browser.page)
                assert (
                    0 < len(page_documents) <= 10
                ), f"Page {page_id} has {len(page_documents)} documents"
                documents[page_id] = page_documents
            else:
                documents[page_id] = None
        return documents

    def find_documents_on_page(self, page: BeautifulSoup) -> list[str]:
        all_links = page.find_all("a", href=True)
        pattern = re.compile(r"^/doc/[A-Za-z0-9]+$")

        filtered_links = []

        for link in all_links:
            href = link["href"]
            if pattern.match(href):
                # Check if the link is within a <span class="powiazane">
                if not link.find_parent("span", class_="powiazane"):
                    filtered_links.append(href)

        return filtered_links


typer.run(main)
