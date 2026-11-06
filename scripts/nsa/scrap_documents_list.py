from datetime import datetime, timedelta
from pathlib import Path

import typer
import urllib3
from loguru import logger

from juddges.data.nsa.reasons import flatten_search_result
from juddges.data.nsa.scraper import NSAScraper
from juddges.data.nsa.state import NSAState, register_documents
from juddges.settings import get_nsa_end_date
from juddges.utils.logging import setup_loguru

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

START_DATE = "1981-01-01"
END_DATE = get_nsa_end_date()
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
DATE_FORMAT = "%Y-%m-%d"


def main(
    start_date: str = typer.Option(START_DATE, help="Start date (YYYY-MM-DD)."),
    end_date: str = typer.Option(
        END_DATE, help="End date (YYYY-MM-DD). Defaults to two weeks ago in Poland."
    ),
    refresh_days_back: int = typer.Option(
        30, help="Re-scrape dates not older than this many days (new judgments may be added)."
    ),
    wait_min: float = typer.Option(5.0, help="Min seconds to wait after each request."),
    wait_max: float = typer.Option(10.0, help="Max seconds to wait after each request."),
    proxy_address: str = typer.Option(None, help="Optional proxy: http://user:password@host:port"),
    max_consecutive_errors: int = typer.Option(5, help="Stop after this many errors in a row."),
    state_dir: Path = typer.Option(None, help="State directory (default: NSA_DATA_PATH/state)."),
    log_file: Path = typer.Option(None, help="Log file to save the logs to."),
) -> None:
    setup_loguru(extra={"script": __file__}, log_file=log_file)
    logger.info("Running scrap_documents_list.py with args:\n" + str(locals()))

    state = NSAState(state_dir)
    done = set(state.load_document_lists())
    docs = state.load_documents()
    refresh_from = (datetime.now() - timedelta(days=refresh_days_back)).strftime(DATE_FORMAT)
    dates = [
        d
        for d in generate_dates(start_date, end_date)
        if d not in done or d >= refresh_from  # newest first
    ]
    logger.info(f"{len(done)} dates already scraped, {len(dates)} dates to scrape")

    proxy = {"http": proxy_address, "https": proxy_address} if proxy_address else None
    scraper = NSAScraper(USER_AGENT, proxy_config=proxy, wait_range=(wait_min, wait_max))
    consecutive_errors = 0
    for date in dates:
        try:
            document_ids = sorted(flatten_search_result(scraper.search_documents_for_date(date)))
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Failed to scrape date {date}: {e}; Error type: {type(e)}")
            if consecutive_errors >= max_consecutive_errors:
                logger.error(f"{consecutive_errors} consecutive errors, stopping")
                break
            continue
        consecutive_errors = 0
        now = datetime.now()
        state.append_document_list(date, document_ids, now)
        docs = register_documents(docs, date, document_ids, now)
        state.save_documents(docs)
        logger.info(f"{date}: {len(document_ids)} documents; {len(docs)} known in total")
    scraper.close()
    logger.info("Finished scraping document lists")


def generate_dates(start_date: str, end_date: str) -> list[str]:
    """All dates from start to end (inclusive), newest first."""
    start = datetime.strptime(start_date, DATE_FORMAT)
    end = datetime.strptime(end_date, DATE_FORMAT)
    return [(end - timedelta(days=i)).strftime(DATE_FORMAT) for i in range((end - start).days + 1)]


if __name__ == "__main__":
    typer.run(main)
