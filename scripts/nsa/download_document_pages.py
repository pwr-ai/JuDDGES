from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
import urllib3
from loguru import logger

from juddges.data.nsa.extractor import has_reasons
from juddges.data.nsa.scraper import NSAScraper
from juddges.data.nsa.state import NSAState
from juddges.utils.logging import setup_loguru

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"


def main(
    limit: int = typer.Option(None, help="Download at most this many pages."),
    flush_every: int = typer.Option(50, help="Save pages and state every N downloaded pages."),
    wait_min: float = typer.Option(5.0, help="Min seconds to wait after each request."),
    wait_max: float = typer.Option(10.0, help="Max seconds to wait after each request."),
    proxy_address: str = typer.Option(None, help="Optional proxy: http://user:password@host:port"),
    max_consecutive_errors: int = typer.Option(5, help="Stop after this many errors in a row."),
    state_dir: Path = typer.Option(None, help="State directory (default: NSA_DATA_PATH/state)."),
    log_file: Path = typer.Option(None, help="Log file to save the logs to."),
) -> None:
    setup_loguru(extra={"script": __file__}, log_file=log_file)
    logger.info("Running download_document_pages.py with args:\n" + str(locals()))

    state = NSAState(state_dir)
    docs = state.load_documents()
    todo = docs[docs["page_downloaded_at"].isna()].sort_values("judgment_date", ascending=False)
    to_download = todo["doc_id"].tolist()[:limit]
    logger.info(f"{len(to_download)} pages to download ({len(docs)} documents known)")

    proxy = {"http": proxy_address, "https": proxy_address} if proxy_address else None
    scraper = NSAScraper(USER_AGENT, proxy_config=proxy, wait_range=(wait_min, wait_max))
    buffer: list[dict] = []
    downloaded = 0
    consecutive_errors = 0

    def flush() -> None:
        state.append_pages(buffer)
        state.save_documents(docs)
        buffer.clear()

    for doc_id in to_download:
        try:
            page = scraper.get_page_for_doc(doc_id)
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Failed to download {doc_id}: {e}; Error type: {type(e)}")
            if consecutive_errors >= max_consecutive_errors:
                logger.error(f"{consecutive_errors} consecutive errors, stopping")
                break
            continue
        consecutive_errors = 0
        now = pd.Timestamp(datetime.now())
        found = has_reasons(page)
        docs.loc[doc_id, "page_downloaded_at"] = now
        docs.loc[doc_id, "has_reasons"] = found
        if found:
            docs.loc[doc_id, "reasons_found_at"] = now
        buffer.append({"doc_id": doc_id, "page": page, "downloaded_at": now})
        downloaded += 1
        if len(buffer) >= flush_every:
            flush()
            logger.info(f"Downloaded {downloaded}/{len(to_download)}")
    flush()
    scraper.close()
    logger.info(f"Finished, downloaded {downloaded} pages")


if __name__ == "__main__":
    typer.run(main)
