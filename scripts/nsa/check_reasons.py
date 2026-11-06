from datetime import datetime
from pathlib import Path

import typer
import urllib3
from loguru import logger
from rich.console import Console
from rich.table import Table

from juddges.data.nsa.reasons import (
    DEFAULT_SCHEDULE,
    TooManyErrors,
    check_date,
    parse_schedule,
    plan_checks,
)
from juddges.data.nsa.scraper import NSAScraper
from juddges.data.nsa.state import NSAState, StateLocked
from juddges.utils.logging import setup_loguru

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"


def main(
    max_age_days: int = typer.Option(730, help="Only check judgments not older than this."),
    schedule: str = typer.Option(
        ",".join(map(str, DEFAULT_SCHEDULE)),
        help="Days between consecutive checks (the last interval repeats).",
    ),
    dry_run: bool = typer.Option(False, help="Print the plan and exit without any request."),
    wait_min: float = typer.Option(5.0, help="Min seconds to wait after each request."),
    wait_max: float = typer.Option(10.0, help="Max seconds to wait after each request."),
    proxy_address: str = typer.Option(None, help="Optional proxy: http://user:password@host:port"),
    max_consecutive_errors: int = typer.Option(5, help="Stop after this many errors in a row."),
    state_dir: Path = typer.Option(None, help="State directory (default: NSA_DATA_PATH/state)."),
    log_file: Path = typer.Option(None, help="Log file to save the logs to."),
) -> None:
    setup_loguru(extra={"script": __file__}, log_file=log_file)
    logger.info("Running check_reasons.py with args:\n" + str(locals()))

    state = NSAState(state_dir)
    try:
        state.acquire_lock()
    except StateLocked as e:
        logger.error(str(e))
        raise typer.Exit(1)
    docs = state.load_documents()
    plans = plan_checks(docs, datetime.now(), max_age_days, parse_schedule(schedule))

    table = Table(title="Planned reasons checks (estimates: search cost uses an upper bound)")
    for col in ("Judgment date", "Flagged docs", "Strategy", "Est. requests"):
        table.add_column(col, justify="left" if col in ("Judgment date", "Strategy") else "right")
    for p in plans:
        table.add_row(p.date, str(len(p.doc_ids)), p.strategy, str(p.est_requests))
    Console().print(table)
    Console().print(
        f"{len(plans)} dates, {sum(len(p.doc_ids) for p in plans)} docs, "
        f"~{sum(p.est_requests for p in plans)} requests"
    )
    if dry_run or not plans:
        return

    proxy = {"http": proxy_address, "https": proxy_address} if proxy_address else None
    scraper = NSAScraper(USER_AGENT, proxy_config=proxy, wait_range=(wait_min, wait_max))

    def flush(pages: list[dict]) -> None:
        state.append_pages(pages)  # pages first: has_reasons=True never lacks its page
        state.save_documents(docs)

    consecutive_errors = 0
    for plan in plans:
        try:
            result = check_date(
                docs,
                plan,
                scraper,
                datetime.now(),
                flush,
                max_consecutive_errors=max_consecutive_errors,
            )
        except TooManyErrors as e:
            logger.error(f"{e}, stopping")
            break
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Failed to check date {plan.date}: {e}; Error type: {type(e)}")
            if consecutive_errors >= max_consecutive_errors:
                logger.error(f"{consecutive_errors} consecutive errors, stopping")
                break
            continue
        consecutive_errors = 0
        logger.info(
            f"{plan.date}: {result.strategy}, checked {result.n_checked}/{len(plan.doc_ids)}, "
            f"new reasons {result.n_new_reasons}, errors {result.n_errors}"
        )
    scraper.close()


if __name__ == "__main__":
    typer.run(main)
