"""Logic for tracking judgments published without reasons ("Uzasadnienie") and re-checking them."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from juddges.data.nsa.extractor import has_reasons
from juddges.data.nsa.scraper import NSAScraper

SEARCH_PAGE_SIZE = 10  # documents per search result page
COUNT_COST = 2  # requests of get_num_docs_for_date: open the form + submit it
SEARCH_SUBMIT_COST = 2  # requests of submitting the search again (the first results page is
# then fetched once more as page 1, which is included in the ceil(n / SEARCH_PAGE_SIZE) pages)
DEFAULT_SCHEDULE = (30, 60, 90, 180, 365)
DEFAULT_MAX_CONSECUTIVE_ERRORS = 5


class TooManyErrors(RuntimeError):
    """Too many consecutive failures; the caller should stop the whole run."""


@dataclass
class DatePlan:
    date: str
    doc_ids: list[str]
    strategy: str  # "direct" or "search"
    est_requests: int


@dataclass
class CheckResult:
    strategy: str
    n_checked: int = 0
    n_errors: int = 0
    n_new_reasons: int = 0


def parse_schedule(schedule: str) -> list[int]:
    days = [int(x) for x in schedule.split(",") if x.strip()]
    if not days or any(d <= 0 for d in days):
        raise ValueError(f"Invalid schedule: {schedule!r}")
    return days


def search_cost(n_with_reasons: int, counted: bool = False) -> int:
    """Requests needed to find the judgments with reasons of one date via the search.

    The count request comes first; when there are no such judgments nothing else is needed.
    `counted=True` excludes the count (already spent).
    """
    count = 0 if counted else COUNT_COST
    if n_with_reasons == 0:
        return count
    return count + SEARCH_SUBMIT_COST + math.ceil(n_with_reasons / SEARCH_PAGE_SIZE)


def choose_strategy(n_flagged: int, n_with_reasons: int, counted: bool = True) -> str:
    """Search for judgments with reasons only if it costs fewer requests than fetching all."""
    return "search" if search_cost(n_with_reasons, counted) < n_flagged else "direct"


def find_due_documents(
    docs: pd.DataFrame, now: datetime, max_age_days: int, schedule: list[int]
) -> pd.DataFrame:
    """Documents flagged as sentencja-only, young enough and due for another check."""
    cutoff = (now - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
    candidates = docs[
        (docs["has_reasons"] == False)  # noqa: E712 - NA must not match
        & (docs["judgment_date"] >= cutoff)
        & docs["page_downloaded_at"].notna()
    ]
    reference = candidates["last_reasons_check_at"].fillna(candidates["page_downloaded_at"])
    idx = (candidates["reasons_check_count"].clip(upper=len(schedule) - 1)).astype(int)
    interval = pd.to_timedelta([schedule[i] for i in idx], unit="D")
    return candidates[(now - reference).to_numpy() >= interval.to_numpy()]


def plan_checks(
    docs: pd.DataFrame, now: datetime, max_age_days: int, schedule: list[int]
) -> list[DatePlan]:
    """One plan per due judgment date (newest first), with an estimated request count.

    The number of judgments with reasons is unknown without network, so the whole date is used
    as its upper bound; the real run refines it with a count request.
    """
    due = find_due_documents(docs, now, max_age_days, schedule)
    docs_per_date = docs.groupby("judgment_date").size()
    plans = []
    for date, group in sorted(due.groupby("judgment_date"), reverse=True):
        n = len(group)
        strategy = choose_strategy(n, int(docs_per_date[date]), counted=False)
        est = search_cost(int(docs_per_date[date])) if strategy == "search" else n
        plans.append(DatePlan(date, group["doc_id"].tolist(), strategy, est))
    return plans


def flatten_search_result(result: dict[int, list[str] | None] | None) -> set[str]:
    if not result:
        return set()
    if any(ids is None for ids in result.values()):
        raise ValueError("Incomplete search result (a result page was not retrieved)")
    return {doc_id for ids in result.values() for doc_id in ids}


def check_date(
    docs: pd.DataFrame,
    plan: DatePlan,
    scraper: NSAScraper,
    now: datetime,
    flush: Callable[[list[dict]], None],
    flush_every: int = 50,
    max_consecutive_errors: int = DEFAULT_MAX_CONSECUTIVE_ERRORS,
) -> CheckResult:
    """Re-check flagged documents of one date; updates `docs` in place.

    New pages are passed to `flush` (which must store them first and the `docs` state afterwards)
    every `flush_every` pages and at the end - also when an exception (or Ctrl-C) interrupts the
    date - so `has_reasons=True` is never persisted without its page.

    Raises:
        TooManyErrors: after `max_consecutive_errors` failed downloads in a row.
    """
    ts = pd.Timestamp(now)
    result = CheckResult(strategy="direct")
    to_download = list(plan.doc_ids)
    buffer: list[dict] = []

    def mark_checked(doc_id: str) -> None:
        docs.loc[doc_id, "last_reasons_check_at"] = ts
        docs.loc[doc_id, "reasons_check_count"] += 1
        result.n_checked += 1

    try:
        if len(plan.doc_ids) > COUNT_COST:
            n_with_reasons = scraper.get_num_docs_for_date(plan.date, with_reasons=True)
            result.strategy = choose_strategy(len(plan.doc_ids), n_with_reasons)
            if result.strategy == "search":
                found = (
                    flatten_search_result(
                        scraper.search_documents_for_date(
                            plan.date, True, expected_count=n_with_reasons
                        )
                    )
                    if n_with_reasons
                    else set()
                )
                to_download = []
                for doc_id in plan.doc_ids:
                    # A document listed on several dates may be reported under another date, so
                    # "not in the result" proves nothing: download it.
                    if doc_id in found or len(docs.at[doc_id, "listed_dates"]) > 1:
                        to_download.append(doc_id)
                    else:
                        mark_checked(doc_id)

        consecutive_errors = 0
        for doc_id in to_download:
            try:
                page = scraper.get_page_for_doc(doc_id)
            except Exception as e:
                logger.error(f"Failed to download {doc_id}: {e}; Error type: {type(e)}")
                result.n_errors += 1
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise TooManyErrors(
                        f"{consecutive_errors} consecutive download errors on {plan.date}"
                    ) from e
                continue
            consecutive_errors = 0
            found_reasons = has_reasons(page)
            buffer.append({"doc_id": doc_id, "page": page, "downloaded_at": ts})
            docs.loc[doc_id, "page_downloaded_at"] = ts
            docs.loc[doc_id, "has_reasons"] = found_reasons
            if found_reasons:
                docs.loc[doc_id, "reasons_found_at"] = ts
                result.n_new_reasons += 1
            mark_checked(doc_id)
            if len(buffer) >= flush_every:
                flush(buffer)
                buffer = []
    finally:
        flush(buffer)
    return result
