"""Logic for tracking judgments published without reasons ("Uzasadnienie") and re-checking them."""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from juddges.data.nsa.extractor import has_reasons
from juddges.data.nsa.scraper import NSAScraper

SEARCH_PAGE_SIZE = 10  # documents per search result page
SEARCH_OVERHEAD = 2  # requests spent on opening the form / counting
DEFAULT_SCHEDULE = (30, 60, 90, 180, 365)


@dataclass
class DatePlan:
    date: str
    doc_ids: list[str]
    strategy: str  # "direct" or "search"
    est_requests: int


@dataclass
class CheckResult:
    strategy: str
    pages: list[dict] = field(default_factory=list)
    n_checked: int = 0
    n_errors: int = 0
    n_new_reasons: int = 0


def parse_schedule(schedule: str) -> list[int]:
    days = [int(x) for x in schedule.split(",") if x.strip()]
    if not days or any(d <= 0 for d in days):
        raise ValueError(f"Invalid schedule: {schedule!r}")
    return days


def search_cost(n_with_reasons: int) -> int:
    return math.ceil(n_with_reasons / SEARCH_PAGE_SIZE) + SEARCH_OVERHEAD


def choose_strategy(n_flagged: int, n_with_reasons: int) -> str:
    """Search for judgments with reasons only if it costs fewer requests than fetching all."""
    return "search" if search_cost(n_with_reasons) < n_flagged else "direct"


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
        strategy = choose_strategy(n, int(docs_per_date[date]))
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
    docs: pd.DataFrame, plan: DatePlan, scraper: NSAScraper, now: datetime
) -> CheckResult:
    """Re-check flagged documents of one date; updates `docs` in place, returns new pages."""
    ts = pd.Timestamp(now)
    n = len(plan.doc_ids)
    to_download = plan.doc_ids
    checked: set[str] = set()
    strategy = "direct"

    if n > SEARCH_OVERHEAD:
        n_with_reasons = scraper.get_num_docs_for_date(plan.date, with_reasons=True)
        strategy = choose_strategy(n, n_with_reasons)
        if strategy == "search":
            found = (
                flatten_search_result(scraper.search_documents_for_date(plan.date, True))
                if n_with_reasons
                else set()
            )
            to_download = [d for d in plan.doc_ids if d in found]
            checked = set(plan.doc_ids) - set(to_download)  # not listed => still no reasons

    result = CheckResult(strategy=strategy)
    for doc_id in to_download:
        try:
            page = scraper.get_page_for_doc(doc_id)
        except Exception as e:
            logger.error(f"Failed to download {doc_id}: {e}; Error type: {type(e)}")
            result.n_errors += 1
            continue
        checked.add(doc_id)
        found_reasons = has_reasons(page)
        result.pages.append({"doc_id": doc_id, "page": page, "downloaded_at": ts})
        docs.loc[doc_id, "page_downloaded_at"] = ts
        docs.loc[doc_id, "has_reasons"] = found_reasons
        if found_reasons:
            docs.loc[doc_id, "reasons_found_at"] = ts
            result.n_new_reasons += 1

    for doc_id in checked:
        docs.loc[doc_id, "last_reasons_check_at"] = ts
        docs.loc[doc_id, "reasons_check_count"] += 1
    result.n_checked = len(checked)
    return result
