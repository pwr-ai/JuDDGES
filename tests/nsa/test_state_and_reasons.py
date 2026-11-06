from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from juddges.data.nsa.reasons import (
    DatePlan,
    check_date,
    choose_strategy,
    find_due_documents,
    flatten_search_result,
    parse_schedule,
    plan_checks,
    search_cost,
)
from juddges.data.nsa.state import NSAState, empty_documents, register_documents

FIXTURES = Path(__file__).parent / "fixtures"
SENTENCJA = (FIXTURES / "judgment_sentencja_only.html").read_text()
WITH_REASONS = (FIXTURES / "judgment_with_reasons.html").read_text()
NOW = datetime(2026, 6, 1, 12, 0, 0)
SCHEDULE = [30, 60, 90]


def make_docs(rows: list[dict]) -> pd.DataFrame:
    """rows: doc_id, judgment_date, downloaded_days_ago, [checked_days_ago, count]"""
    docs = register_documents(empty_documents(), "x", [], NOW)
    for r in rows:
        docs = register_documents(docs, r["judgment_date"], [r["doc_id"]], NOW)
        docs.loc[r["doc_id"], "page_downloaded_at"] = NOW - timedelta(days=r["downloaded"])
        docs.loc[r["doc_id"], "has_reasons"] = r.get("has_reasons", False)
        if "checked" in r:
            docs.loc[r["doc_id"], "last_reasons_check_at"] = NOW - timedelta(days=r["checked"])
        docs.loc[r["doc_id"], "reasons_check_count"] = r.get("count", 0)
    return docs


class FakeScraper:
    def __init__(self, with_reasons_ids: set[str], fail: set[str] = frozenset()):
        self.with_reasons_ids = with_reasons_ids
        self.fail = fail
        self.requests = []

    def get_num_docs_for_date(self, date, with_reasons=None):
        self.requests.append("count")
        return len(self.with_reasons_ids)

    def search_documents_for_date(self, date, with_reasons=None):
        self.requests.append("search")
        return {1: sorted(self.with_reasons_ids)} if self.with_reasons_ids else None

    def get_page_for_doc(self, doc_id):
        self.requests.append(f"page:{doc_id}")
        if doc_id in self.fail:
            raise RuntimeError("boom")
        return WITH_REASONS if doc_id in self.with_reasons_ids else SENTENCJA


def test_state_roundtrip(tmp_path):
    state = NSAState(tmp_path / "state")
    assert state.load_documents().empty
    assert state.load_document_lists() == {}

    state.append_document_list("2026-01-01", ["a"], NOW)
    state.append_document_list("2026-01-01", ["a", "b"], NOW)
    assert state.load_document_lists()["2026-01-01"]["document_ids"] == ["a", "b"]

    docs = register_documents(state.load_documents(), "2026-01-01", ["a", "b"], NOW)
    docs = register_documents(docs, "2026-01-01", ["b", "c"], NOW)
    assert list(docs["doc_id"]) == ["a", "b", "c"]
    state.save_documents(docs)
    loaded = state.load_documents()
    assert list(loaded.columns) == list(empty_documents().columns)
    assert loaded.loc["a", "has_reasons"] is pd.NA
    assert loaded.loc["c", "reasons_check_count"] == 0
    assert not list(tmp_path.rglob("*.tmp"))


def test_latest_page_wins(tmp_path):
    state = NSAState(tmp_path)
    assert list(state.iter_latest_pages()) == []
    t1, t2 = pd.Timestamp(NOW), pd.Timestamp(NOW + timedelta(days=1))
    state.append_pages(
        [
            {"doc_id": "a", "page": "old", "downloaded_at": t1},
            {"doc_id": "b", "page": "b1", "downloaded_at": t1},
        ]
    )
    state.append_pages([{"doc_id": "a", "page": "new", "downloaded_at": t2}])
    latest = pd.concat(state.iter_latest_pages()).set_index("doc_id")["page"].to_dict()
    assert latest == {"a": "new", "b": "b1"}


def test_parse_schedule():
    assert parse_schedule("30, 60,90") == [30, 60, 90]
    with pytest.raises(ValueError):
        parse_schedule("")


def test_strategy_choice():
    assert search_cost(0) == 2 and search_cost(10) == 3 and search_cost(11) == 4
    assert choose_strategy(n_flagged=1, n_with_reasons=5) == "direct"
    assert choose_strategy(n_flagged=30, n_with_reasons=10) == "search"
    assert choose_strategy(n_flagged=3, n_with_reasons=3) == "direct"  # cost 3, not cheaper
    assert choose_strategy(n_flagged=4, n_with_reasons=3) == "search"


def test_due_documents_respect_schedule_and_age():
    docs = make_docs(
        [
            {"doc_id": "never_30", "judgment_date": "2026-04-01", "downloaded": 31},
            {"doc_id": "never_10", "judgment_date": "2026-04-01", "downloaded": 10},
            {
                "doc_id": "chk1_due",
                "judgment_date": "2026-03-01",
                "downloaded": 200,
                "checked": 61,
                "count": 1,
            },
            {
                "doc_id": "chk1_not",
                "judgment_date": "2026-03-01",
                "downloaded": 200,
                "checked": 59,
                "count": 1,
            },
            {
                "doc_id": "past_end",
                "judgment_date": "2026-03-01",
                "downloaded": 300,
                "checked": 91,
                "count": 7,
            },
            {"doc_id": "too_old", "judgment_date": "2020-01-01", "downloaded": 900},
            {"doc_id": "has", "judgment_date": "2026-04-01", "downloaded": 90, "has_reasons": True},
        ]
    )
    docs = register_documents(docs, "2026-04-01", ["no_page"], NOW)  # has_reasons is NA
    due = find_due_documents(docs, NOW, max_age_days=730, schedule=SCHEDULE)
    assert sorted(due["doc_id"]) == ["chk1_due", "never_30", "past_end"]


def test_plan_checks_groups_by_date_and_picks_strategy():
    rows = [{"doc_id": f"d{i}", "judgment_date": "2026-04-01", "downloaded": 40} for i in range(30)]
    rows.append({"doc_id": "solo", "judgment_date": "2026-03-01", "downloaded": 40})
    plans = plan_checks(make_docs(rows), NOW, 730, SCHEDULE)
    assert [p.date for p in plans] == ["2026-04-01", "2026-03-01"]  # newest first
    assert (plans[0].strategy, plans[0].est_requests) == ("search", 5)  # ceil(30/10)+2
    assert (plans[1].strategy, plans[1].est_requests) == ("direct", 1)


def test_check_date_search_strategy_downloads_only_flagged_with_reasons():
    ids = [f"d{i}" for i in range(20)]
    docs = make_docs([{"doc_id": i, "judgment_date": "2026-04-01", "downloaded": 40} for i in ids])
    scraper = FakeScraper({"d3", "d7"})
    plan = DatePlan("2026-04-01", ids, "search", 4)
    result = check_date(docs, plan, scraper, NOW)

    assert result.strategy == "search"
    assert scraper.requests == ["count", "search", "page:d3", "page:d7"]
    assert result.n_new_reasons == 2 and result.n_checked == 20
    assert [p["doc_id"] for p in result.pages] == ["d3", "d7"]
    assert docs.loc["d3", "has_reasons"] and docs.loc["d3", "reasons_found_at"] == NOW
    assert not docs.loc["d0", "has_reasons"]
    assert (docs["reasons_check_count"] == 1).all()
    assert (docs["last_reasons_check_at"] == NOW).all()
    assert docs.loc["d3", "page_downloaded_at"] == NOW
    assert docs.loc["d0", "page_downloaded_at"] == NOW - timedelta(days=40)


def test_check_date_search_with_no_reasons_skips_search():
    ids = [f"d{i}" for i in range(20)]
    docs = make_docs([{"doc_id": i, "judgment_date": "2026-04-01", "downloaded": 40} for i in ids])
    scraper = FakeScraper(set())
    result = check_date(docs, DatePlan("2026-04-01", ids, "search", 2), scraper, NOW)
    assert scraper.requests == ["count"]
    assert result.n_checked == 20 and result.pages == []


def test_check_date_direct_strategy_and_errors():
    docs = make_docs(
        [{"doc_id": i, "judgment_date": "2026-04-01", "downloaded": 40} for i in ("a", "b")]
    )
    scraper = FakeScraper({"a"}, fail={"b"})
    result = check_date(docs, DatePlan("2026-04-01", ["a", "b"], "direct", 2), scraper, NOW)
    assert scraper.requests == ["page:a", "page:b"]  # no count request for tiny groups
    assert result.n_errors == 1 and result.n_checked == 1
    assert docs.loc["a", "has_reasons"] and docs.loc["a", "reasons_check_count"] == 1
    assert docs.loc["b", "reasons_check_count"] == 0  # failed => retried next time
    assert pd.isna(docs.loc["b", "last_reasons_check_at"])


def test_flatten_search_result():
    assert flatten_search_result(None) == set()
    assert flatten_search_result({1: ["a"], 2: ["b"]}) == {"a", "b"}
    with pytest.raises(ValueError):
        flatten_search_result({1: ["a"], 2: None})
