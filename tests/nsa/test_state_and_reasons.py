import math
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from juddges.data.nsa.reasons import (
    DatePlan,
    TooManyErrors,
    check_date,
    choose_strategy,
    find_due_documents,
    flatten_search_result,
    parse_schedule,
    plan_checks,
    search_cost,
)
from juddges.data.nsa.state import (
    NSAState,
    StateLocked,
    empty_documents,
    register_documents,
)

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
        for extra_date in r.get("also_listed", []):
            docs = register_documents(docs, extra_date, [r["doc_id"]], NOW)
    return docs


class FakeScraper:
    """Counts requests like the real scraper: a count is 2, a search 2 + result pages, a page 1."""

    def __init__(self, with_reasons_ids: set[str], fail: set[str] = frozenset()):
        self.with_reasons_ids = with_reasons_ids
        self.fail = fail
        self.requests = []
        self.n_requests = 0

    def get_num_docs_for_date(self, date, with_reasons=None):
        self.requests.append("count")
        self.n_requests += 2
        return len(self.with_reasons_ids)

    def search_documents_for_date(self, date, with_reasons=None, expected_count=None):
        self.requests.append("search")
        if expected_count is None:
            expected_count = self.get_num_docs_for_date(date, with_reasons)
        if expected_count == 0:
            return None
        self.n_requests += 2 + math.ceil(expected_count / 10)
        return {1: sorted(self.with_reasons_ids)}

    def get_page_for_doc(self, doc_id):
        self.requests.append(f"page:{doc_id}")
        self.n_requests += 1
        if doc_id in self.fail:
            raise RuntimeError("boom")
        return WITH_REASONS if doc_id in self.with_reasons_ids else SENTENCJA


class Sink:
    """Fake flush target: keeps stored pages and a snapshot of the stored docs state."""

    def __init__(self, docs):
        self.docs = docs
        self.pages = []
        self.snapshot = None

    def __call__(self, pages):
        self.pages.extend(pages)
        self.snapshot = self.docs.copy()


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


def test_search_cost_model():
    assert search_cost(0) == 2  # only the count
    assert search_cost(0, counted=True) == 0
    assert search_cost(10) == 2 + 2 + 1
    assert search_cost(11, counted=True) == 2 + 2


def test_strategy_choice():
    assert choose_strategy(n_flagged=1, n_with_reasons=5) == "direct"
    assert choose_strategy(n_flagged=30, n_with_reasons=10) == "search"
    assert choose_strategy(n_flagged=3, n_with_reasons=0) == "search"  # nothing to fetch at all
    assert choose_strategy(n_flagged=4, n_with_reasons=5) == "search"  # marginal 2 + 1 = 3 < 4
    assert choose_strategy(n_flagged=3, n_with_reasons=5) == "direct"
    assert choose_strategy(n_flagged=4, n_with_reasons=5, counted=False) == "direct"


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
    assert (plans[0].strategy, plans[0].est_requests) == ("search", 2 + 2 + 3)  # upper bound: 30
    assert (plans[1].strategy, plans[1].est_requests) == ("direct", 1)


def _docs_for(ids, **kw):
    return make_docs(
        [{"doc_id": i, "judgment_date": "2026-04-01", "downloaded": 40, **kw} for i in ids]
    )


def test_check_date_search_strategy_downloads_only_flagged_with_reasons():
    ids = [f"d{i}" for i in range(20)]
    docs = _docs_for(ids)
    scraper = FakeScraper({"d3", "d7"})
    sink = Sink(docs)
    result = check_date(docs, DatePlan("2026-04-01", ids, "search", 4), scraper, NOW, sink)

    assert result.strategy == "search"
    assert scraper.requests == ["count", "search", "page:d3", "page:d7"]
    assert scraper.n_requests == search_cost(2) + 2  # modeled cost + two page downloads
    assert result.n_new_reasons == 2 and result.n_checked == 20
    assert [p["doc_id"] for p in sink.pages] == ["d3", "d7"]
    assert docs.loc["d3", "has_reasons"] and docs.loc["d3", "reasons_found_at"] == NOW
    assert not docs.loc["d0", "has_reasons"]
    assert (docs["reasons_check_count"] == 1).all()
    assert (docs["last_reasons_check_at"] == NOW).all()
    assert docs.loc["d3", "page_downloaded_at"] == NOW
    assert docs.loc["d0", "page_downloaded_at"] == NOW - timedelta(days=40)


def test_check_date_search_with_no_reasons_skips_search():
    ids = [f"d{i}" for i in range(20)]
    docs = _docs_for(ids)
    scraper = FakeScraper(set())
    sink = Sink(docs)
    result = check_date(docs, DatePlan("2026-04-01", ids, "search", 2), scraper, NOW, sink)
    assert scraper.requests == ["count"]
    assert scraper.n_requests == search_cost(0)
    assert result.n_checked == 20 and sink.pages == []
    assert (sink.snapshot["reasons_check_count"] == 1).all()


def test_search_strategy_downloads_docs_listed_on_several_dates():
    # NSA may report such a doc under the other date => absence from the result proves nothing
    docs = make_docs(
        [{"doc_id": f"d{i}", "judgment_date": "2026-04-01", "downloaded": 40} for i in range(19)]
        + [
            {
                "doc_id": "shifted",
                "judgment_date": "2026-04-01",
                "downloaded": 40,
                "also_listed": ["2026-04-02"],
            }
        ]
    )
    ids = list(docs["doc_id"])
    scraper = FakeScraper({"d3"})
    result = check_date(docs, DatePlan("2026-04-01", ids, "search", 4), scraper, NOW, Sink(docs))
    assert scraper.requests == ["count", "search", "page:d3", "page:shifted"]
    assert result.n_checked == 20


def test_register_records_all_listed_dates():
    docs = register_documents(empty_documents(), "2026-04-01", ["a", "b"], NOW)
    docs = register_documents(docs, "2026-04-02", ["b", "c"], NOW)
    docs = register_documents(docs, "2026-04-02", ["b"], NOW)  # re-scrape: no duplicates
    assert docs.loc["a", "listed_dates"] == ["2026-04-01"]
    assert docs.loc["b", "listed_dates"] == ["2026-04-01", "2026-04-02"]
    assert docs.loc["b", "judgment_date"] == "2026-04-01"  # first listing stays


def test_check_date_direct_strategy_and_errors():
    docs = _docs_for(["a", "b"])
    scraper = FakeScraper({"a"}, fail={"b"})
    result = check_date(
        docs, DatePlan("2026-04-01", ["a", "b"], "direct", 2), scraper, NOW, Sink(docs)
    )
    assert scraper.requests == ["page:a", "page:b"]  # no count request for tiny groups
    assert result.n_errors == 1 and result.n_checked == 1
    assert docs.loc["a", "has_reasons"] and docs.loc["a", "reasons_check_count"] == 1
    assert docs.loc["b", "reasons_check_count"] == 0  # failed => retried next time
    assert pd.isna(docs.loc["b", "last_reasons_check_at"])


def test_check_date_circuit_breaker_stops_after_n_consecutive_errors():
    ids = [f"d{i}" for i in range(2)]
    docs = _docs_for(ids)
    scraper = FakeScraper(set(), fail=set(ids))
    with pytest.raises(TooManyErrors):
        check_date(
            docs,
            DatePlan("2026-04-01", ids, "direct", 2),
            scraper,
            NOW,
            Sink(docs),
            max_consecutive_errors=2,
        )
    assert scraper.requests == ["page:d0", "page:d1"]

    ids = [f"e{i}" for i in range(10)]
    docs = _docs_for(ids)
    scraper = FakeScraper(set(ids), fail=set(ids))
    with pytest.raises(TooManyErrors):
        check_date(
            docs,
            DatePlan("2026-04-01", ids, "direct", 10),
            scraper,
            NOW,
            Sink(docs),
            max_consecutive_errors=3,
        )
    # count + search, then stops after 3 failed downloads
    assert scraper.requests == ["count", "search", "page:e0", "page:e1", "page:e2"]


def test_check_date_success_resets_consecutive_errors():
    ids = ["a", "b", "c", "d"]
    docs = _docs_for(ids)
    scraper = FakeScraper(set(), fail={"a", "c"})
    result = check_date(
        docs,
        DatePlan("2026-04-01", ids[:2], "direct", 2),
        scraper,
        NOW,
        Sink(docs),
        max_consecutive_errors=2,
    )
    assert result.n_errors == 1  # "a" failed, "b" ok => no breaker


class InterruptingScraper(FakeScraper):
    def get_page_for_doc(self, doc_id):
        if doc_id == "d2":
            raise KeyboardInterrupt
        return super().get_page_for_doc(doc_id)


def test_interrupted_date_leaves_consistent_state():
    ids = [f"d{i}" for i in range(2)] + ["d2", "d3"]
    docs = _docs_for(ids)
    sink = Sink(docs)
    scraper = InterruptingScraper({"d0", "d1", "d2", "d3"})
    plan = DatePlan("2026-04-01", ids, "direct", 4)
    with patch("juddges.data.nsa.reasons.COUNT_COST", 10):  # tiny group => direct, no count
        with pytest.raises(KeyboardInterrupt):
            check_date(docs, plan, scraper, NOW, sink, flush_every=1)
    stored = {p["doc_id"] for p in sink.pages}
    assert stored == {"d0", "d1"}
    flagged = set(sink.snapshot.index[sink.snapshot["has_reasons"].fillna(False).astype(bool)])
    assert flagged <= stored and flagged == {"d0", "d1"}
    assert sink.snapshot.loc["d2", "reasons_check_count"] == 0


def test_flush_failure_keeps_flag_out_of_state():
    # pages are stored before docs; if storing pages fails nothing is saved
    docs = _docs_for(["a"])
    saved = []

    def flush(pages):
        raise OSError("disk full")  # pages not stored => docs must not be saved either

    with pytest.raises(OSError):
        check_date(docs, DatePlan("2026-04-01", ["a"], "direct", 1), FakeScraper({"a"}), NOW, flush)
    assert saved == []


def test_flatten_search_result():
    assert flatten_search_result(None) == set()
    assert flatten_search_result({1: ["a"], 2: ["b"]}) == {"a", "b"}
    with pytest.raises(ValueError):
        flatten_search_result({1: ["a"], 2: None})


def test_due_exactly_at_interval_boundary():
    docs = make_docs([{"doc_id": "a", "judgment_date": "2026-04-01", "downloaded": 30}])
    docs.loc["a", "page_downloaded_at"] = NOW - timedelta(days=30)
    assert len(find_due_documents(docs, NOW, 730, SCHEDULE)) == 1
    docs.loc["a", "page_downloaded_at"] = NOW - timedelta(days=30) + timedelta(seconds=1)
    assert len(find_due_documents(docs, NOW, 730, SCHEDULE)) == 0


def test_latest_page_tie_breaks_on_later_chunk(tmp_path):
    state = NSAState(tmp_path)
    t = pd.Timestamp(NOW)
    state.append_pages([{"doc_id": "a", "page": "first", "downloaded_at": t}])
    state.append_pages([{"doc_id": "a", "page": "second", "downloaded_at": t}])
    (latest,) = state.iter_latest_pages()
    assert latest["page"].tolist() == ["second"]


def test_iter_latest_pages_batches_by_min_rows(tmp_path):
    state = NSAState(tmp_path)
    t = pd.Timestamp(NOW)
    for i in range(5):
        state.append_pages([{"doc_id": f"d{i}", "page": "p", "downloaded_at": t}])
    sizes = [len(df) for df in state.iter_latest_pages(min_rows=2)]
    assert sizes == [2, 2, 1]


def test_malformed_trailing_line_is_skipped(tmp_path):
    state = NSAState(tmp_path)
    state.append_document_list("2026-01-01", ["a"], NOW)
    with state.lists_path.open("a") as f:
        f.write('{"date": "2026-01-02", "num_do')  # crash during append
    assert list(state.load_document_lists()) == ["2026-01-01"]


def test_lock_is_exclusive(tmp_path):
    first, second = NSAState(tmp_path), NSAState(tmp_path)
    first.acquire_lock()
    with pytest.raises(StateLocked):
        second.acquire_lock()
    first.release_lock()
    second.acquire_lock()
    second.release_lock()
