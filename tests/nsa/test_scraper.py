from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from requests import HTTPError, Request
from requests.exceptions import ConnectionError as RequestsConnectionError

from juddges.data.nsa.scraper import NSAScraper

FIXTURES = Path(__file__).parent / "fixtures"
FORM_URL = "https://orzeczenia.nsa.gov.pl/cbo/query"


@pytest.fixture
def scraper() -> NSAScraper:
    return NSAScraper(user_agent="test", wait=False)


def _submit_and_capture_body(scraper: NSAScraper, with_reasons: bool | None) -> dict[str, str]:
    """Run the form-filling helper on the real form fixture and return the POST body fields."""
    captured = {}

    def fake_open(url: str) -> None:
        scraper.browser.open_fake_page((FIXTURES / "search_form.html").read_text(), url=FORM_URL)

    def fake_submit() -> None:
        form = scraper.browser.form
        kwargs = scraper.browser.get_request_kwargs(form.form, FORM_URL)
        prepared = Request(**kwargs).prepare()
        body = prepared.body
        captured["method"] = prepared.method
        captured["body"] = dict(
            pair.split("=", 1)
            for pair in (body.decode() if isinstance(body, bytes) else body).split("&")
        )

    with (
        patch.object(scraper, "_browser_open", side_effect=fake_open),
        patch.object(scraper, "_browser_submit_selected", side_effect=fake_submit),
    ):
        scraper._submit_date_search("2025-10-15", with_reasons)
    assert captured["method"] == "POST"
    return captured["body"]


def test_with_reasons_true_sends_only_tak(scraper):
    body = _submit_and_capture_body(scraper, True)
    assert body["takUzasadnienie"] == "on"
    assert "nieUzasadnienie" not in body
    assert body["odDaty"] == body["doDaty"] == "2025-10-15"


def test_with_reasons_false_sends_only_nie(scraper):
    body = _submit_and_capture_body(scraper, False)
    assert body["nieUzasadnienie"] == "on"
    assert "takUzasadnienie" not in body


def test_with_reasons_none_sends_no_checkbox(scraper):
    body = _submit_and_capture_body(scraper, None)
    assert "takUzasadnienie" not in body
    assert "nieUzasadnienie" not in body


def test_public_methods_pass_with_reasons(scraper):
    results = (FIXTURES / "search_results.html").read_text()
    with (
        patch.object(scraper, "_submit_date_search") as submit,
        patch.object(scraper, "_any_documents_found", return_value=True),
        patch.object(scraper, "_extract_document_count", return_value=634),
    ):
        scraper.browser.open_fake_page(results, url="https://orzeczenia.nsa.gov.pl/cbo/find")
        assert scraper.get_num_docs_for_date("2025-10-15", with_reasons=True) == 634
    submit.assert_called_once_with("2025-10-15", True)


def test_wait_range_used_for_sleep():
    scraper = NSAScraper(user_agent="test", wait_range=(5, 10))
    response = MagicMock()
    with (
        patch.object(scraper, "_correct_page", return_value=True),
        patch("juddges.data.nsa.scraper.time.sleep") as sleep,
    ):
        scraper._post_call(response)
    (slept,) = sleep.call_args.args
    assert 5 <= slept <= 10


def test_wait_false_does_not_sleep(scraper):
    with (
        patch.object(scraper, "_correct_page", return_value=True),
        patch("juddges.data.nsa.scraper.time.sleep") as sleep,
    ):
        scraper._post_call(MagicMock())
    sleep.assert_not_called()


def test_invalid_wait_range():
    with pytest.raises(ValueError):
        NSAScraper(user_agent="test", wait_range=(3, 1))


def test_retry_policy_does_not_retry_403(scraper):
    retries = scraper.browser.session.get_adapter("https://orzeczenia.nsa.gov.pl").max_retries
    assert 403 not in retries.status_forcelist
    assert {429, 500, 502, 503, 504} <= set(retries.status_forcelist)
    assert retries.respect_retry_after_header is True


def test_wait_applied_even_when_response_is_an_error():
    scraper = NSAScraper(user_agent="test", wait_range=(5, 10))
    response = MagicMock()
    response.raise_for_status.side_effect = HTTPError("503")
    with patch("juddges.data.nsa.scraper.time.sleep") as sleep:
        with pytest.raises(HTTPError):
            scraper._post_call(response)
    assert 5 <= sleep.call_args.args[0] <= 10


def test_urllib3_retry_is_small_and_slow(scraper):
    retries = scraper.browser.session.get_adapter("https://orzeczenia.nsa.gov.pl").max_retries
    assert retries.total <= 2
    assert retries.backoff_factor >= 5


def test_failing_requests_have_bounded_attempts_and_sleep_between_them():
    scraper = NSAScraper(user_agent="test", wait_range=(5, 10))
    response = MagicMock()
    response.raise_for_status.side_effect = HTTPError("503")
    with (
        patch.object(scraper.browser, "open", return_value=response) as open_,
        patch("juddges.data.nsa.scraper.time.sleep") as sleep,
    ):
        with pytest.raises(HTTPError):
            scraper.get_page_for_doc("abc")
    assert open_.call_count == 5
    # every attempt waits wait_range, and the retry decorator adds a growing delay between them
    waits = [c.args[0] for c in sleep.call_args_list]
    assert len(waits) >= 5 + 4
    assert sum(w for w in waits if w > 10) > 0


def test_connection_errors_also_wait():
    scraper = NSAScraper(user_agent="test", wait_range=(5, 10))
    with (
        patch.object(scraper.browser, "open", side_effect=RequestsConnectionError("x")) as open_,
        patch("juddges.data.nsa.scraper.time.sleep") as sleep,
    ):
        with pytest.raises(RequestsConnectionError):
            scraper.get_page_for_doc("abc")
    assert open_.call_count == 5
    assert sleep.call_count >= 5 + 4


def test_search_with_expected_count_zero_does_not_submit(scraper):
    with (
        patch.object(scraper, "get_num_docs_for_date") as count,
        patch.object(scraper, "_submit_date_search") as submit,
    ):
        assert scraper.search_documents_for_date("2025-10-15", True, expected_count=0) is None
    count.assert_not_called()
    submit.assert_not_called()


def test_search_without_expected_count_counts_first_and_skips_empty(scraper):
    with (
        patch.object(scraper, "get_num_docs_for_date", return_value=0) as count,
        patch.object(scraper, "_submit_date_search") as submit,
    ):
        assert scraper.search_documents_for_date("2025-10-15") is None
    count.assert_called_once_with("2025-10-15", None)
    submit.assert_not_called()


def test_search_with_expected_count_skips_internal_count(scraper):
    with (
        patch.object(scraper, "get_num_docs_for_date") as count,
        patch.object(scraper, "_submit_date_search"),
        patch.object(scraper, "_any_documents_found", return_value=True),
        patch.object(scraper, "_retrieve_documents", return_value={1: ["a", "b"]}),
    ):
        assert scraper.search_documents_for_date("2025-10-15", True, expected_count=2) == {
            1: ["a", "b"]
        }
    count.assert_not_called()
