from pathlib import Path

from juddges.data.nsa.extractor import has_reasons

FIXTURES = Path(__file__).parent / "fixtures"


def test_has_reasons_false_for_sentencja_only():
    assert not has_reasons((FIXTURES / "judgment_sentencja_only.html").read_text())


def test_has_reasons_true_for_page_with_reasons():
    assert has_reasons((FIXTURES / "judgment_with_reasons.html").read_text())
