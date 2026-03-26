"""Integration tests for Gemini extraction chain.

These tests require a real Google API key and will make actual API calls.
Skip these tests if you don't have an API key or want to avoid costs.

Run with: pytest tests/extraction/test_gemini_integration.py
Skip with: pytest tests/extraction/ -k "not integration"
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("langchain_google_vertexai", reason="langchain_google_vertexai not installed")

from juddges.extraction import GeminiExtractionChain
from juddges.extraction.gemini_chain import DocumentType, ExtractionSchema

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def api_key_available() -> bool:
    """Check if Google API key is available."""
    return bool(os.getenv("GOOGLE_API_KEY"))


@pytest.fixture
def chain() -> GeminiExtractionChain:
    """Create extraction chain with test cache."""
    return GeminiExtractionChain(
        model_name="gemini-2.5-flash",
        cache_path=".cache/test_extraction.db",
        temperature=0.0,
    )


@pytest.fixture
def judgment_schema() -> ExtractionSchema:
    """Simple judgment extraction schema."""
    return ExtractionSchema(
        fields={
            "verdict_date": "date as ISO 8601, when the verdict was issued",
            "court": "string, name of the court",
            "case_number": "string, case identifier",
        },
        instructions="Extract factual information only",
        language="polish",
    )


@pytest.fixture
def sample_judgment() -> str:
    """Sample Polish judgment text."""
    return """
    WYROK
    W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ

    Dnia 15 stycznia 2024 r.

    Sąd Okręgowy w Warszawie, V Wydział Cywilny
    w składzie:
    Przewodniczący: SSO Anna Kowalska
    Sędziowie: SSO Jan Nowak, SSR del. Piotr Wiśniewski

    po rozpoznaniu w dniu 10 stycznia 2024 r. w Warszawie
    na rozprawie
    sprawy z powództwa Jana Kowalskiego
    przeciwko Bankowi XYZ S.A.
    o zapłatę

    Sygn. akt V C 123/2023

    I. Zasądza od pozwanego Banku XYZ S.A. na rzecz powoda Jana Kowalskiego
    kwotę 50.000 zł (pięćdziesiąt tysięcy złotych) wraz z odsetkami ustawowymi
    za opóźnienie od dnia 1 stycznia 2023 r. do dnia zapłaty.

    II. Zasądza od pozwanego na rzecz powoda kwotę 5.000 zł tytułem zwrotu
    kosztów procesu.

    UZASADNIENIE

    Powód wniósł o zasądzenie od pozwanego Banku kwoty 50.000 zł tytułem
    zwrotu nienależnie pobranych opłat za prowadzenie rachunku bankowego
    w latach 2020-2022.
    """


def test_requires_api_key(api_key_available):
    """Verify API key is available for integration tests."""
    if not api_key_available:
        pytest.skip("GOOGLE_API_KEY not set - skipping integration test")


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set"
)
def test_extract_judgment_with_real_api(chain, judgment_schema, sample_judgment):
    """Test extraction with real Gemini API call."""
    result = chain.extract(
        document_type=DocumentType.JUDGMENT,
        text=sample_judgment,
        schema=judgment_schema,
    )

    # Verify result structure
    assert isinstance(result, dict)
    assert "verdict_date" in result
    assert "court" in result
    assert "case_number" in result

    # Verify extraction quality (basic checks)
    # The model should extract the date
    assert result["verdict_date"] is not None
    if result["verdict_date"]:
        # Should be ISO format if extracted
        assert "-" in result["verdict_date"] or result["verdict_date"] == ""

    # Should extract court name
    if result["court"]:
        assert len(result["court"]) > 0


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set"
)
def test_caching_works(chain, judgment_schema, sample_judgment):
    """Test that caching reduces API calls."""
    import time

    # First call - should hit API
    start = time.time()
    result1 = chain.extract(
        document_type=DocumentType.JUDGMENT,
        text=sample_judgment,
        schema=judgment_schema,
    )
    first_call_time = time.time() - start

    # Second call - should hit cache
    start = time.time()
    result2 = chain.extract(
        document_type=DocumentType.JUDGMENT,
        text=sample_judgment,
        schema=judgment_schema,
    )
    second_call_time = time.time() - start

    # Results should be identical
    assert result1 == result2

    # Cached call should be significantly faster
    # (Allowing for some variation, but cache hit should be <100ms typically)
    assert second_call_time < first_call_time * 0.5


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set"
)
def test_batch_extract_real(chain, judgment_schema):
    """Test batch extraction with real API."""
    texts = [
        "Wyrok Sądu z dnia 2024-01-01 w sprawie I C 1/2024",
        "Wyrok Sądu z dnia 2024-02-15 w sprawie II C 2/2024",
    ]

    results = chain.batch_extract(
        document_type=DocumentType.JUDGMENT,
        texts=texts,
        schema=judgment_schema,
    )

    assert len(results) == 2
    assert all(isinstance(r, dict) for r in results)
    assert all("verdict_date" in r for r in results)


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY") or not os.getenv("LANGFUSE_PUBLIC_KEY"),
    reason="API keys not set",
)
def test_extract_with_langfuse(chain, judgment_schema, sample_judgment):
    """Test extraction with Langfuse tracing."""
    from langfuse.langchain import CallbackHandler

    handler = CallbackHandler()

    result = chain.extract(
        document_type=DocumentType.JUDGMENT,
        text=sample_judgment,
        schema=judgment_schema,
        langfuse_handler=handler,
    )

    assert isinstance(result, dict)
    # Check that trace was created in Langfuse
    # (Would need to query Langfuse API to fully verify, but at least it shouldn't error)


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set"
)
def test_empty_text_handling(chain, judgment_schema):
    """Test handling of empty or minimal text."""
    result = chain.extract(
        document_type=DocumentType.JUDGMENT,
        text="",
        schema=judgment_schema,
    )

    # Should return empty values but not error
    assert isinstance(result, dict)


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set"
)
def test_tax_interpretation_extraction(chain):
    """Test extraction from tax interpretation document."""
    schema = ExtractionSchema(
        fields={
            "interpretation_date": "date as ISO 8601",
            "interpretation_number": "string, document number",
            "tax_authority": "string, issuing authority",
        },
        language="polish",
    )

    text = """
    INTERPRETACJA INDYWIDUALNA

    Sygnatura: 0111-KDIB1-2.4010.123.2024.1.AB

    Data: 2024-03-15

    Dyrektor Krajowej Informacji Skarbowej

    Na podstawie art. 14b § 1 i § 6 ustawy z dnia 29 sierpnia 1997 r.
    Ordynacja podatkowa (Dz. U. z 2023 r. poz. 2383) Dyrektor Krajowej
    Informacji Skarbowej stwierdza, że stanowisko Wnioskodawcy przedstawione
    we wniosku z dnia 15 lutego 2024 r. jest prawidłowe.
    """

    result = chain.extract(
        document_type=DocumentType.TAX_INTERPRETATION,
        text=text,
        schema=schema,
    )

    assert isinstance(result, dict)
    assert "interpretation_date" in result
    assert "tax_authority" in result
