"""Unit tests for Gemini extraction chain."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("langchain_google_vertexai", reason="langchain_google_vertexai not installed")

from langchain_core.messages import AIMessage

from juddges.extraction.gemini_chain import (
    DocumentType,
    ExtractionSchema,
    GeminiExtractionChain,
)


@pytest.fixture
def sample_judgment_schema() -> ExtractionSchema:
    """Sample schema for judgment extraction."""
    return ExtractionSchema(
        fields={
            "verdict_date": "date as ISO 8601, when the verdict was issued",
            "court": "string, name of the court",
            "case_number": "string, case identifier",
            "parties": "List[string], involved parties",
        },
        instructions="Extract factual information only",
        language="polish",
    )


@pytest.fixture
def sample_judgment_text() -> str:
    """Sample judgment text."""
    return """
    WYROK
    W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ

    Dnia 15 stycznia 2024 r.

    Sąd Okręgowy w Warszawie, V Wydział Cywilny
    w składzie:
    Przewodniczący: SSO Anna Kowalska

    Sprawa z powództwa Jana Kowalskiego
    przeciwko Bankowi XYZ S.A.
    o zapłatę

    Sygn. akt: V C 123/23

    I. Zasądza od pozwanego Banku XYZ S.A. na rzecz powoda Jana Kowalskiego
    kwotę 50.000 zł wraz z odsetkami ustawowymi.
    """


class TestExtractionSchema:
    """Tests for ExtractionSchema."""

    def test_schema_creation(self):
        """Test schema can be created with valid fields."""
        schema = ExtractionSchema(
            fields={"field1": "string, description", "field2": "date as ISO 8601"},
            language="polish",
        )
        assert "field1" in schema.fields
        assert "field2" in schema.fields
        assert schema.language == "polish"

    def test_schema_to_string(self):
        """Test schema converts to string format."""
        schema = ExtractionSchema(
            fields={"verdict_date": "date as ISO 8601", "court": "string, court name"},
            language="polish",
        )
        schema_str = schema.to_schema_string()
        assert "verdict_date: date as ISO 8601" in schema_str
        assert "court: string, court name" in schema_str

    def test_schema_with_instructions(self):
        """Test schema can include instructions."""
        schema = ExtractionSchema(
            fields={"field1": "string"},
            instructions="Extract carefully",
            language="english",
        )
        assert schema.instructions == "Extract carefully"


class TestGeminiExtractionChain:
    """Tests for GeminiExtractionChain."""

    @patch("juddges.extraction.gemini_chain.ChatGoogleGenerativeAI")
    def test_chain_initialization(self, mock_llm_class):
        """Test chain can be initialized."""
        chain = GeminiExtractionChain(
            model_name="gemini-2.5-flash",
            temperature=0.0,
            cache_path=None,
        )
        assert chain.model_name == "gemini-2.5-flash"
        assert chain.temperature == 0.0
        mock_llm_class.assert_called_once()

    @patch("juddges.extraction.gemini_chain.ChatGoogleGenerativeAI")
    def test_chain_with_cache(self, mock_llm_class):
        """Test chain creates cache directory."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.db"
            chain = GeminiExtractionChain(
                model_name="gemini-2.5-flash",
                cache_path=str(cache_path),
            )
            assert cache_path.parent.exists()

    @patch("juddges.extraction.gemini_chain.ChatGoogleGenerativeAI")
    def test_build_prompt_for_judgment(self, mock_llm_class):
        """Test prompt building for judgment documents."""
        chain = GeminiExtractionChain()
        prompt = chain._build_extraction_prompt(DocumentType.JUDGMENT)

        # Check prompt contains judgment-specific context
        prompt_template = prompt.format(
            text="test",
            schema="test",
            language="polish",
            additional_instructions="",
        )
        assert "court judgments" in prompt_template.lower() or "legal decisions" in prompt_template.lower()

    @patch("juddges.extraction.gemini_chain.ChatGoogleGenerativeAI")
    def test_build_prompt_for_tax_interpretation(self, mock_llm_class):
        """Test prompt building for tax interpretation documents."""
        chain = GeminiExtractionChain()
        prompt = chain._build_extraction_prompt(DocumentType.TAX_INTERPRETATION)

        prompt_template = prompt.format(
            text="test",
            schema="test",
            language="polish",
            additional_instructions="",
        )
        assert "tax" in prompt_template.lower()

    @patch("juddges.extraction.gemini_chain.ChatGoogleGenerativeAI")
    @patch("juddges.extraction.gemini_chain.langchain")
    def test_extract_with_mock(
        self, mock_langchain, mock_llm_class, sample_judgment_schema, sample_judgment_text
    ):
        """Test extraction with mocked LLM response."""
        # Disable caching for tests
        mock_langchain.llm_cache = None

        # Mock the LLM to return a structured response
        mock_llm = MagicMock()
        mock_response = AIMessage(
            content='```json\n{"verdict_date": "2024-01-15", "court": "Sąd Okręgowy w Warszawie", "case_number": "V C 123/23", "parties": ["Jan Kowalski", "Bank XYZ S.A."]}\n```'
        )
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        chain = GeminiExtractionChain(cache_path=None)
        chain.llm = mock_llm  # Use the mock directly

        result = chain.extract(
            document_type=DocumentType.JUDGMENT,
            text=sample_judgment_text,
            schema=sample_judgment_schema,
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "verdict_date" in result
        assert "court" in result
        assert result["verdict_date"] == "2024-01-15"
        assert "Warszaw" in result["court"]

    @patch("juddges.extraction.gemini_chain.ChatGoogleGenerativeAI")
    @patch("juddges.extraction.gemini_chain.langchain")
    def test_extract_truncates_long_text(self, mock_langchain, mock_llm_class):
        """Test that long texts are truncated."""
        mock_langchain.llm_cache = None

        mock_llm = MagicMock()
        mock_response = AIMessage(content='```json\n{"field": "value"}\n```')
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        chain = GeminiExtractionChain(cache_path=None)
        chain.llm = mock_llm
        schema = ExtractionSchema(fields={"field": "string"}, language="polish")

        long_text = "x" * 200000  # Longer than default max
        result = chain.extract(
            document_type=DocumentType.JUDGMENT,
            text=long_text,
            schema=schema,
            max_text_length=1000,
        )

        # Verify LLM was called with truncated text
        call_args = mock_llm.invoke.call_args
        assert len(call_args[0][0].messages[0].content) <= 5000  # Check prompt length is reasonable

    @patch("juddges.extraction.gemini_chain.ChatGoogleGenerativeAI")
    @patch("juddges.extraction.gemini_chain.langchain")
    def test_batch_extract(self, mock_langchain, mock_llm_class, sample_judgment_schema):
        """Test batch extraction."""
        mock_langchain.llm_cache = None

        mock_llm = MagicMock()
        mock_response = AIMessage(
            content='```json\n{"verdict_date": "2024-01-15", "court": "Sąd", "case_number": "123", "parties": []}\n```'
        )
        mock_llm.batch.return_value = [mock_response, mock_response]
        mock_llm_class.return_value = mock_llm

        chain = GeminiExtractionChain(cache_path=None)
        chain.llm = mock_llm

        texts = ["text1", "text2"]
        results = chain.batch_extract(
            document_type=DocumentType.JUDGMENT,
            texts=texts,
            schema=sample_judgment_schema,
        )

        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)

    @patch("juddges.extraction.gemini_chain.ChatGoogleGenerativeAI")
    @patch("juddges.extraction.gemini_chain.langchain")
    def test_extract_with_langfuse_handler(
        self, mock_langchain, mock_llm_class, sample_judgment_schema, sample_judgment_text
    ):
        """Test extraction with Langfuse callback handler."""
        mock_langchain.llm_cache = None

        mock_llm = MagicMock()
        mock_response = AIMessage(
            content='```json\n{"verdict_date": "2024-01-15", "court": "Sąd", "case_number": "123", "parties": []}\n```'
        )
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        chain = GeminiExtractionChain(cache_path=None)
        chain.llm = mock_llm
        mock_handler = MagicMock()

        result = chain.extract(
            document_type=DocumentType.JUDGMENT,
            text=sample_judgment_text,
            schema=sample_judgment_schema,
            langfuse_handler=mock_handler,
        )

        # Verify extraction succeeded
        assert isinstance(result, dict)
        # Note: Langfuse config is passed internally, hard to verify with mocks


class TestDocumentType:
    """Tests for DocumentType enum."""

    def test_document_types_exist(self):
        """Test all expected document types are defined."""
        assert DocumentType.JUDGMENT == "judgment"
        assert DocumentType.TAX_INTERPRETATION == "tax_interpretation"

    def test_can_iterate_document_types(self):
        """Test can iterate over document types."""
        types = list(DocumentType)
        assert len(types) >= 2
        assert DocumentType.JUDGMENT in types
        assert DocumentType.TAX_INTERPRETATION in types
