"""LangChain extraction chain using Gemini 2.5 Pro with caching and observability."""

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

import langchain
from langchain.output_parsers.json import parse_json_markdown
from langchain_community.cache import SQLiteCache
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types for extraction."""

    TAX_INTERPRETATION = "tax_interpretation"
    JUDGMENT = "judgment"


class ExtractionSchema(BaseModel):
    """Schema definition for information extraction.

    Attributes:
        fields: Dictionary mapping field names to their descriptions and types
        instructions: Additional instructions for the extraction process
        language: Language of the document and extraction (e.g., 'polish', 'english')
    """

    fields: dict[str, str] = Field(
        ...,
        description="Field definitions in format 'field_name: type, description'",
        example={
            "verdict_date": "date as ISO 8601, when the verdict was issued",
            "verdict": "string, full text of the verdict",
            "parties": "List[string], names of involved parties",
        },
    )
    instructions: Optional[str] = Field(
        None,
        description="Additional extraction instructions (what/how/why)",
    )
    language: str = Field(
        default="polish",
        description="Language for extraction",
    )

    def to_schema_string(self) -> str:
        """Convert schema to string format for prompt."""
        return "\n".join(f"{key}: {val}" for key, val in self.fields.items())


class GeminiExtractionChain:
    """LangChain extraction chain using Gemini 2.5 Pro.

    Features:
    - Google Gemini 2.5 Pro/Flash model support
    - SQLite caching to avoid redundant API calls
    - Langfuse callback integration for observability
    - Structured output parsing to dictionary
    - Document type-aware prompting

    Example:
        >>> chain = GeminiExtractionChain(
        ...     model_name="gemini-2.5-pro",
        ...     cache_path="cache/extraction.db",
        ...     temperature=0.0,
        ... )
        >>>
        >>> schema = ExtractionSchema(
        ...     fields={
        ...         "verdict_date": "date as ISO 8601",
        ...         "court": "string, name of the court",
        ...     },
        ...     language="polish",
        ... )
        >>>
        >>> result = chain.extract(
        ...     document_type=DocumentType.JUDGMENT,
        ...     text="Sąd Najwyższy orzekł dnia 2024-01-15...",
        ...     schema=schema,
        ...     langfuse_handler=my_langfuse_handler,  # Optional
        ... )
        >>> print(result)  # {"verdict_date": "2024-01-15", "court": "Sąd Najwyższy"}
    """

    def __init__(
        self,
        model_name: Literal["gemini-2.5-pro", "gemini-2.5-flash"] = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        cache_path: Optional[str | Path] = None,
        max_output_tokens: Optional[int] = 8192,
    ):
        """Initialize Gemini extraction chain.

        Args:
            model_name: Gemini model to use ('gemini-2.5-pro' or 'gemini-2.5-flash')
            api_key: Google API key (if not set via GOOGLE_API_KEY env var)
            temperature: Sampling temperature (0.0 for deterministic)
            cache_path: Path to SQLite cache file (default: .cache/langchain.db)
            max_output_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        # Set up caching
        if cache_path:
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            langchain.llm_cache = SQLiteCache(database_path=str(cache_file))
            logger.info(f"Enabled LangChain SQLite cache: {cache_file}")
        else:
            default_cache = Path(".cache/langchain.db")
            default_cache.parent.mkdir(parents=True, exist_ok=True)
            langchain.llm_cache = SQLiteCache(database_path=str(default_cache))
            logger.info(f"Enabled LangChain SQLite cache: {default_cache}")

        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        logger.info(f"Initialized GeminiExtractionChain with {model_name}")

    def _build_extraction_prompt(
        self,
        document_type: DocumentType,
    ) -> ChatPromptTemplate:
        """Build extraction prompt template based on document type.

        Args:
            document_type: Type of legal document

        Returns:
            ChatPromptTemplate configured for the document type
        """
        # Base prompt adapted from existing info_extraction_json.yaml
        base_prompt = """Act as a highly skilled legal analyst specializing in extracting structured information from {document_type_name}.

Your task is to carefully analyze the provided document text and extract specific information according to the schema provided.

Key instructions:
- Language: Extract information in {language}, maintaining the original language of the document
- Accuracy: Only extract information that is explicitly stated in the text
- Empty fields: Use empty string "" when information cannot be found
- Consistency: Ensure extracted values match the specified data types and enums
- Context: Consider the full context when extracting information
- Validation: Double-check that extracted values are supported by the text
- Objectivity: Extract factual information without interpretation

For boolean fields:
- Only mark as true when explicitly confirmed in the text
- Default to false when information is unclear or not mentioned

For enum fields:
- Only use values from the provided options
- Use empty string if none of the options match exactly

For date fields:
- Use ISO 8601 format (YYYY-MM-DD)
- Extract complete dates when available
- Leave empty if date is partial or ambiguous

For list fields:
- Return as JSON array
- Include all relevant items found in the text

{additional_instructions}

Schema for extraction:
====
{schema}
====

Document text to analyze:
====
{text}
====

Format response as valid JSON, ensuring all schema fields are included. Return ONLY the JSON without any additional text or markdown formatting."""

        # Document type specific context
        doc_type_names = {
            DocumentType.JUDGMENT: "court judgments and legal decisions",
            DocumentType.TAX_INTERPRETATION: "tax interpretations and fiscal rulings",
        }

        return ChatPromptTemplate.from_template(
            base_prompt,
            partial_variables={"document_type_name": doc_type_names[document_type]},
        )

    def _build_chain(
        self,
        document_type: DocumentType,
    ) -> RunnableSequence:
        """Build the extraction chain.

        Args:
            document_type: Type of legal document

        Returns:
            Configured RunnableSequence for extraction
        """
        prompt = self._build_extraction_prompt(document_type)

        # Chain: prompt -> LLM -> parse JSON
        chain = prompt | self.llm | (lambda x: parse_json_markdown(x.content))

        return chain

    def extract(
        self,
        document_type: DocumentType,
        text: str,
        schema: ExtractionSchema,
        langfuse_handler: Optional[BaseCallbackHandler] = None,
        max_text_length: int = 150000,
    ) -> dict[str, Any]:
        """Extract structured information from document text.

        Args:
            document_type: Type of document (judgment or tax interpretation)
            text: Full text of the document
            schema: Extraction schema defining fields and instructions
            langfuse_handler: Optional Langfuse callback handler for observability
            max_text_length: Maximum text length to process (truncates if longer)

        Returns:
            Dictionary with extracted information matching schema fields

        Example:
            >>> schema = ExtractionSchema(
            ...     fields={
            ...         "verdict_date": "date as ISO 8601",
            ...         "court": "string, court name",
            ...         "case_number": "string, case identifier",
            ...     },
            ...     language="polish",
            ... )
            >>> result = chain.extract(
            ...     document_type=DocumentType.JUDGMENT,
            ...     text="Sąd Okręgowy w Warszawie...",
            ...     schema=schema,
            ... )
        """
        # Truncate text if too long
        if len(text) > max_text_length:
            logger.warning(
                f"Text length {len(text)} exceeds max {max_text_length}, truncating"
            )
            text = text[:max_text_length]

        # Build chain for this document type
        chain = self._build_chain(document_type)

        # Prepare input
        chain_input = {
            "text": text,
            "schema": schema.to_schema_string(),
            "language": schema.language,
            "additional_instructions": (
                f"\nAdditional instructions:\n{schema.instructions}"
                if schema.instructions
                else ""
            ),
        }

        # Execute with optional Langfuse callback
        config = {}
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]
            logger.debug("Executing extraction with Langfuse tracing")

        try:
            result = chain.invoke(chain_input, config=config)
            logger.info(
                f"Successfully extracted {len(result)} fields from {document_type.value}"
            )
            return result
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise

    def batch_extract(
        self,
        document_type: DocumentType,
        texts: list[str],
        schema: ExtractionSchema,
        langfuse_handler: Optional[BaseCallbackHandler] = None,
        max_text_length: int = 150000,
    ) -> list[dict[str, Any]]:
        """Extract information from multiple documents in batch.

        Args:
            document_type: Type of documents
            texts: List of document texts
            schema: Extraction schema
            langfuse_handler: Optional Langfuse callback handler
            max_text_length: Maximum text length per document

        Returns:
            List of extraction results as dictionaries
        """
        chain = self._build_chain(document_type)

        # Prepare batch inputs
        batch_inputs = [
            {
                "text": text[:max_text_length],
                "schema": schema.to_schema_string(),
                "language": schema.language,
                "additional_instructions": (
                    f"\nAdditional instructions:\n{schema.instructions}"
                    if schema.instructions
                    else ""
                ),
            }
            for text in texts
        ]

        # Execute batch
        config = {}
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]

        try:
            results = chain.batch(batch_inputs, config=config)
            logger.info(
                f"Successfully extracted from {len(results)} {document_type.value} documents"
            )
            return results
        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            raise
