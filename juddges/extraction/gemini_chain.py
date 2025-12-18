"""LangChain extraction chain using Gemini 2.5 Pro with caching and observability."""

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

import langchain
from langchain.output_parsers.json import parse_json_markdown
from langchain_community.cache import SQLAlchemyCache, SQLAlchemyMd5Cache, SQLiteCache
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_vertexai import ChatVertexAI
from loguru import logger
from pydantic import BaseModel, Field, create_model


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

    def to_pydantic_model(self, model_name: str = "ExtractionOutput") -> type[BaseModel]:
        """Convert schema to a Pydantic model for structured output.

        Creates a dynamic Pydantic model with all fields as Optional[Any] to handle
        the variety of data types defined in the schema (strings, lists, dicts, etc.).

        Args:
            model_name: Name for the generated Pydantic model

        Returns:
            Dynamically created Pydantic BaseModel class
        """
        # Create field definitions - all fields are Optional[Any] to handle diverse types
        field_definitions = {
            field_name: (
                Optional[Any],
                Field(default=None, description=field_desc[:500]),
            )  # Truncate long descriptions
            for field_name, field_desc in self.fields.items()
        }

        # Create dynamic Pydantic model
        return create_model(
            model_name,
            **field_definitions,
            __doc__=f"Structured extraction output for {self.language} legal documents",
        )


class GeminiExtractionChain:
    """LangChain extraction chain using Gemini 2.5 Pro with guaranteed valid JSON output.

    Features:
    - Google Gemini 2.5 Pro/Flash model support
    - Native structured output via with_structured_output() - guarantees valid JSON responses
    - Eliminates JSON parsing errors by using Gemini's response_schema API
    - Optional extended thinking mode for Gemini 2.5 models (disabled by default)
    - PostgreSQL caching (via POSTGRES_CACHE_URL env var) with SQLite fallback
    - Langfuse callback integration for observability
    - Document type-aware prompting
    - Dynamic Pydantic model generation from ExtractionSchema

    Thinking Mode (Gemini 2.5 only):
    - Extended thinking mode shows the model's reasoning process before providing the answer
    - Can improve accuracy for complex reasoning tasks
    - Increases latency and token usage
    - Default: disabled (enable_thinking=False)
    - Recommended: keep disabled for structured extraction tasks, enable for complex reasoning

    Cache Configuration:
    - Set POSTGRES_CACHE_URL environment variable for PostgreSQL caching
    - Falls back to SQLite if PostgreSQL is unavailable or not configured
    - Custom SQLite path can be specified via cache_path parameter

    Example (default - no thinking):
        >>> chain = GeminiExtractionChain(
        ...     model_name="gemini-2.5-pro",
        ...     cache_path="cache/extraction.db",  # SQLite fallback path
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

    Example (with thinking enabled):
        >>> chain = GeminiExtractionChain(
        ...     model_name="gemini-2.5-pro",
        ...     enable_thinking=True,  # Enable extended thinking mode
        ... )
        >>> # Model will show reasoning process in responses
    """

    def __init__(
        self,
        model_name: Literal[
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ] = "gemini-2.5-pro",
        project: Optional[str] = None,
        location: str = "us-central1",
        temperature: float = 0.0,
        cache_path: Optional[str | Path] = None,
        max_output_tokens: Optional[int] = 8192,
        enable_thinking: bool = False,
    ):
        """Initialize Gemini extraction chain using Vertex AI.

        Args:
            model_name: Gemini model to use (via Vertex AI)
            project: GCP project ID (defaults to VERTEX_PROJECT or gcloud default)
            location: GCP region (default: us-central1)
            temperature: Sampling temperature (0.0 for deterministic)
            cache_path: Path to SQLite cache file (used as fallback if PostgreSQL unavailable)
            max_output_tokens: Maximum tokens in response
            enable_thinking: Enable extended thinking mode for Gemini 2.5 models (default: False).
                           When enabled, the model shows its reasoning process before answering.
                           This can improve accuracy for complex tasks but increases latency and token usage.
                           Recommended for complex reasoning tasks, not for simple structured extraction.

        Environment Variables:
            POSTGRES_CACHE_URL: PostgreSQL connection string for LLM caching (preferred)
            VERTEX_PROJECT: GCP project ID for Vertex AI
            GOOGLE_CLOUD_PROJECT: Alternative GCP project ID
        """
        import os

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.enable_thinking = enable_thinking

        # Get project from env or parameter
        self.project = project or os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project:
            logger.warning("No GCP project specified, will use gcloud default")

        self.location = location

        # Set up caching - prefer PostgreSQL via SQLAlchemy with MD5, fallback to SQLite
        postgres_url = os.getenv("POSTGRES_CACHE_URL")

        if postgres_url:
            try:
                from sqlalchemy import create_engine

                # Create PostgreSQL cache using SQLAlchemyMd5Cache (avoids 8KB index size limit)
                engine = create_engine(postgres_url)
                langchain.llm_cache = SQLAlchemyMd5Cache(engine)
                # Extract host/port from URL for logging (format: postgresql://user:pass@host:port/db)
                db_location = postgres_url.split("@")[1] if "@" in postgres_url else "configured"
                logger.info(f"Enabled LangChain PostgreSQL cache (SQLAlchemy MD5): {db_location}")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize PostgreSQL cache: {e}, falling back to SQLite"
                )
                # Fallback to SQLite
                default_cache = Path(".cache/langchain.db")
                default_cache.parent.mkdir(parents=True, exist_ok=True)
                langchain.llm_cache = SQLiteCache(database_path=str(default_cache))
                logger.info(f"Enabled LangChain SQLite cache (fallback): {default_cache}")
        elif cache_path:
            # Use custom SQLite path if provided
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            langchain.llm_cache = SQLiteCache(database_path=str(cache_file))
            logger.info(f"Enabled LangChain SQLite cache: {cache_file}")
        else:
            # Default SQLite cache
            default_cache = Path(".cache/langchain.db")
            default_cache.parent.mkdir(parents=True, exist_ok=True)
            langchain.llm_cache = SQLiteCache(database_path=str(default_cache))
            logger.info(f"Enabled LangChain SQLite cache: {default_cache}")

        # Initialize Vertex AI Gemini model (uses application default credentials)
        # Configure model kwargs for thinking mode if enabled
        model_kwargs = {}
        if self.enable_thinking and "2.5" in model_name:
            # Gemini 2.5 models support extended thinking mode
            # This enables the model to show its reasoning process
            model_kwargs["thinking"] = True
            logger.info(f"Extended thinking mode enabled for {model_name}")
        elif self.enable_thinking:
            logger.warning(
                f"Thinking mode requested but not supported for {model_name}. "
                "Only Gemini 2.5 models support extended thinking."
            )

        # Build kwargs for ChatVertexAI - only include model_kwargs if not empty
        llm_kwargs = {
            "model": model_name,
            "project": self.project,
            "location": self.location,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }

        # Only add model_kwargs if there are any (empty dict causes issues with LangChain)
        if model_kwargs:
            llm_kwargs["model_kwargs"] = model_kwargs

        self.llm = ChatVertexAI(**llm_kwargs)

        logger.info(
            f"Initialized VertexAI GeminiExtractionChain with {model_name} "
            f"(project: {self.project}, location: {self.location}, "
            f"thinking: {self.enable_thinking})"
        )

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
            DocumentType.TAX_INTERPRETATION: "tax interpretations",
        }

        return ChatPromptTemplate.from_template(
            base_prompt,
            partial_variables={"document_type_name": doc_type_names[document_type]},
        )

    def _build_chain(
        self,
        document_type: DocumentType,
        schema: ExtractionSchema,
    ) -> RunnableSequence:
        """Build the extraction chain with structured output.

        Args:
            document_type: Type of legal document
            schema: Extraction schema to convert to Pydantic model

        Returns:
            Configured RunnableSequence for extraction with guaranteed valid JSON
        """
        prompt = self._build_extraction_prompt(document_type)

        # Convert schema to Pydantic model for structured output
        pydantic_model = schema.to_pydantic_model(f"{document_type.value}_extraction")

        # Use with_structured_output to guarantee valid JSON responses
        # This uses Gemini's native structured output API (response_schema)
        structured_llm = self.llm.with_structured_output(pydantic_model)

        # Chain: prompt -> structured LLM (returns Pydantic model) -> convert to dict
        chain = (
            prompt
            | structured_llm
            | (lambda x: x.model_dump() if hasattr(x, "model_dump") else x.dict())
        )

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
            logger.warning(f"Text length {len(text)} exceeds max {max_text_length}, truncating")
            text = text[:max_text_length]

        # Build chain for this document type with structured output
        chain = self._build_chain(document_type, schema)

        # Prepare input
        chain_input = {
            "text": text,
            "schema": schema.to_schema_string(),
            "language": schema.language,
            "additional_instructions": (
                f"\nAdditional instructions:\n{schema.instructions}" if schema.instructions else ""
            ),
        }

        # Execute with optional Langfuse callback
        config = {}
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]
            logger.debug("Executing extraction with Langfuse tracing (structured output mode)")

        try:
            result = chain.invoke(chain_input, config=config)

            # Check if result is None (API returned nothing)
            if result is None:
                error_msg = "API returned None - likely rate limit, timeout, or API error"
                logger.error(f"Extraction failed: {error_msg}")
                raise ValueError(error_msg)

            logger.info(
                f"Successfully extracted {len(result)} fields from {document_type.value} using structured output"
            )
            return result
        except Exception as e:
            # Enhanced error logging with exception details
            error_type = type(e).__name__
            error_details = {
                "error_type": error_type,
                "error_message": str(e),
                "document_type": document_type.value,
                "text_length": len(text),
            }

            # Check for specific API errors
            if hasattr(e, 'code'):
                error_details["http_code"] = e.code
            if hasattr(e, 'status_code'):
                error_details["status_code"] = e.status_code

            # Log detailed error
            logger.error(
                f"Extraction failed: {error_type} - {str(e)} | "
                f"Details: {error_details}"
            )
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
        # Build chain with structured output
        chain = self._build_chain(document_type, schema)

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

            # Check for None results in batch
            if results is None or None in results:
                none_count = results.count(None) if results else len(texts)
                error_msg = f"API returned None for {none_count}/{len(texts)} documents - likely rate limit or API error"
                logger.error(f"Batch extraction failed: {error_msg}")
                raise ValueError(error_msg)

            logger.info(
                f"Successfully extracted from {len(results)} {document_type.value} documents using structured output"
            )
            return results
        except Exception as e:
            # Enhanced error logging
            error_type = type(e).__name__
            error_details = {
                "error_type": error_type,
                "error_message": str(e),
                "document_type": document_type.value,
                "batch_size": len(texts),
            }

            # Check for specific API errors
            if hasattr(e, 'code'):
                error_details["http_code"] = e.code
            if hasattr(e, 'status_code'):
                error_details["status_code"] = e.status_code
            if "429" in str(e):
                error_details["likely_cause"] = "Rate limit exceeded"
            elif "500" in str(e) or "503" in str(e):
                error_details["likely_cause"] = "API server error"
            elif "timeout" in str(e).lower():
                error_details["likely_cause"] = "Request timeout"

            logger.error(
                f"Batch extraction failed: {error_type} - {str(e)} | "
                f"Details: {error_details}"
            )
            raise
