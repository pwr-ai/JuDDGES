"""Information extraction module using LangChain and Gemini."""

# Core extraction chain — requires langchain_google_vertexai (optional)
try:
    from juddges.extraction.gemini_chain import (
        DocumentType,
        ExtractionSchema,
        GeminiExtractionChain,
    )
except ImportError:
    GeminiExtractionChain = None  # type: ignore[assignment, misc]
    DocumentType = None  # type: ignore[assignment, misc]
    ExtractionSchema = None  # type: ignore[assignment, misc]

from juddges.extraction.extraction_storage import ExtractionStorage

# Schema definitions
from juddges.extraction.schema import create_polish_legal_schema

# Field mapping and transformations
from juddges.extraction.field_mapping import (
    EXTRACTION_TO_WEAVIATE_MAPPING,
    FieldMapper,
    build_update_payload,
)

# Weaviate client
from juddges.extraction.weaviate_client import WeaviateRestClient

# Statistics and reporting
from juddges.extraction.statistics import (
    ExtractionStatistics,
    FieldCoverage,
    IngestionStatistics,
    calculate_field_coverage,
    display_extraction_results,
    display_ingestion_results,
    generate_extraction_summary,
    save_extraction_results,
    save_ingestion_report,
)

# Batch processing
from juddges.extraction.batch_processor import BatchProcessor

# Ingestion service
from juddges.extraction.ingestion import WeaviateIngestionService

__all__ = [
    # Core
    "GeminiExtractionChain",
    "DocumentType",
    "ExtractionSchema",
    "ExtractionStorage",
    # Schema
    "create_polish_legal_schema",
    # Field mapping
    "FieldMapper",
    "build_update_payload",
    "EXTRACTION_TO_WEAVIATE_MAPPING",
    # Weaviate client
    "WeaviateRestClient",
    # Statistics
    "ExtractionStatistics",
    "IngestionStatistics",
    "FieldCoverage",
    "calculate_field_coverage",
    "generate_extraction_summary",
    "save_extraction_results",
    "display_extraction_results",
    "save_ingestion_report",
    "display_ingestion_results",
    # Processing
    "BatchProcessor",
    "WeaviateIngestionService",
]
