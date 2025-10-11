# Extraction Module

Information extraction module using LangChain and Google Gemini models.

## Quick Start

```python
from juddges.extraction import GeminiExtractionChain
from juddges.extraction.gemini_chain import DocumentType, ExtractionSchema

# Initialize chain
chain = GeminiExtractionChain(model_name="gemini-2.5-flash")

# Define schema
schema = ExtractionSchema(
    fields={
        "verdict_date": "date as ISO 8601",
        "court": "string, court name",
        "parties": "List[string], party names",
    },
    language="polish",
)

# Extract
result = chain.extract(
    document_type=DocumentType.JUDGMENT,
    text="Your document text...",
    schema=schema,
)
```

## Features

- ✅ Google Gemini 2.5 Pro/Flash support
- ✅ SQLite caching for cost reduction
- ✅ Langfuse observability integration
- ✅ Batch processing
- ✅ Document type-aware prompts
- ✅ Flexible schema definition

## Documentation

See [GEMINI_EXTRACTION.md](../../docs/GEMINI_EXTRACTION.md) for full documentation.

## Example Script

```bash
python scripts/extraction/extract_with_gemini.py --help
```
