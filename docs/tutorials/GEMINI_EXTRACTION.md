# Gemini 2.5 Information Extraction

This document describes the Gemini-based information extraction chain for extracting structured information from legal documents using Google's Gemini 2.5 Pro and Flash models.

## Overview

The `GeminiExtractionChain` provides a LangChain-based extraction pipeline that:

- Extracts structured information from legal documents (judgments, tax interpretations)
- Uses Google Gemini 2.5 Pro/Flash models for high-quality extraction
- Implements SQLite caching to avoid redundant API calls
- Supports Langfuse callback integration for observability
- Returns structured dictionaries matching user-defined schemas

## Features

### 🚀 Key Capabilities

1. **Document Type Awareness**: Specialized prompts for judgments vs. tax interpretations
2. **Caching**: SQLite-based LangChain cache reduces API costs
3. **Observability**: Optional Langfuse integration for tracing and monitoring
4. **Batch Processing**: Efficient batch extraction for multiple documents
5. **Flexible Schemas**: Define custom extraction schemas with detailed instructions
6. **Structured Output**: Returns clean dictionaries with parsed JSON

### 🎯 Supported Document Types

- `DocumentType.JUDGMENT` - Court judgments and legal decisions
- `DocumentType.TAX_INTERPRETATION` - Tax interpretations and fiscal rulings

## Installation

Install the required dependencies:

```bash
# Using uv (recommended)
uv pip install -e ".[full]"

# Or using pip
pip install -e ".[full]"
```

This installs:

- `langchain-google-genai>=2.0.8` - Gemini model integration
- `langfuse>=2.59.1` - Observability and tracing

## ⚠️ Important: Authentication Setup

If you have Google Cloud SDK (`gcloud`) installed, you may encounter 403 authentication errors when using LangChain with Gemini. This is because LangChain tries to use Application Default Credentials (ADC) before checking for API keys.

**✅ Solution 1: Use Helper Script (Recommended)**

```bash
./scripts/extraction/run_extraction.sh test_langfuse_simple.py
./scripts/extraction/run_extraction.sh run_10_examples.py
```

**✅ Solution 2: Disable Google Cloud SDK Temporarily**

```bash
CLOUDSDK_CONFIG=/dev/null python scripts/extraction/your_script.py
```

**✅ Solution 3: Explicitly Pass API Key in Code**

```python
import os
chain = GeminiExtractionChain(
    model_name="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),  # ✅ Explicit API key
)
```

📚 **Full details:** See [Gemini API Troubleshooting](../how-to/troubleshooting/GEMINI_API_TROUBLESHOOTING.md) for complete explanation and troubleshooting.

## Quick Start

### Basic Usage

```python
import os
from juddges.extraction import GeminiExtractionChain
from juddges.extraction.gemini_chain import DocumentType, ExtractionSchema

# Initialize the chain
chain = GeminiExtractionChain(
    model_name="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),  # Explicitly pass API key
    cache_path=".cache/extraction.db",
    temperature=0.0,
)

# Define extraction schema
schema = ExtractionSchema(
    fields={
        "verdict_date": "date as ISO 8601, when the verdict was issued",
        "court": "string, name of the court",
        "parties": "List[string], names of involved parties",
        "verdict": "string, text of the verdict",
    },
    instructions="Extract only factual information from the judgment text.",
    language="polish",
)

# Extract information
result = chain.extract(
    document_type=DocumentType.JUDGMENT,
    text="Your judgment text here...",
    schema=schema,
)

print(result)
# {'verdict_date': '2024-01-15', 'court': 'Sąd Okręgowy w Warszawie', ...}
```

### With Langfuse Tracing

```python
import os
from langfuse.langchain import CallbackHandler

# Set Langfuse credentials
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-..."

# Create Langfuse handler
langfuse_handler = CallbackHandler()

# Extract with tracing
result = chain.extract(
    document_type=DocumentType.JUDGMENT,
    text=judgment_text,
    schema=schema,
    langfuse_handler=langfuse_handler,
)
```

### Batch Processing

```python
# Extract from multiple documents
texts = [judgment1, judgment2, judgment3]

results = chain.batch_extract(
    document_type=DocumentType.JUDGMENT,
    texts=texts,
    schema=schema,
    langfuse_handler=langfuse_handler,  # Optional
)

# results is a list of dictionaries
for i, result in enumerate(results):
    print(f"Document {i+1}: {result}")
```

## Schema Definition

The `ExtractionSchema` defines what information to extract and how:

```python
schema = ExtractionSchema(
    fields={
        # Field definitions in format: "field_name": "type, description"
        "field1": "string, description of field1",
        "field2": "date as ISO 8601, when something happened",
        "field3": "List[string], list of items",
        "field4": "boolean, whether condition is true",
    },
    instructions="Additional instructions for extraction process",
    language="polish",  # or "english"
)
```

### Schema Field Types

Supported field types:

- `string` - Text fields
- `date as ISO 8601` - Dates in YYYY-MM-DD format
- `List[string]` - Lists of strings
- `List[int]` - Lists of integers
- `boolean` - True/False values
- `int` - Integer numbers
- `float` - Decimal numbers

### Example Schemas

#### Judgment Schema

```python
judgment_schema = ExtractionSchema(
    fields={
        "verdict_date": "date as ISO 8601, when the verdict was issued",
        "verdict_id": "string, official case identifier",
        "court": "string, name of the court that issued the judgment",
        "judge_names": "List[string], names of judges",
        "parties": "List[string], names of involved parties",
        "legal_basis": "List[string], referenced laws and articles",
        "verdict": "string, text representing the verdict",
        "verdict_summary": "string, concise summary of the verdict",
    },
    instructions=(
        "Focus on extracting factual information only. "
        "For dates, ensure ISO 8601 format. "
        "For lists, include all mentioned items."
    ),
    language="polish",
)
```

#### Tax Interpretation Schema

```python
tax_schema = ExtractionSchema(
    fields={
        "interpretation_date": "date as ISO 8601, when issued",
        "interpretation_number": "string, official document number",
        "tax_authority": "string, issuing tax authority",
        "applicant": "string, who requested the interpretation",
        "subject_matter": "string, brief description of the tax issue",
        "legal_basis": "List[string], referenced tax laws and articles",
        "conclusion": "string, final ruling or conclusion",
    },
    instructions="Extract key legal information and maintain accuracy of legal references.",
    language="polish",
)
```

## Configuration

### Model Selection

Choose between Gemini 2.5 Pro and Flash:

```python
# Pro model - higher quality, slower, more expensive
chain_pro = GeminiExtractionChain(model_name="gemini-2.5-pro")

# Flash model - faster, cheaper, good quality
chain_flash = GeminiExtractionChain(model_name="gemini-2.5-flash")
```

### Caching

Cache is enabled by default with SQLite:

```python
# Default cache location
chain = GeminiExtractionChain()  # Uses .cache/langchain.db

# Custom cache location
chain = GeminiExtractionChain(cache_path="my_cache/extraction.db")

# Disable caching (not recommended)
chain = GeminiExtractionChain(cache_path=None)
```

### Temperature

Control output randomness:

```python
# Deterministic (recommended for extraction)
chain = GeminiExtractionChain(temperature=0.0)

# More creative (not recommended for factual extraction)
chain = GeminiExtractionChain(temperature=0.7)
```

## Example Script

Run the provided example script:

```bash
# Basic usage
python scripts/extraction/extract_with_gemini.py

# With Langfuse tracing
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
python scripts/extraction/extract_with_gemini.py --use-langfuse

# Using Pro model
python scripts/extraction/extract_with_gemini.py --model gemini-2.5-pro

# Tax interpretation
python scripts/extraction/extract_with_gemini.py --document-type tax_interpretation

# Batch processing
python scripts/extraction/extract_with_gemini.py --batch-size 10
```

## API Reference

### GeminiExtractionChain

```python
class GeminiExtractionChain:
    def __init__(
        self,
        model_name: Literal["gemini-2.5-pro", "gemini-2.5-flash"] = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        cache_path: Optional[str | Path] = None,
        max_output_tokens: Optional[int] = 8192,
    ):
        """Initialize Gemini extraction chain."""

    def extract(
        self,
        document_type: DocumentType,
        text: str,
        schema: ExtractionSchema,
        langfuse_handler: Optional[BaseCallbackHandler] = None,
        max_text_length: int = 150000,
    ) -> dict[str, Any]:
        """Extract structured information from single document."""

    def batch_extract(
        self,
        document_type: DocumentType,
        texts: list[str],
        schema: ExtractionSchema,
        langfuse_handler: Optional[BaseCallbackHandler] = None,
        max_text_length: int = 150000,
    ) -> list[dict[str, Any]]:
        """Extract information from multiple documents in batch."""
```

### ExtractionSchema

```python
class ExtractionSchema(BaseModel):
    fields: dict[str, str]  # Field definitions
    instructions: Optional[str] = None  # Additional instructions
    language: str = "polish"  # Extraction language

    def to_schema_string(self) -> str:
        """Convert schema to string format for prompt."""
```

### DocumentType

```python
class DocumentType(str, Enum):
    TAX_INTERPRETATION = "tax_interpretation"
    JUDGMENT = "judgment"
```

## Environment Variables

```bash
# Required for Gemini API
export GOOGLE_API_KEY="your-google-api-key"

# Optional for Langfuse tracing
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"  # Optional, defaults to cloud
```

## Best Practices

### 1. Schema Design

- **Be specific**: Provide clear descriptions for each field
- **Use standard types**: Stick to common types (string, date, List[string], boolean)
- **Add context**: Include why and how to extract in the field description
- **Set language**: Always specify the document language

### 2. Caching

- **Enable caching**: Always use cache for production (default behavior)
- **Shared cache**: Use same cache path for related extractions
- **Monitor size**: Check cache size periodically, clean if too large

### 3. Error Handling

```python
from loguru import logger

try:
    result = chain.extract(
        document_type=DocumentType.JUDGMENT,
        text=text,
        schema=schema,
    )
except Exception as e:
    logger.error(f"Extraction failed: {e}")
    # Handle error appropriately
```

### 4. Performance

- **Use Flash model**: For most cases, `gemini-2.5-flash` is sufficient
- **Batch processing**: Use `batch_extract()` for multiple documents
- **Text length**: Documents are auto-truncated to 150k chars (adjust if needed)
- **Cache hits**: Identical inputs return cached results instantly

### 5. Langfuse Integration

```python
# Set up once at application start
langfuse_handler = CallbackHandler(
    trace_name="judgment-extraction",
    metadata={"environment": "production"},
)

# Reuse across extractions
for text in texts:
    result = chain.extract(..., langfuse_handler=langfuse_handler)
```

## Comparison with Existing Extraction

### Old Approach (juddges/prompts/information_extraction.py)

```python
from juddges.prompts.information_extraction import prepare_information_extraction_chain

# Uses OpenAI GPT-4
chain = prepare_information_extraction_chain(model_name="gpt-4-0125-preview")
result = chain.invoke({"TEXT": text, "SCHEMA": schema, "LANGUAGE": "polish"})
```

### New Gemini Approach

```python
from juddges.extraction import GeminiExtractionChain

# Uses Google Gemini
chain = GeminiExtractionChain(model_name="gemini-2.5-flash")
result = chain.extract(
    document_type=DocumentType.JUDGMENT,
    text=text,
    schema=schema,
)
```

### Key Differences

| Feature | Old (OpenAI) | New (Gemini) |
|---------|--------------|--------------|
| Model | GPT-4 | Gemini 2.5 Pro/Flash |
| Caching | SQLAlchemy/Postgres | SQLite (simpler) |
| Observability | MLflow | Langfuse |
| Document Types | Generic | Judgment/Tax Interpretation |
| Schema Format | String | ExtractionSchema (Pydantic) |
| Type Safety | Limited | Full Pydantic validation |

## Troubleshooting

### "API key not found"

```bash
export GOOGLE_API_KEY="your-api-key"
```

Get your API key from: https://ai.google.dev/gemini-api/docs/api-key

### "Langfuse keys not set"

```bash
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
```

### Cache permission errors

```python
# Ensure cache directory is writable
from pathlib import Path
cache_dir = Path(".cache")
cache_dir.mkdir(parents=True, exist_ok=True)
```

### Low extraction quality

1. **Use more specific schema**: Add detailed field descriptions
2. **Add instructions**: Provide clear extraction guidelines
3. **Try Pro model**: `model_name="gemini-2.5-pro"`
4. **Validate output**: Check if document contains requested information

## Performance Metrics

Typical extraction times (approximate):

| Model | Document Size | First Call | Cached Call |
|-------|---------------|------------|-------------|
| Flash | 5,000 tokens | 2-3s | <0.1s |
| Flash | 20,000 tokens | 5-8s | <0.1s |
| Pro | 5,000 tokens | 4-6s | <0.1s |
| Pro | 20,000 tokens | 10-15s | <0.1s |

_Cache hits return instantly, making repeated extractions extremely fast._

## Contributing

To extend the extraction chain:

1. Add new document types to `DocumentType` enum
2. Update `_build_extraction_prompt()` with new prompts
3. Add example schemas in documentation
4. Test with representative documents

## Related Documentation

- [LangChain Gemini Integration](https://python.langchain.com/docs/integrations/chat/google_generative_ai/)
- [Langfuse Documentation](https://langfuse.com/docs)
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Project Overview](./PROJECT_OVERVIEW.md)

## Support

For issues or questions:

- Check existing code in `juddges/extraction/`
- Review example script in `scripts/extraction/extract_with_gemini.py`
- Open an issue on the project repository
