# Gemini Extraction Chain

LangChain-based information extraction using Google Gemini 2.5 Pro/Flash with caching and observability.

## Overview

The `juddges.extraction.gemini_chain` module provides a production-ready extraction pipeline using Google's Gemini 2.5 models. It's designed for structured information extraction from legal documents with:

- Caching to reduce API costs
- Langfuse integration for observability
- Schema-driven extraction
- Batch processing support
- Automatic error handling

## Key Features

- **Multiple Models**: Gemini 2.5 Pro and Flash support
- **SQLite Caching**: Avoid redundant API calls (cost savings)
- **Langfuse Observability**: Track extraction runs, costs, and performance
- **Structured Output**: Parse JSON responses to dictionaries
- **Document Type Aware**: Optimized prompts for judgments vs tax interpretations
- **Batch Extraction**: Process multiple documents efficiently
- **Automatic Truncation**: Handle documents exceeding token limits

## Usage Examples

### Basic Extraction

```python
from juddges.extraction.gemini_chain import (
    GeminiExtractionChain,
    ExtractionSchema,
    DocumentType
)

# Initialize chain
chain = GeminiExtractionChain(
    model_name="gemini-2.5-flash",
    temperature=0.0,
    cache_path="cache/extraction.db"
)

# Define extraction schema
schema = ExtractionSchema(
    fields={
        "verdict_date": "date as ISO 8601, when the verdict was issued",
        "court": "string, name of the court",
        "case_number": "string, case identifier",
        "parties": "List[string], names of involved parties"
    },
    instructions="Focus on extracting factual information only.",
    language="polish"
)

# Extract from judgment
result = chain.extract(
    document_type=DocumentType.JUDGMENT,
    text="Sąd Okręgowy w Warszawie dnia 15 stycznia 2024...",
    schema=schema
)

print(result)
# {
#     "verdict_date": "2024-01-15",
#     "court": "Sąd Okręgowy w Warszawie",
#     "case_number": "...",
#     "parties": ["Jan Kowalski", "XYZ Bank"]
# }
```

### Extraction with Langfuse Observability

```python
from langfuse.callback import CallbackHandler

# Initialize Langfuse handler
langfuse_handler = CallbackHandler(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"
)

# Extract with tracing
result = chain.extract(
    document_type=DocumentType.JUDGMENT,
    text=judgment_text,
    schema=schema,
    langfuse_handler=langfuse_handler
)

# View trace in Langfuse dashboard
```

### Batch Extraction

```python
# Extract from multiple documents
texts = [judgment1, judgment2, judgment3]

results = chain.batch_extract(
    document_type=DocumentType.JUDGMENT,
    texts=texts,
    schema=schema,
    langfuse_handler=langfuse_handler
)

# Process results
for i, result in enumerate(results):
    print(f"Document {i}: {result}")
```

### Tax Interpretation Extraction

```python
# Different document type with specialized prompt
schema = ExtractionSchema(
    fields={
        "interpretation_date": "date as ISO 8601",
        "tax_authority": "string, issuing tax authority",
        "taxpayer": "string, name of taxpayer",
        "interpretation_subject": "string, subject of interpretation"
    },
    language="polish"
)

result = chain.extract(
    document_type=DocumentType.TAX_INTERPRETATION,
    text=tax_interpretation_text,
    schema=schema
)
```

## API Reference

::: juddges.extraction.gemini_chain.GeminiExtractionChain
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: juddges.extraction.gemini_chain.ExtractionSchema
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: juddges.extraction.gemini_chain.DocumentType
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## Schema Design Best Practices

### Field Definitions

Be explicit and specific in field descriptions:

```python
# Good: Specific type and format
"verdict_date": "date as ISO 8601 (YYYY-MM-DD), when the verdict was issued"

# Bad: Vague description
"verdict_date": "the date"
```

### Enum Fields

Provide explicit choices:

```python
schema = ExtractionSchema(
    fields={
        "judgment_type": "enum: one of [Wyrok, Postanowienie, Uchwała]",
        "finality": "enum: one of [Prawomocne, Nieprawomocne]"
    }
)
```

### List Fields

Specify list format clearly:

```python
schema = ExtractionSchema(
    fields={
        "parties": "List[string], names of all parties involved in the case",
        "legal_bases": "List[string], legal bases cited (e.g., 'Art. 123 KC')"
    }
)
```

### Boolean Fields

Use clear true/false criteria:

```python
schema = ExtractionSchema(
    fields={
        "appeal_allowed": "boolean, true if appeal is explicitly allowed, false otherwise"
    },
    instructions="Only mark boolean fields as true when explicitly confirmed."
)
```

## Caching

### How Caching Works

The chain uses SQLite caching to store API responses:

```python
chain = GeminiExtractionChain(
    cache_path="cache/extraction.db"  # SQLite database file
)

# First call: Makes API request
result1 = chain.extract(...)  # API call

# Second call with same input: Returns cached result
result2 = chain.extract(...)  # No API call (cached)
```

### Cache Benefits

- **Cost Reduction**: Avoid repeated API charges
- **Speed**: Instant responses for cached queries
- **Reliability**: Work offline with previously cached data

### Cache Location

Default cache: `.cache/langchain.db`

Custom cache:

```python
chain = GeminiExtractionChain(
    cache_path="my_cache/extraction.db"
)
```

## Langfuse Integration

### Setup Langfuse

```python
from langfuse.callback import CallbackHandler

handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host="https://cloud.langfuse.com"
)
```

### Tracked Metrics

Langfuse tracks:

- **Traces**: Full extraction pipelines
- **Latency**: Response times
- **Token Usage**: Input/output tokens
- **Costs**: API costs per extraction
- **Errors**: Failed extractions

### Viewing Results

Access Langfuse dashboard:

```
https://cloud.langfuse.com
```

Filter by:

- Document type
- Date range
- Success/failure status
- Cost thresholds

## Error Handling

```python
try:
    result = chain.extract(
        document_type=DocumentType.JUDGMENT,
        text=text,
        schema=schema
    )
except ValueError as e:
    # Invalid document type or schema
    print(f"Validation error: {e}")
except Exception as e:
    # API errors, parsing errors, etc.
    print(f"Extraction failed: {e}")
    # Check Langfuse for detailed trace
```

## Model Selection

### Gemini 2.5 Flash (Recommended)

```python
chain = GeminiExtractionChain(
    model_name="gemini-2.5-flash"
)
```

**Pros**:

- Faster responses
- Lower cost
- Good accuracy for structured tasks

**Cons**:

- Slightly lower accuracy on complex cases

### Gemini 2.5 Pro

```python
chain = GeminiExtractionChain(
    model_name="gemini-2.5-pro"
)
```

**Pros**:

- Highest accuracy
- Better on complex documents
- More reliable enum classification

**Cons**:

- Higher cost
- Slower responses

## Performance Optimization

### Batch Processing

Process multiple documents in one API call:

```python
# More efficient than individual extractions
results = chain.batch_extract(
    document_type=DocumentType.JUDGMENT,
    texts=texts,  # List of 10-100 documents
    schema=schema
)
```

### Text Truncation

Long documents are automatically truncated:

```python
result = chain.extract(
    document_type=DocumentType.JUDGMENT,
    text=very_long_text,
    schema=schema,
    max_text_length=150000  # Truncate at 150k chars
)
```

### Temperature Control

Use temperature=0.0 for deterministic extraction:

```python
chain = GeminiExtractionChain(
    temperature=0.0  # Deterministic outputs
)
```

## Related

- [Evaluation Metrics](../evals/metrics.md) - Evaluate extraction quality
- [Gemini Tutorial](../../../tutorials/GEMINI_EXTRACTION.md) - Complete tutorial
- [Langfuse Setup](../../../tutorials/LANGFUSE_SETUP.md) - Observability setup

## Common Patterns

### Production Extraction Pipeline

```python
import os
from juddges.extraction.gemini_chain import (
    GeminiExtractionChain,
    ExtractionSchema,
    DocumentType
)
from langfuse.callback import CallbackHandler

# Initialize components
chain = GeminiExtractionChain(
    model_name="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    cache_path="cache/production.db"
)

langfuse = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY")
)

# Load schema
schema = ExtractionSchema.from_file("schemas/judgment_schema.yaml")

# Extract with monitoring
results = []
for doc in documents:
    try:
        result = chain.extract(
            document_type=DocumentType.JUDGMENT,
            text=doc["text"],
            schema=schema,
            langfuse_handler=langfuse
        )
        results.append(result)
    except Exception as e:
        logger.error(f"Failed on doc {doc['id']}: {e}")
        continue
```

### Cost-Optimized Extraction

```python
# Use Flash model for bulk extraction
chain = GeminiExtractionChain(
    model_name="gemini-2.5-flash",  # Lower cost
    cache_path="cache/bulk.db",      # Enable caching
    max_output_tokens=4096           # Limit token usage
)

# Batch process for efficiency
results = chain.batch_extract(
    document_type=DocumentType.JUDGMENT,
    texts=texts,
    schema=schema
)
```
