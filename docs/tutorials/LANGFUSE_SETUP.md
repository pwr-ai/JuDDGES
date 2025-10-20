# Langfuse Integration Setup

Guide for setting up Langfuse observability with the Gemini extraction chain.

## What is Langfuse?

Langfuse is an open-source LLM observability platform that helps you:
- 📊 Track all LLM calls and their performance
- 💰 Monitor costs and token usage
- 🐛 Debug prompts and responses
- 📈 Analyze trends and patterns
- 🔍 Search and filter traces
- 👥 Track user sessions

## Quick Start

### 1. Create Langfuse Account

**Option A: Cloud (Recommended)**
```bash
# Sign up at https://cloud.langfuse.com
# Free tier includes:
# - 50,000 observations/month
# - All features unlocked
# - No credit card required
```

**Option B: Self-Hosted**
```bash
# See: https://langfuse.com/docs/deployment/self-host
docker pull langfuse/langfuse
```

### 2. Get API Keys

1. Log in to Langfuse
2. Create a project (or use default)
3. Go to **Settings → API Keys**
4. Click **Create New Key**
5. Copy both keys:
   - `LANGFUSE_PUBLIC_KEY` (starts with `pk-lf-`)
   - `LANGFUSE_SECRET_KEY` (starts with `sk-lf-`)

### 3. Set Environment Variables

```bash
# Required
export GOOGLE_API_KEY="your-google-api-key"
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."

# Optional (defaults to cloud.langfuse.com)
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

### 4. Install Dependencies

```bash
# Langfuse is already in pyproject.toml
uv pip install -e ".[full]"
```

### 5. Test Integration

```bash
python scripts/extraction/test_langfuse.py
```

## Usage Examples

### Basic Usage

```python
from langfuse.langchain import CallbackHandler
from juddges.extraction import GeminiExtractionChain
from juddges.extraction.gemini_chain import DocumentType, ExtractionSchema

# Create Langfuse handler
langfuse_handler = CallbackHandler(
    trace_name="judgment_extraction",
    session_id="user_123",
    user_id="john_doe",
)

# Create extraction chain
chain = GeminiExtractionChain(model_name="gemini-2.5-flash")

# Define schema
schema = ExtractionSchema(
    fields={"verdict_date": "date as ISO 8601", "court": "string"},
    language="polish",
)

# Extract with tracing
result = chain.extract(
    document_type=DocumentType.JUDGMENT,
    text="Wyrok Sądu...",
    schema=schema,
    langfuse_handler=langfuse_handler,  # Pass handler here
)

# Flush to ensure trace is sent
langfuse_handler.langfuse.flush()
```

### Session Tracking

Track multiple extractions in a single session:

```python
from langfuse.langchain import CallbackHandler

session_id = "batch_extraction_20240115"

for i, judgment_text in enumerate(judgments):
    # Create handler for each extraction
    handler = CallbackHandler(
        trace_name=f"extract_{i}",
        session_id=session_id,  # Same session
        metadata={"judgment_index": i},
    )

    result = chain.extract(
        document_type=DocumentType.JUDGMENT,
        text=judgment_text,
        schema=schema,
        langfuse_handler=handler,
    )

    handler.langfuse.flush()
```

### Adding Metadata and Tags

Enrich traces with context:

```python
handler = CallbackHandler(
    trace_name="production_extraction",
    session_id=session_id,
    user_id=user_id,
    metadata={
        "environment": "production",
        "model": "gemini-2.5-flash",
        "document_type": "judgment",
        "case_number": case_num,
        "batch_id": batch_id,
    },
    tags=["production", "judgment", "important"],
)

result = chain.extract(..., langfuse_handler=handler)
```

### Cost Tracking

Langfuse automatically tracks token usage and costs:

```python
# After extraction, view in Langfuse dashboard:
# - Total tokens used
# - Input/output token breakdown
# - Estimated cost per extraction
# - Cost trends over time
```

## Langfuse Dashboard

### Viewing Traces

1. **Navigate to Traces**
   - Go to your project dashboard
   - Click "Traces" in sidebar

2. **Find Your Traces**
   - Use filters: session, user, tags
   - Search by trace name
   - Sort by timestamp, duration, cost

3. **Inspect a Trace**
   - Click on any trace to see:
     - Full prompt sent to Gemini
     - Model response
     - Token usage
     - Execution time
     - Metadata
     - Tags

### Useful Filters

```
# Filter by production extractions
tag: production

# Filter by specific session
session_id: batch_extraction_20240115

# Filter by user
user_id: john_doe

# Filter by model
metadata.model: gemini-2.5-flash

# Filter by date range
created_at: 2024-01-15 to 2024-01-16
```

### Analytics

Langfuse provides built-in analytics:

- **Latency**: p50, p95, p99 response times
- **Cost**: Total and per-trace costs
- **Volume**: Number of traces over time
- **Errors**: Error rate and types
- **Token Usage**: Input/output token trends

### Sessions View

See all extractions in a session:

1. Go to "Sessions"
2. Click on session ID
3. View timeline of all traces
4. Analyze session patterns

## Advanced Features

### 1. Scoring Traces

Rate extraction quality:

```python
from langfuse import Langfuse

langfuse = Langfuse()

# After extraction, score the result
langfuse.score(
    trace_id=trace_id,
    name="extraction_quality",
    value=0.95,  # 0-1 scale
    comment="High quality extraction",
)
```

### 2. Generations vs Spans

Langfuse automatically creates:
- **Generation**: LLM call (Gemini API call)
- **Span**: Chain execution (entire extraction)

View hierarchy in dashboard:
```
Trace: judgment_extraction
  └─ Span: extraction_chain
      └─ Generation: gemini_call
```

### 3. Feedback Loop

Collect user feedback on extractions:

```python
langfuse.score(
    trace_id=trace_id,
    name="user_feedback",
    value=1,  # 1 = helpful, 0 = not helpful
    comment=user_feedback_text,
)
```

### 4. Datasets

Create evaluation datasets:

```python
# Add extraction to dataset
langfuse.create_dataset_item(
    dataset_name="judgment_extractions",
    input={"text": judgment_text, "schema": schema_dict},
    expected_output=gold_standard_result,
)
```

### 5. Experiments

Track different extraction approaches:

```python
handler = CallbackHandler(
    trace_name="experiment_gemini_pro",
    metadata={
        "experiment": "model_comparison",
        "variant": "gemini-2.5-pro",
    },
)

# Compare with Flash model
handler_flash = CallbackHandler(
    trace_name="experiment_gemini_flash",
    metadata={
        "experiment": "model_comparison",
        "variant": "gemini-2.5-flash",
    },
)

# Compare results in Langfuse dashboard
```

## Best Practices

### 1. Consistent Naming

Use clear, consistent trace names:

```python
# Good
trace_name = "judgment_extraction"
trace_name = "tax_interpretation_extraction"
trace_name = "batch_extract_judgments"

# Bad
trace_name = "test"
trace_name = "extraction1"
trace_name = "x"
```

### 2. Session Management

Group related extractions:

```python
# One session per user request
session_id = f"user_{user_id}_{timestamp}"

# One session per batch job
session_id = f"batch_{batch_id}"

# One session per API endpoint call
session_id = f"api_{endpoint}_{request_id}"
```

### 3. Metadata Standards

Use consistent metadata keys:

```python
metadata = {
    "environment": "production",  # or "staging", "development"
    "model": "gemini-2.5-flash",
    "document_type": "judgment",
    "batch_id": batch_id,
    "user_id": user_id,
    "tenant_id": tenant_id,
}
```

### 4. Tag Strategy

Use tags for filtering:

```python
tags = [
    "production",      # Environment
    "judgment",        # Document type
    "high-priority",   # Business priority
    "experiment-v2",   # Experiment version
]
```

### 5. Flush Regularly

Ensure traces are sent:

```python
# After each extraction
handler.langfuse.flush()

# Or at the end of batch
for handler in handlers:
    handler.langfuse.flush()
```

## Troubleshooting

### Connection Issues

```python
# Test connection
from langfuse import Langfuse

langfuse = Langfuse()
trace = langfuse.trace(name="test")
print(f"Trace ID: {trace.id}")
langfuse.flush()
```

### Missing Traces

**Possible causes:**
1. Forgot to call `flush()`
2. Network issues
3. Wrong API keys
4. Wrong host URL

**Solution:**
```python
# Always flush after extraction
handler.langfuse.flush()

# Check environment variables
import os
print(os.getenv("LANGFUSE_PUBLIC_KEY"))
print(os.getenv("LANGFUSE_HOST"))
```

### Authentication Errors

```bash
# Verify keys are correct
echo $LANGFUSE_PUBLIC_KEY  # Should start with pk-lf-
echo $LANGFUSE_SECRET_KEY  # Should start with sk-lf-

# Check for whitespace
export LANGFUSE_PUBLIC_KEY=$(echo $LANGFUSE_PUBLIC_KEY | tr -d ' ')
export LANGFUSE_SECRET_KEY=$(echo $LANGFUSE_SECRET_KEY | tr -d ' ')
```

### Self-Hosted Issues

```bash
# Check host URL includes protocol
export LANGFUSE_HOST="https://your-domain.com"  # ✓ Correct
export LANGFUSE_HOST="your-domain.com"          # ✗ Wrong

# Test connectivity
curl $LANGFUSE_HOST/api/public/health
```

## Cost and Limits

### Langfuse Cloud Free Tier

- **50,000 observations/month**
- All features included
- No credit card required
- Perfect for development and testing

**What's an observation?**
- Each LLM call = 1 observation
- Each span = 1 observation
- Typical extraction = 2-3 observations

**Monthly usage estimate:**
```
10,000 extractions × 3 observations = 30,000 observations
Still within free tier!
```

### Paid Plans

- **Team**: $59/month - 200K observations
- **Pro**: Custom pricing - Unlimited

See: https://langfuse.com/pricing

## Resources

- **Documentation**: https://langfuse.com/docs
- **LangChain Integration**: https://langfuse.com/docs/integrations/langchain
- **API Reference**: https://langfuse.com/docs/api
- **GitHub**: https://github.com/langfuse/langfuse
- **Discord**: https://langfuse.com/discord

## Example Workflow

Complete workflow with Langfuse:

```python
#!/usr/bin/env python
"""Production extraction with full Langfuse observability."""

import os
from langfuse.langchain import CallbackHandler
from juddges.extraction import GeminiExtractionChain
from juddges.extraction.gemini_chain import DocumentType, ExtractionSchema

def extract_with_observability(judgment_text: str, user_id: str) -> dict:
    """Extract with full Langfuse tracking."""

    # Create handler
    handler = CallbackHandler(
        trace_name="production_judgment_extraction",
        session_id=f"user_{user_id}",
        user_id=user_id,
        metadata={
            "environment": "production",
            "model": "gemini-2.5-flash",
        },
        tags=["production", "judgment"],
    )

    # Extract
    chain = GeminiExtractionChain(model_name="gemini-2.5-flash")
    schema = ExtractionSchema(
        fields={
            "verdict_date": "date as ISO 8601",
            "court": "string, court name",
            "verdict": "string, verdict text",
        },
        language="polish",
    )

    try:
        result = chain.extract(
            document_type=DocumentType.JUDGMENT,
            text=judgment_text,
            schema=schema,
            langfuse_handler=handler,
        )

        # Score quality
        handler.langfuse.score(
            trace_id=handler.trace.id,
            name="extraction_success",
            value=1,
        )

    except Exception as e:
        # Log error
        handler.langfuse.score(
            trace_id=handler.trace.id,
            name="extraction_success",
            value=0,
            comment=str(e),
        )
        raise

    finally:
        # Always flush
        handler.langfuse.flush()

    return result

# Use it
result = extract_with_observability(judgment_text, user_id="user_123")
```

## Next Steps

1. ✅ Run test script: `python scripts/extraction/test_langfuse.py`
2. ✅ View traces in dashboard
3. ✅ Set up alerts for errors
4. ✅ Create evaluation datasets
5. ✅ Track cost trends
6. ✅ Monitor performance metrics
