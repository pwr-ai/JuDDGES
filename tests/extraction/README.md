# Extraction Tests

This directory contains tests for the Gemini extraction chain.

## Test Structure

```
tests/extraction/
├── test_gemini_chain.py          # Unit tests (mocked, no API calls)
├── test_gemini_integration.py    # Integration tests (real API calls)
└── README.md                      # This file

scripts/extraction/
└── test_extraction_manual.py     # Interactive manual test script
```

## Prerequisites

### 1. Install Dependencies

```bash
# Install test dependencies
uv pip install -e ".[full,dev]"
```

### 2. Set Up API Key

Get your Google API key from: https://ai.google.dev/gemini-api/docs/api-key

```bash
export GOOGLE_API_KEY="your-google-api-key-here"
```

### 3. Optional: Langfuse (for observability tests)

```bash
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
```

## Running Tests

### Option 1: Unit Tests (Recommended First)

Unit tests use mocks and don't make real API calls. They're fast and free.

```bash
# Run all unit tests
pytest tests/extraction/test_gemini_chain.py -v

# Run specific test
pytest tests/extraction/test_gemini_chain.py::TestExtractionSchema::test_schema_creation -v

# Run with coverage
pytest tests/extraction/test_gemini_chain.py --cov=juddges.extraction --cov-report=html
```

**Expected output:**
```
tests/extraction/test_gemini_chain.py::TestExtractionSchema::test_schema_creation PASSED
tests/extraction/test_gemini_chain.py::TestExtractionSchema::test_schema_to_string PASSED
tests/extraction/test_gemini_chain.py::TestGeminiExtractionChain::test_chain_initialization PASSED
...
==================== 15 passed in 2.34s ====================
```

### Option 2: Integration Tests (Requires API Key)

Integration tests make real API calls to Gemini. **This will use your API quota.**

```bash
# Run integration tests (requires GOOGLE_API_KEY)
pytest tests/extraction/test_gemini_integration.py -v -s

# Skip tests if API key not set
pytest tests/extraction/test_gemini_integration.py -v
# (automatically skips if GOOGLE_API_KEY not set)

# Run single integration test
pytest tests/extraction/test_gemini_integration.py::test_extract_judgment_with_real_api -v -s
```

**Note:** Integration tests are marked with `@pytest.mark.integration` and automatically skip if API keys are not set.

### Option 3: Manual Interactive Test (Recommended for First Run)

The manual test script is the easiest way to test interactively:

```bash
# Set API key first
export GOOGLE_API_KEY="your-key"

# Run interactive tests
python scripts/extraction/test_extraction_manual.py
```

This will:
- ✅ Check if your API key is set
- ✅ Run basic extraction test
- ✅ Test caching performance
- ✅ Test batch extraction
- ✅ Test with real judgment data (if available)
- ✅ Show results in rich console format

**Example output:**
```
╭──────────────────────────────────────────────╮
│ Gemini Extraction Chain - Manual Test Suite │
│                                              │
│ This script tests the extraction chain      │
│ with real API calls.                        │
╰──────────────────────────────────────────────╯

✓ GOOGLE_API_KEY found

Test 1: Basic Judgment Extraction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Judgment text:
┌──────────────────────────────────┐
│ WYROK                            │
│ W IMIENIU RZECZYPOSPOLITEJ...    │
└──────────────────────────────────┘

Creating extraction chain...
Schema fields:
  • verdict_date: date as ISO 8601
  • court: string, court name
  ...

Calling Gemini API for extraction...

✓ Extraction successful!

Extracted Information:
{
  "verdict_date": "2024-01-15",
  "court": "Sąd Okręgowy w Warszawie",
  ...
}
```

### Option 4: Run All Tests

```bash
# Run all tests (unit + integration, if keys are set)
pytest tests/extraction/ -v

# Run only unit tests (skip integration)
pytest tests/extraction/ -v -m "not integration"

# Run with verbose output
pytest tests/extraction/ -v -s
```

## Test Categories

### Unit Tests (`test_gemini_chain.py`)

Tests that use mocks and don't require API access:

- ✅ Schema creation and validation
- ✅ Chain initialization
- ✅ Prompt building
- ✅ Text truncation
- ✅ Batch processing logic
- ✅ Langfuse handler integration

**Cost:** Free (no API calls)
**Speed:** Fast (~2 seconds)

### Integration Tests (`test_gemini_integration.py`)

Tests that make real API calls:

- 🌐 Real extraction from judgment text
- 🌐 Caching performance verification
- 🌐 Batch extraction
- 🌐 Langfuse tracing (if configured)
- 🌐 Tax interpretation extraction

**Cost:** Uses API quota (approximately $0.001-0.01 per test)
**Speed:** Slower (~30 seconds)

**Note:** Integration tests automatically skip if `GOOGLE_API_KEY` is not set.

## Example Workflows

### First Time Setup

```bash
# 1. Get API key from https://ai.google.dev/gemini-api/docs/api-key
export GOOGLE_API_KEY="your-key"

# 2. Run manual test to verify everything works
python scripts/extraction/test_extraction_manual.py

# 3. If successful, run unit tests
pytest tests/extraction/test_gemini_chain.py -v

# 4. Run one integration test
pytest tests/extraction/test_gemini_integration.py::test_extract_judgment_with_real_api -v -s
```

### During Development

```bash
# Quick check with unit tests (no API cost)
pytest tests/extraction/test_gemini_chain.py -v

# Test specific functionality
pytest tests/extraction/test_gemini_chain.py::TestExtractionSchema -v

# Full test with integration
pytest tests/extraction/ -v
```

### Before Commit

```bash
# Run all unit tests
pytest tests/extraction/test_gemini_chain.py -v --cov=juddges.extraction

# Optionally run integration tests
pytest tests/extraction/test_gemini_integration.py -v
```

## Troubleshooting

### "API key not found"

```bash
export GOOGLE_API_KEY="your-key"
# Verify it's set
echo $GOOGLE_API_KEY
```

### "Module not found: juddges.extraction"

Install the package in development mode:
```bash
uv pip install -e .
```

### "Langfuse keys not set" (for Langfuse tests)

```bash
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
```

### Integration tests are slow

Integration tests make real API calls. To skip them:
```bash
pytest tests/extraction/ -v -m "not integration"
```

### Cache permission errors

```bash
# Ensure cache directory exists and is writable
mkdir -p .cache
chmod 755 .cache
```

## Cost Estimates

Gemini 2.5 Flash pricing (approximate as of 2025):
- Input: $0.00001875 per 1K tokens
- Output: $0.000075 per 1K tokens

Typical extraction:
- Input: ~2K tokens (judgment + schema + prompt)
- Output: ~200 tokens (extracted JSON)
- **Cost per extraction: ~$0.00005 (less than $0.0001)**

Running all integration tests (~10 extractions):
- **Total cost: ~$0.0005-0.001**

**Caching drastically reduces costs** - identical extractions are free after the first call.

## CI/CD Integration

For GitHub Actions or other CI:

```yaml
# .github/workflows/test.yml
- name: Run extraction tests
  env:
    GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
  run: |
    # Run only unit tests in CI (no API costs)
    pytest tests/extraction/test_gemini_chain.py -v

    # Optionally run integration tests (if API key is set)
    # pytest tests/extraction/test_gemini_integration.py -v
```

## Writing New Tests

### Unit Test Example

```python
# tests/extraction/test_gemini_chain.py

@patch("juddges.extraction.gemini_chain.ChatGoogleGenerativeAI")
def test_my_feature(mock_llm_class):
    """Test description."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content='{"result": "value"}')
    mock_llm_class.return_value = mock_llm

    chain = GeminiExtractionChain(cache_path=None)
    # ... test your feature
```

### Integration Test Example

```python
# tests/extraction/test_gemini_integration.py

@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="API key not set")
def test_my_integration(chain, judgment_schema):
    """Integration test description."""
    result = chain.extract(
        document_type=DocumentType.JUDGMENT,
        text="test judgment",
        schema=judgment_schema,
    )
    assert isinstance(result, dict)
```

## Questions?

See the main documentation:
- [GEMINI_EXTRACTION.md](../../docs/GEMINI_EXTRACTION.md) - Full extraction guide
- [extraction/README.md](../../juddges/extraction/README.md) - Module overview
