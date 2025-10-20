# How to Test Code Examples in Documentation

This guide shows you how to test Python code examples in the JuDDGES documentation to ensure they remain accurate and functional.

## Problem

Code examples in documentation can become outdated as the codebase evolves, leading to:

- Syntax errors
- Import errors (modules renamed or moved)
- Deprecated APIs
- Broken examples that confuse users

## Solution

JuDDGES uses an automated code example testing system that:

1. Extracts Python code blocks from markdown files
2. Validates syntax and imports
3. Runs tests in isolated environments
4. Reports failures with file locations
5. Integrates with CI/CD pipelines

## Prerequisites

- Python 3.11+
- JuDDGES installed in development mode: `pip install -e .`
- Dependencies: `pyyaml`, `rich`, `loguru` (included in base install)

## Quick Start

### Test All Documentation

```bash
# Run standalone validator
python scripts/docs/test_code_examples.py

# Or use pytest
pytest tests/docs/test_documentation_examples.py -v
```

### Test Specific File

```bash
python scripts/docs/test_code_examples.py docs/reference/api/preprocessing/text_chunker.md
```

### Show Detailed Output

```bash
python scripts/docs/test_code_examples.py --verbose
```

### Generate Report

```bash
python scripts/docs/test_code_examples.py --report code-examples-report.json
```

## Writing Testable Examples

### Complete Examples

Write self-contained examples that can be executed:

```python
from juddges.preprocessing.text_chunker import TextChunker

# Initialize with all required parameters
chunker = TextChunker(
    id_col="judgment_id",
    text_col="full_text",
    chunk_size=512,
    chunk_overlap=50
)

# Use sample data
dataset = {
    "judgment_id": ["doc1", "doc2"],
    "full_text": ["Sample text...", "More text..."]
}

# Execute operation
result = chunker(dataset)
```

### Skip Non-Executable Examples

For demonstration code that can't be tested:

```python
# doctest: +SKIP
# This requires a live production database

import weaviate
client = weaviate.Client("http://production-db:8080")
results = client.query.get(...)
```

### Mark External Dependencies

For code requiring external services:

```python
# requires: weaviate
# This will use a mock Weaviate client during tests

from juddges.data import BaseWeaviateDatabase

db = BaseWeaviateDatabase(url="http://localhost:8080")
collections = db.list_collections()
```

## Available Annotations

Use these annotations in your code examples:

### Skip Markers

- `# doctest: +SKIP` - Skip this entire example
- `# doctest: SKIP` - Alternative skip syntax
- `# skip-test` - Simple skip marker
- `# demonstration-only` - Mark as demonstration code
- `# demo` - Short form for demonstration

### Requirement Markers

- `# requires: weaviate` - Requires Weaviate (uses mock)
- `# requires: gemini` - Requires Gemini API (uses mock)
- `# requires: gpu` - Requires GPU
- `# requires: network` - Requires network access

### Expected Output

```python
result = compute_value()
print(result)
# expected-output: 42
```

## Common Patterns

### Pattern 1: Import and Initialize

```python
from juddges.preprocessing.text_chunker import TextChunker

chunker = TextChunker(
    id_col="judgment_id",
    text_col="full_text",
    chunk_size=512
)
```

### Pattern 2: Complete Workflow

```python
from juddges.preprocessing.text_chunker import TextChunker

# Setup
chunker = TextChunker(
    id_col="judgment_id",
    text_col="full_text",
    chunk_size=512,
    chunk_overlap=50
)

# Sample data
dataset = {
    "judgment_id": ["doc1"],
    "full_text": ["This is a sample legal document. " * 100]
}

# Execute
chunked = chunker(dataset)

# Verify
assert "chunk_text" in chunked
assert len(chunked["judgment_id"]) > 0
```

### Pattern 3: External Service (with Mock)

```python
# requires: weaviate
from juddges.data import BaseWeaviateDatabase

# This uses a mock during tests
db = BaseWeaviateDatabase(url="http://localhost:8080")
collections = db.list_collections()
```

### Pattern 4: Demonstration Only

```python
# doctest: +SKIP
# Production example - not testable

from juddges.models import ModelFactory

# This requires 40GB VRAM
model = ModelFactory.create(
    model_name="meta-llama/Llama-3.1-70B-Instruct",
    device="cuda"
)

predictions = model.predict(large_dataset)
```

## Local Testing Workflow

### Before Committing

1. **Write your documentation** with code examples

2. **Test locally**:

   ```bash
   python scripts/docs/test_code_examples.py docs/your-new-file.md
   ```

3. **Fix any failures**:
   - Add skip markers if needed
   - Ensure imports are correct
   - Verify syntax is valid

4. **Run pre-commit hooks**:

   ```bash
   pre-commit run --all-files
   ```

5. **Commit and push**:

   ```bash
   git add docs/your-new-file.md
   git commit -m "docs: add new guide"
   git push
   ```

### CI/CD Validation

When you create a PR:

1. GitHub Actions automatically runs code example tests
2. Check the **Code Examples** job in PR checks
3. If failures occur:
   - Click on the job to see details
   - Download the `code-examples-report.json` artifact
   - Fix issues and push again

## Troubleshooting

### Issue: Import Error

**Problem**: Test fails with `ModuleNotFoundError`

**Solution**: Ensure module exists and import path is correct

```python
# Bad: Module doesn't exist
from juddges.old_module import something

# Good: Correct import path
from juddges.preprocessing.text_chunker import TextChunker
```

### Issue: Undefined Variables

**Problem**: Test reports undefined variables

**Solution**: Define all variables or add skip marker

```python
# Bad: config undefined
chunker = TextChunker(config)

# Good: config defined
config = {
    "id_col": "judgment_id",
    "text_col": "full_text",
    "chunk_size": 512
}
chunker = TextChunker(**config)
```

### Issue: All Tests Skipped

**Problem**: No tests run, all skipped

**Solution**: Check for skip markers or placeholders

```python
# This will be skipped (placeholder)
api_key = "YOUR_API_KEY"

# Remove placeholder or add explicit skip
# doctest: +SKIP
api_key = "YOUR_API_KEY"
```

### Issue: External Service Required

**Problem**: Code needs Weaviate/Gemini/etc.

**Solution**: Add requirement annotation

```python
# requires: weaviate
import weaviate

# Test will use mock automatically
client = weaviate.Client("http://localhost:8080")
```

## Configuration

### Basic Configuration

Edit `.doctest.yaml` in project root:

```yaml
scan_paths:
  - docs/

execution:
  timeout: 30
  max_workers: 4

annotations:
  skip:
    - "# doctest: +SKIP"
    - "# skip-test"
```

### Advanced Configuration

```yaml
mocks:
  weaviate:
    enabled: true
    mock_class: "tests.docs.fixtures.MockWeaviateClient"

quality:
  staleness_days: 90
  max_example_lines: 50

reporting:
  format: text
  verbosity: 1
  show_skipped: true
```

See [Code Example Testing Reference](/home/laugustyniak/github/legal-ai/JuDDGES/docs/reference/CODE_EXAMPLE_TESTING.md) for all options.

## Best Practices

### DO

- ✅ Write complete, executable examples
- ✅ Use clear variable names
- ✅ Include all imports
- ✅ Define all variables
- ✅ Use skip markers for non-executable code
- ✅ Test locally before committing

### DON'T

- ❌ Use undefined variables
- ❌ Include placeholders without skip markers
- ❌ Write incomplete code
- ❌ Omit import statements
- ❌ Assume previous context
- ❌ Skip all examples

## Integration with pytest

### Run Specific Tests

```bash
# All documentation tests
pytest tests/docs/ -v

# Only syntax validation
pytest tests/docs/ -v -k "test_syntax"

# Skip integration tests
pytest tests/docs/ -m "docs and not integration"
```

### Custom Markers

```bash
# Tests requiring Weaviate
pytest tests/docs/ -m "requires_weaviate"

# Skip slow tests
pytest tests/docs/ -m "not slow"
```

### Using Fixtures

The test suite provides fixtures for testing:

- `mock_weaviate_client`: Mock Weaviate client
- `mock_gemini_api`: Mock Gemini API
- `sample_dataset`: Sample dataset for testing
- `mock_tokenizer`: Mock HuggingFace tokenizer

## Continuous Integration

### GitHub Actions Workflow

The `docs-quality-checks.yaml` workflow:

1. Extracts code examples from markdown
2. Validates syntax with AST parsing
3. Checks imports are available
4. Runs pytest test suite
5. Generates report artifact
6. Fails PR if tests fail

### Viewing Results

1. Go to PR checks
2. Click on "Documentation Quality Checks"
3. Click on "Code Examples" job
4. View test output and failures
5. Download `code-examples-report.json` artifact

## See Also

- [Code Example Testing Reference](/home/laugustyniak/github/legal-ai/JuDDGES/docs/reference/CODE_EXAMPLE_TESTING.md) - Complete technical reference
- [Contributing to Documentation](/home/laugustyniak/github/legal-ai/JuDDGES/docs/how-to/documentation/CONTRIBUTING_TO_DOCS.md) - General documentation guide
- [Documentation Quick Start](/home/laugustyniak/github/legal-ai/JuDDGES/docs/DOCUMENTATION_QUICK_START.md) - Quick setup guide

## Summary

Testing code examples ensures documentation accuracy:

1. Write testable examples with complete code
2. Use skip markers for demonstration code
3. Test locally before committing
4. CI/CD validates automatically on PR
5. Fix failures and iterate

Keep documentation code examples working, and users will thank you!
