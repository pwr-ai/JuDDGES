# Code Example Testing - Quick Start Card

One-page reference for testing documentation code examples in JuDDGES.

## 🚀 Quick Commands

```bash
# Test all documentation
python scripts/docs/test_code_examples.py

# Test specific file
python scripts/docs/test_code_examples.py docs/reference/api/README.md

# Verbose output
python scripts/docs/test_code_examples.py --verbose

# Generate report
python scripts/docs/test_code_examples.py --report report.json

# Use pytest
pytest tests/docs/test_documentation_examples.py -v

# Pre-commit check
pre-commit run test-doc-code-examples --all-files
```

## ✍️ Writing Testable Examples

### ✅ Good (Will Pass)

```python
from juddges.preprocessing.text_chunker import TextChunker

# All variables defined
chunker = TextChunker(
    id_col="judgment_id",
    text_col="full_text",
    chunk_size=512
)

# Sample data provided
dataset = {
    "judgment_id": ["doc1"],
    "full_text": ["Sample text..."]
}

# Complete example
result = chunker(dataset)
```

### ❌ Bad (Will Fail)

```python
# Missing imports
chunker = TextChunker(config)  # config undefined

# Incomplete code
def process():
    ...  # Not executable

# Placeholders
api_key = "YOUR_API_KEY"  # Will be detected
```

## 🏷️ Annotations

### Skip This Example

```python
# doctest: +SKIP
# Production code not suitable for testing
```

### Require External Service

```python
# requires: weaviate
import weaviate
client = weaviate.Client("http://localhost:8080")
# Will use mock during tests
```

### Demonstration Only

```python
# demonstration-only
# Complex production example
```

## 📊 Check Status

### Local

```bash
# Run tests
python scripts/docs/test_code_examples.py --verbose

# Check git
git status

# Run pre-commit
pre-commit run --all-files
```

### CI/CD

1. Create PR
2. Check "Code Examples" job in PR checks
3. Download `code-examples-report.json` artifact if failed
4. Fix issues
5. Push again

## 🔧 Troubleshooting

### Import Error

```python
# Bad
from juddges.old_module import something

# Good
from juddges.preprocessing.text_chunker import TextChunker
```

### Undefined Variable

```python
# Bad
result = process(data)  # data undefined

# Good
data = {"key": "value"}
result = process(data)
```

### External Service Needed

```python
# Solution 1: Skip
# doctest: +SKIP
import weaviate

# Solution 2: Use mock
# requires: weaviate
import weaviate  # Will use mock
```

## 📚 Documentation

- **Quick Guide**: `/home/laugustyniak/github/legal-ai/JuDDGES/docs/how-to/documentation/TEST_CODE_EXAMPLES.md`
- **Reference**: `/home/laugustyniak/github/legal-ai/JuDDGES/docs/reference/CODE_EXAMPLE_TESTING.md`
- **Implementation**: `/home/laugustyniak/github/legal-ai/JuDDGES/docs/CODE_EXAMPLE_TESTING_IMPLEMENTATION.md`

## 🎯 Available Markers

```bash
# All doc tests
pytest -m docs

# Skip integration
pytest -m "docs and not integration"

# Specific requirement
pytest -m requires_weaviate
```

## ⚙️ Configuration

Edit `.doctest.yaml` for:

- Paths to scan
- Execution timeout
- Skip patterns
- Mock settings
- Report format

## 🏁 Workflow

1. **Write** documentation with code examples
2. **Test** locally: `python scripts/docs/test_code_examples.py --verbose`
3. **Fix** any failures with annotations or corrections
4. **Commit** - pre-commit hook runs automatically
5. **Push** - CI/CD validates on PR
6. **Merge** - live docs update automatically

## 💡 Best Practices

- ✅ Write complete, self-contained examples
- ✅ Define all variables
- ✅ Include all imports
- ✅ Use skip markers for non-executable code
- ✅ Test locally before committing
- ❌ Don't use placeholders without skip markers
- ❌ Don't assume previous context
- ❌ Don't skip all examples

## 📞 Getting Help

- **Usage Questions**: See how-to guide
- **Technical Details**: See reference docs
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

**Quick Reference Version**: 1.0
**Last Updated**: 2025-10-11
**System Status**: ✅ Production Ready
