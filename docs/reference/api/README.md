# API Reference Documentation

Welcome to the JuDDGES API reference documentation! This section provides comprehensive, auto-generated documentation for all public Python APIs.

## Quick Links

- **[Complete API Index](index.md)** - Browse all modules and functions
- **[Data Management](data/index.md)** - Dataset loading and Weaviate operations
- **[LLM Operations](llm/factory.md)** - Model loading and inference
- **[Information Extraction](extraction/gemini_chain.md)** - Structured extraction with Gemini
- **[Evaluation](evals/metrics.md)** - Evaluation metrics
- **[Preprocessing](preprocessing/text_chunker.md)** - Text processing utilities

## Getting Started

### Installation

API documentation requires MkDocs and dependencies:

```bash
# Install documentation dependencies
uv pip install mkdocs mkdocs-material mkdocstrings[python]

# Or install all dev dependencies
uv pip install -e ".[dev]"
```

### Viewing Documentation

```bash
# Serve locally (auto-reload on changes)
mkdocs serve

# Or use the convenience script
./scripts/docs/generate_api_docs.sh --serve

# Visit http://127.0.0.1:8000
```

### Building Static Site

```bash
# Build static HTML
mkdocs build

# Output in site/ directory
# Deploy to any static hosting service
```

## Documentation Structure

### By Module Type

**Core Configuration**:
- [Config](core/config.md) - Hydra configuration
- [Settings](core/settings.md) - Application settings
- [Data Models](core/data_models.md) - Core data structures

**Data Management**:
- [Loaders](data/loaders.md) - Dataset loading
- [Judgments DB](data/judgments_weaviate_db.md) - Weaviate operations
- [Stream Ingester](data/stream_ingester.md) - Production ingestion

**LLM Operations**:
- [Factory](llm/factory.md) - Model creation
- [Predict](llm/predict.md) - Inference pipeline

**Information Extraction**:
- [Gemini Chain](extraction/gemini_chain.md) - Gemini-based extraction

**Evaluation**:
- [Metrics](evals/metrics.md) - Field-level metrics
- [LLM as Judge](llm_as_judge/judge.md) - LLM-based evaluation

**Preprocessing**:
- [Text Chunker](preprocessing/text_chunker.md) - Document chunking
- [Text Encoder](preprocessing/text_encoder.md) - Tokenization

### By Use Case

**Data Ingestion**:
1. [Loaders](data/loaders.md) - Load datasets
2. [Stream Ingester](data/stream_ingester.md) - Ingest to Weaviate
3. [Judgments DB](data/judgments_weaviate_db.md) - Database operations

**Model Inference**:
1. [Factory](llm/factory.md) - Load model
2. [Predict](llm/predict.md) - Generate predictions
3. [Metrics](evals/metrics.md) - Evaluate results

**Information Extraction**:
1. [Gemini Chain](extraction/gemini_chain.md) - Extract with Gemini
2. [Metrics](evals/metrics.md) - Evaluate extractions
3. [LLM as Judge](llm_as_judge/judge.md) - LLM evaluation

## Key Features

### Auto-Generated from Code

All documentation is automatically generated from Python docstrings:

- **Always Up-to-Date**: Docs sync with code
- **Type Annotations**: Full type information
- **Examples**: Usage examples from docstrings
- **Source Links**: Jump to source code

### Google-Style Docstrings

Consistent documentation format:

```python
def function(arg: str) -> dict:
    """Brief description of function.

    Longer description with more context.

    Args:
        arg: Description of argument

    Returns:
        Description of return value

    Example:
        >>> result = function("test")
        >>> print(result)
        {'key': 'value'}
    """
```

### Rich Features

- **Search**: Full-text search across all APIs
- **Syntax Highlighting**: Python, Shell, YAML, JSON
- **Dark Mode**: Light/dark theme toggle
- **Mobile Friendly**: Responsive design
- **Cross-References**: Links between related APIs

## Contributing

### Adding API Documentation

1. **Write Docstrings**: Add Google-style docstrings to your code
2. **Create Page**: Add markdown file in `docs/reference/api/`
3. **Include Module**: Use `:::` directive to include API
4. **Test Locally**: Run `mkdocs serve`
5. **Commit**: Include docs with code changes

### Documentation Standards

- **Google-style docstrings** required
- **Type annotations** for all parameters
- **Usage examples** in docstrings
- **Cross-references** to related APIs
- **Common patterns** documented

## Maintenance

### Generation Script

Use the provided script for common tasks:

```bash
# Validate documentation
./scripts/docs/generate_api_docs.sh

# Serve locally
./scripts/docs/generate_api_docs.sh --serve

# Build for production
./scripts/docs/generate_api_docs.sh --build

# Strict validation (warnings as errors)
./scripts/docs/generate_api_docs.sh --build --strict
```

### Quality Metrics

Current documentation coverage:

| Metric | Current | Target |
|--------|---------|--------|
| Core modules documented | 8/20 | 20/20 |
| API pages | 10 | 25+ |
| Docstring coverage | ~60% | 90%+ |
| Examples per function | 1 | 1-2 |

## Resources

### Documentation

- [API Index](index.md) - Complete module listing

### External Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

## Need Help?

- **Missing Documentation**: [Report an issue](https://github.com/laugustyniak/JuDDGES/issues)
- **Unclear API**: Request clarification in GitHub Discussions
- **Contributing**: See [Contributing Guide](../../../CONTRIBUTING.md)

---

**Last Updated**: 2025-10-11
**Coverage**: ~60% of public APIs
**Next Review**: 2025-11-01

**Ready to explore?** → [Browse Complete API Index](index.md)
