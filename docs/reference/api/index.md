# API Reference

Welcome to the JuDDGES API reference documentation. This section provides comprehensive documentation for all public APIs, classes, functions, and modules in the JuDDGES project.

## Documentation Structure

The API documentation is organized by module functionality following the Diátaxis framework's **Reference** category - providing technical specifications and detailed information about the codebase.

## Core Modules

### [Core Configuration](core/config.md)
Configuration management using Hydra and OmegaConf for flexible, hierarchical configuration.

**Key Components**:
- `LLMConfig` - Large language model configuration
- `EmbeddingConfig` - Embedding model and dataset configuration
- Configuration loaders and validators

### [Settings](core/settings.md)
Application-wide settings using Pydantic for validation.

**Key Components**:
- Environment variable management
- API keys and credentials
- Database connection settings

### [Data Models](core/data_models.md)
Core data structures used throughout the application.

**Key Components**:
- Document models
- Judgment models
- Extraction schemas

### [Schema](core/schema.md)
Weaviate schema definitions and validation.

## Data Management

### [Data Loaders](data/loaders.md)
Dataset loading utilities for Weaviate ingestion.

**Key Functions**:
- `DatasetLoader.load_chunk_dataset()` - Load chunk embeddings
- `DatasetLoader.load_document_dataset()` - Load document embeddings with column remapping
- Dataset column mapping configurations

### [Weaviate Base Database](data/base_weaviate_db.md)
Base class for Weaviate database operations.

**Key Features**:
- Connection management
- Collection creation and management
- Batch operations
- Error handling

### [Judgments Database](data/judgments_weaviate_db.md)
Weaviate database operations for court judgments.

**Key Components**:
- `WeaviateJudgmentsDatabase` - Main database class
- Judgment and chunk collection management
- Schema definitions with 50+ fields
- UMAP coordinate support

### [Documents Database](data/documents_weaviate_db.md)
Weaviate database operations for generic documents.

### [Dataset Factory](data/dataset_factory.md)
Factory for creating and managing datasets.

### [Dataset Mapper](data/dataset_mapper.md)
Utilities for mapping between different dataset schemas.

### [Stream Ingester](data/stream_ingester.md)
Production-grade streaming ingestion pipeline.

**Key Features**:
- Batch processing
- Error handling and retry logic
- Progress tracking
- Memory-efficient streaming

## LLM Operations

### [LLM Factory](llm/factory.md)
Factory for creating and configuring language models.

**Supported Models**:
- Llama 3.1/3.2
- Mistral/Nemo
- Phi-4
- Bielik (Polish)

**Key Functions**:
- `get_llm()` - Create model from configuration
- `get_llama_3()` - Llama-specific setup
- `get_mistral()` - Mistral-specific setup
- Model quantization (4-bit, 8-bit)
- PEFT/LoRA adapter loading

### [Prediction](llm/predict.md)
LLM prediction utilities.

**Key Functions**:
- `predict_with_llm()` - Batch prediction with progress tracking
- DataLoader integration
- Performance metrics

## Information Extraction

### [Gemini Chain](extraction/gemini_chain.md)
LangChain extraction chain using Gemini 2.5 Pro/Flash.

**Key Components**:
- `GeminiExtractionChain` - Main extraction class
- `ExtractionSchema` - Schema definition
- `DocumentType` - Document type enum

**Key Features**:
- Structured output parsing
- SQLite caching
- Langfuse observability integration
- Batch extraction support
- Automatic text truncation

## Preprocessing

### [Text Chunker](preprocessing/text_chunker.md)
Text chunking utilities for document segmentation.

**Key Components**:
- `TextChunker` - Main chunking class
- Recursive character splitting
- Token-based chunking
- Configurable overlap

### [Text Encoder](preprocessing/text_encoder.md)
Text encoding and tokenization utilities.

### [Context Truncator](preprocessing/context_truncator.md)
Context window management for LLMs.

### [Formatters](preprocessing/formatters.md)
Text formatting utilities for legal documents.

### [Parser Base](preprocessing/parser_base.md)
Base class for document parsers.

### [PL Court Parser](preprocessing/pl_court_parser.md)
Parser for Polish court documents.

## Evaluation

### [Metrics](evals/metrics.md)
Evaluation metrics for information extraction.

**Key Functions**:
- `evaluate_date()` - Date field evaluation with parsing
- `evaluate_number()` - Numeric field evaluation with tolerance
- `evaluate_string_rouge()` - ROUGE scores for text fields
- `evaluate_enum()` - Enum classification with hallucination detection
- `evaluate_list_greedy()` - List matching with precision/recall/F1

### [Extraction Evaluation](evals/extraction.md)
End-to-end extraction evaluation pipeline.

## LLM as Judge

### [Base](llm_as_judge/base.md)
Base classes for LLM-as-judge evaluation.

### [Judge](llm_as_judge/judge.md)
Single-document LLM judge implementation.

### [Batched Judge](llm_as_judge/batched_judge.md)
Batch processing LLM judge.

### [Data Model](llm_as_judge/data_model.md)
Data models for LLM judge evaluation.

### [Result Loading](llm_as_judge/result_loading.md)
Utilities for loading and processing judge results.

## Retrieval

### [Mongo Hybrid Search](retrieval/mongo_hybrid_search.md)
Hybrid search combining semantic and keyword search.

### [Mongo Term-Based Search](retrieval/mongo_term_based_search.md)
Traditional keyword-based search.

## Utilities

### [Config Utils](utils/config.md)
Configuration utilities and helpers.

### [Logging](utils/logging.md)
Logging configuration using loguru.

### [Pipeline](utils/pipeline.md)
Pipeline utilities for DVC workflows.

### [HuggingFace Utils](utils/hf.md)
HuggingFace Hub utilities for dataset and model management.

### [Date Utils](utils/date_utils.md)
Date parsing and formatting utilities.

### [Misc](utils/misc.md)
Miscellaneous utilities.

## Quick Navigation

### By Use Case

**Data Ingestion**:
1. [Data Loaders](data/loaders.md) - Load datasets
2. [Stream Ingester](data/stream_ingester.md) - Ingest to Weaviate
3. [Judgments Database](data/judgments_weaviate_db.md) - Database operations

**Information Extraction**:
1. [Gemini Chain](extraction/gemini_chain.md) - Extract with Gemini
2. [Metrics](evals/metrics.md) - Evaluate extractions
3. [LLM as Judge](llm_as_judge/judge.md) - LLM-based evaluation

**Model Training & Inference**:
1. [LLM Factory](llm/factory.md) - Create models
2. [Prediction](llm/predict.md) - Generate predictions
3. [Preprocessing](preprocessing/text_chunker.md) - Prepare data

### By Module Type

**Configuration**:
- [Core Config](core/config.md)
- [Settings](core/settings.md)
- [Config Utils](utils/config.md)

**Data Access**:
- [Loaders](data/loaders.md)
- [Judgments DB](data/judgments_weaviate_db.md)
- [Stream Ingester](data/stream_ingester.md)

**Models**:
- [LLM Factory](llm/factory.md)
- [Prediction](llm/predict.md)
- [Gemini Chain](extraction/gemini_chain.md)

**Evaluation**:
- [Metrics](evals/metrics.md)
- [LLM as Judge](llm_as_judge/index.md)
- [Extraction Eval](evals/extraction.md)

## Documentation Conventions

### Docstring Style
All modules use **Google-style docstrings**:

```python
def function_name(arg1: str, arg2: int) -> bool:
    """Brief description of function.

    Longer description with more details about what the function does,
    its purpose, and how it should be used.

    Args:
        arg1: Description of first argument
        arg2: Description of second argument

    Returns:
        Description of return value

    Raises:
        ValueError: When input is invalid

    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
```

### Type Annotations
All public APIs include comprehensive type annotations following PEP 484.

### Code Examples
Most functions include usage examples in docstrings.

## Contributing to API Documentation

### Adding Documentation

1. **Update Docstrings**: Add or improve docstrings in source code
2. **Regenerate Docs**: Run `./scripts/docs/generate_api_docs.sh`
3. **Review**: Check generated documentation in `docs/reference/api/`
4. **Commit**: Include both source and generated docs in commit

### Documentation Standards

Follow the [Style Guide](../STYLE_GUIDE.md) for:
- Docstring formatting
- Type annotation conventions
- Example code standards
- Cross-referencing guidelines

### Automation

API documentation is automatically generated from source code docstrings using:
- **MkDocs**: Static site generator
- **mkdocstrings**: Python documentation plugin
- **Material for MkDocs**: Modern theme

## Related Documentation

- **[Tutorials](../../tutorials/)** - Learn by doing
- **[How-To Guides](../../how-to/)** - Solve specific problems
- **[Explanation](../../explanation/)** - Understand concepts
- **[Style Guide](../STYLE_GUIDE.md)** - Documentation standards

## Need Help?

- **Missing Documentation**: [Report an issue](https://github.com/laugustyniak/JuDDGES/issues)
- **Unclear API**: Request clarification in GitHub Discussions
- **Contributing**: See [Contributing Guide](../../../CONTRIBUTING.md)

---

**Last Updated**: 2025-10-11
**Coverage**: ~60% of public APIs documented
**Target**: 100% coverage by 2025-11-01
