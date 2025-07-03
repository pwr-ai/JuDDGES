# Weaviate Ingestion Test Suite

This directory contains comprehensive test cases for the Weaviate ingestion functionality in the JuDDGES project.

## Test Files Overview

### 1. `test_ingest_to_weaviate.py`

Core unit tests for the main ingestion functionality:

- **`TestGenerateDeterministicUuid`**: Tests UUID generation for documents and chunks
- **`TestIngestConfig`**: Tests configuration handling
- **`TestCollectionIngester`**: Base ingester functionality tests
- **`TestChunkIngester`**: Chunk-specific ingestion tests
- **`TestDocumentIngester`**: Document-specific ingestion tests
- **`TestDatasetLoader`**: Dataset loading functionality tests
- **`TestIntegrationScenarios`**: End-to-end integration tests

### 2. `test_ingest_integration.py`

Real-world scenario and integration tests:

- **`TestRealWorldScenarios`**: Complex real-world data scenarios
- **`TestErrorHandlingAndRecovery`**: Error handling and edge cases
- **`TestPerformanceAndMemory`**: Performance and memory efficiency tests

### 3. `test_dataset_loader.py`

Dedicated tests for the DatasetLoader class:

- **`TestDatasetLoader`**: DatasetLoader functionality tests
- **`TestDatasetLoaderWithRealFiles`**: Tests with actual parquet files

### 4. `conftest.py`

Shared fixtures and test utilities:

- Sample datasets and configurations
- Mock database instances
- Embedding vectors and metadata

## Test Categories

### Unit Tests

- UUID generation and determinism
- Configuration validation
- Dataset filtering and validation
- Batch processing logic

### Integration Tests

- Complete ingestion workflows
- Document and chunk coordination
- Database interaction simulation
- Memory efficiency with large datasets

### Edge Cases

- Empty datasets
- Missing fields
- Malformed JSON
- Invalid embeddings
- Connection failures

### Performance Tests

- Large dataset processing
- Batch size optimization
- Memory usage validation
- Concurrent processing

## Running Tests

### Run All Tests

```bash
cd tests/embeddings
python run_tests.py
```

### Run Quick Tests Only

```bash
python run_tests.py quick
```

### Run Specific Test Class

```bash
python run_tests.py class TestChunkIngester
```

### Using pytest Directly

```bash
# Run all tests with verbose output
pytest -v

# Run specific test file
pytest test_ingest_to_weaviate.py -v

# Run specific test class
pytest -k TestDocumentIngester -v

# Run tests with coverage
pytest --cov=scripts.embed.ingest_to_weaviate

# Run tests and stop on first failure
pytest -x
```

## Test Data Structure

### Sample Document Dataset

```python
{
    "document_id": ["doc_1", "doc_2"],
    "title": ["Legal Case 1", "Tax Interpretation"],
    "language": ["en", "pl"],
    "country": ["US", "PL"],
    "date_issued": ["2023-01-01", "2023-02-01"],
    "document_type": ["judgment", "tax_interpretation"],
    "full_text": ["Legal text...", "Tax text..."]
}
```

### Sample Chunk Dataset

```python
{
    "document_id": ["doc_1", "doc_1"],
    "chunk_id": ["chunk_1", "chunk_2"],
    "chunk_text": ["First chunk...", "Second chunk..."],
    "embedding": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
    "position": [0, 1]
}
```

## Mock Objects

### WeaviateLegalDocumentsDatabase Mock

The tests use comprehensive mocks for the Weaviate database:

```python
mock_db = Mock()
mock_db.legal_documents_collection = Mock()
mock_db.document_chunks_collection = Mock()
mock_db.get_collection_size.return_value = 0
mock_db.get_uuids.return_value = []
```

### Batch Operation Mocks

Batch operations are mocked to verify correct batching behavior:

```python
batch_op = Mock()
collection.batch.fixed_size.return_value.__enter__.return_value = batch_op
```

## Key Test Scenarios

### 1. Document Ingestion

- Validates document metadata handling
- Tests UUID generation and deduplication
- Verifies batch processing efficiency
- Checks default value application

### 2. Chunk Ingestion

- Tests chunk-to-document relationship validation
- Verifies metadata merging from parent documents
- Validates position and reference handling
- Tests complex JSON field processing

### 3. Incremental Ingestion

- Tests upsert vs. insert-only modes
- Validates existing data filtering
- Checks UUID-based deduplication

### 4. Error Handling

- Invalid embedding dimensions
- Malformed JSON in fields
- Database connection failures
- Missing required fields

### 5. Performance Scenarios

- Large dataset processing (1000+ documents)
- Memory efficiency validation
- Batch size optimization
- Concurrent processing simulation

## Configuration Testing

The tests validate various configuration scenarios:

```python
# Debug mode
IngestConfig(
    batch_size=2,
    max_documents=5,
    processing_proc=1,
    ingest_proc=1
)

# Production mode
IngestConfig(
    batch_size=32,
    max_documents=None,
    processing_proc=4,
    ingest_proc=2
)
```

## Dependencies

The test suite requires:

- `pytest`
- `numpy`
- `datasets` (HuggingFace)
- `unittest.mock`
- The main ingestion modules from `scripts.embed.ingest_to_weaviate`

## Test Coverage

The test suite covers:

✅ UUID generation and determinism
✅ Configuration validation
✅ Dataset loading and validation
✅ Batch processing logic
✅ Document ingestion workflows
✅ Chunk ingestion workflows
✅ Error handling and recovery
✅ Performance and memory efficiency
✅ Real-world data scenarios
✅ Edge cases and boundary conditions

## Continuous Integration

For CI/CD pipelines, use:

```yaml
- name: Run Weaviate Ingestion Tests
  run: |
    cd tests/embeddings
    python -m pytest -v --tb=short --cov=scripts.embed.ingest_to_weaviate
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the project root is in PYTHONPATH
2. **Mock Failures**: Verify mock objects are properly configured
3. **Memory Issues**: Use smaller test datasets for local development
4. **File Permissions**: Ensure test files have appropriate permissions

### Debug Mode

Enable debug mode in tests by setting environment variables:

```bash
export DEBUG=true
pytest -v -s  # -s shows print statements
```

## Contributing

When adding new tests:

1. Follow the existing naming conventions
2. Use appropriate fixtures from `conftest.py`
3. Include both positive and negative test cases
4. Add docstrings explaining test purpose
5. Update this README if adding new test categories
