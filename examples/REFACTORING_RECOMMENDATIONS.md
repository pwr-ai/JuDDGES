# Refactoring Recommendations for Standalone Package

Based on your current streaming ingester, here are the key changes needed to create a standalone, reusable package for legal document ingestion.

## Current State Analysis

Your existing `stream_ingester.py` is already well-designed with:

- ✅ Streaming processing (constant memory usage)
- ✅ SQLite-based progress tracking
- ✅ Weaviate v4 API support
- ✅ Rich CLI interface
- ✅ Error handling and resume capability

## Key Changes Needed

### 1. **Configuration System** ⭐⭐⭐

**Current**: Hardcoded field mappings in `_process_document()`
**Change**: JSON/YAML configuration files

```python
# Current (hardcoded)
doc_id = doc.get('document_id') or doc.get('judgment_id') or doc.get('id')

# Proposed (configurable)
config = DatasetConfig.load("polish_courts.json")
doc_id = self._extract_field(doc, config.id_field_mappings)
```

**Files to create**:

- `config/datasets/polish_courts.json`
- `config/datasets/tax_interpretations.json`
- `config/manager.py`

### 2. **Pluggable Transformations** ⭐⭐⭐

**Current**: No systematic data transformation
**Change**: Transformation engine with built-in and custom transformers

```python
# Current (no transformation)
court_name = doc.get('court_name')

# Proposed (with transformation)
court_name = transformer.transform(doc.get('court_id'), 'court_id_to_name')
```

**Use cases**:

- `court_id` → `court_name` mapping
- Date format standardization
- JSON serialization for metadata
- Legal reference normalization

### 3. **API Redesign** ⭐⭐

**Current**: Single class with constructor parameters
**Change**: Factory pattern with configuration

```python
# Current
ingester = StreamingIngester(weaviate_url="...", embedding_model="...")

# Proposed
ingester = create_ingester("polish-courts", weaviate_url="...")
```

### 4. **Package Structure** ⭐⭐

**Current**: Part of JuDDGES repository
**Change**: Standalone pip-installable package

```
legal-doc-ingester/
├── src/legal_doc_ingester/
│   ├── core/          # Main ingestion logic
│   ├── config/        # Configuration management
│   ├── transforms/    # Data transformations
│   ├── storage/       # Weaviate and tracking
│   └── cli/          # Command-line interface
```

## Implementation Priority

### Phase 1: Core Refactoring (Week 1)

1. **Extract Configuration System**
   - Create `DatasetConfig` class
   - Move field mappings to JSON files
   - Add configuration manager

2. **Add Transformation Engine**
   - Implement basic transformers (date, JSON, lookup)
   - Create transformation registry
   - Add court mapping transformer

3. **Enhance API Interface**
   - Add factory function
   - Improve error handling
   - Add validation

### Phase 2: Package Creation (Week 2)

1. **Package Structure**
   - Create proper package structure
   - Add pyproject.toml
   - Set up build system

2. **CLI Enhancement**
   - Add Typer-based CLI
   - Rich progress displays
   - Configuration validation commands

3. **Documentation**
   - README with examples
   - Configuration guide
   - API documentation

### Phase 3: Advanced Features (Week 3)

1. **Domain-Specific Modules**
   - Polish legal transformations
   - English legal transformations
   - Tax interpretation specializations

2. **Storage Options**
   - Redis tracker option
   - PostgreSQL tracker option
   - Configurable tracking backends

3. **Performance Optimizations**
   - Async processing options
   - Memory usage optimization
   - Batch size auto-tuning

## Concrete Changes to Current Code

### 1. Modify `StreamingIngester.__init__()`

```python
# Current
def __init__(self, weaviate_url: str = "http://localhost:8080", ...):

# Proposed
def __init__(self, config: DatasetConfig, weaviate_url: str = "http://localhost:8080", ...):
    self.config = config
    self.transformation_engine = transformation_engine or TransformationEngine()
```

### 2. Replace `_process_document()` logic

```python
# Current
def _process_document(self, doc):
    doc_id = doc.get('document_id') or doc.get('judgment_id') or doc.get('id')
    # ... hardcoded mappings

# Proposed
def _process_document(self, doc):
    return self.document_processor.process_document(doc, self.config)
```

### 3. Add configuration loading

```python
# New factory function
def create_ingester(dataset_name: str, **kwargs):
    config_manager = ConfigurationManager()
    config = config_manager.get_config(dataset_name)

    transformation_engine = TransformationEngine()
    if "polish" in dataset_name:
        setup_polish_transformations(transformation_engine)

    return StreamingIngester(config=config, transformation_engine=transformation_engine, **kwargs)
```

## Migration Strategy

### Backward Compatibility

Keep current interface working while adding new one:

```python
class StreamingIngester:
    def __init__(self, config=None, weaviate_url="...", **kwargs):
        if config is None:
            # Legacy mode - use defaults
            config = self._create_legacy_config(**kwargs)

        self.config = config
        # ... rest of initialization
```

### Data Migration

Current SQLite tracker is compatible - no changes needed.

### Testing Strategy

1. Keep existing tests working
2. Add configuration tests
3. Add transformation tests
4. Integration tests with real datasets

## Benefits of Refactoring

### For Users

- **Easier Configuration**: JSON/YAML configs vs Python code
- **Reusability**: One package for all legal datasets
- **Extensibility**: Add custom transformations without code changes
- **Better Documentation**: Clear configuration examples

### For Developers

- **Maintainability**: Separate concerns (config, transforms, ingestion)
- **Testability**: Unit test configurations and transformations
- **Flexibility**: Support new datasets without code changes
- **Distribution**: pip install vs git clone

### For Legal AI Community

- **Standardization**: Common interface for legal document ingestion
- **Collaboration**: Share configurations and transformations
- **Reproducibility**: Version-controlled configurations
- **Adoption**: Lower barrier to entry

## Risk Mitigation

### Development Risks

- **Scope Creep**: Start with minimal viable product
- **Compatibility**: Maintain backward compatibility during transition
- **Performance**: Benchmark before/after refactoring

### Adoption Risks

- **Migration Burden**: Provide migration tools and documentation
- **Learning Curve**: Keep API simple and well-documented
- **Ecosystem Lock-in**: Support multiple vector databases (future)

## Success Metrics

### Technical Metrics

- Package size < 50MB
- Installation time < 30 seconds
- Memory usage same or better than current
- Processing speed same or better than current

### Usability Metrics

- Configuration time < 10 minutes for new dataset
- Documentation covers 95% of use cases
- CLI covers 80% of common tasks
- Migration guide enables smooth transition

## Next Steps

1. **Start with Configuration System**: Most immediate impact
2. **Add Basic Transformations**: Address court_id mapping need
3. **Create Package Structure**: Enable distribution
4. **Enhance CLI**: Improve user experience
5. **Add Documentation**: Ensure adoption

The refactoring aligns perfectly with your goal of creating a reusable package while building on the solid foundation of your current streaming ingester.
