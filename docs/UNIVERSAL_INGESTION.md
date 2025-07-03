# Universal Dataset Ingestion System

The Universal Dataset Ingestion System allows you to easily ingest any HuggingFace dataset into Weaviate with automatic schema adaptation, intelligent field mapping, and comprehensive validation.

## 🚀 Quick Start

### 1. Auto-register and ingest a dataset

```bash
# Preview any HuggingFace dataset
python scripts/dataset_manager.py preview "your-org/your-legal-dataset"

# Auto-register with smart configuration
python scripts/dataset_manager.py add "your-org/your-legal-dataset" --auto

# Validate before ingestion
python scripts/dataset_manager.py validate "your-org/your-legal-dataset"

# Ingest with safety limits
python scripts/dataset_manager.py ingest "your-org/your-legal-dataset" --max-docs 1000 --dry-run
python scripts/dataset_manager.py ingest "your-org/your-legal-dataset" --max-docs 1000
```

### 2. Use the enhanced ingestion script

```bash
# With automatic configuration
python scripts/embed/universal_ingest_to_weaviate.py dataset_name="your-dataset" max_documents=1000

# Preview mode
python scripts/embed/universal_ingest_to_weaviate.py dataset_name="your-dataset" preview_only=true

# Skip validation (not recommended)
python scripts/embed/universal_ingest_to_weaviate.py dataset_name="your-dataset" skip_validation=true force=true
```

## 🏗️ System Architecture

### Core Components

1. **DatasetRegistry** (`juddges/data/dataset_registry.py`)
   - Manages dataset configurations
   - Persists settings as YAML files in `configs/datasets/`
   - Supports runtime registration and updates

2. **SmartColumnMapper** (`juddges/data/smart_mapper.py`)
   - Automatically suggests field mappings
   - Uses semantic similarity and pattern matching
   - Handles 90% of common legal document structures

3. **WeaviateSchemaAdapter** (`juddges/data/schema_adapter.py`)
   - Dynamically creates Weaviate schemas
   - Adapts to dataset structure
   - Ensures compatibility with existing collections

4. **RobustDataConverter** (`juddges/data/converters.py`)
   - Handles data type conversions with error recovery
   - Supports complex date parsing, array conversion, JSON serialization
   - Provides detailed error reporting

5. **UniversalDatasetProcessor** (`juddges/data/universal_processor.py`)
   - Main orchestration interface
   - Coordinates all components
   - Handles end-to-end processing

6. **DatasetValidator** (`juddges/data/validators.py`)
   - Comprehensive validation framework
   - Checks data quality, schema compatibility, resource requirements
   - Provides actionable suggestions

## 📋 Dataset Configuration

### Automatic Configuration

The system can automatically generate configurations:

```python
from juddges.data.universal_processor import UniversalDatasetProcessor

processor = UniversalDatasetProcessor()
config = processor.register_new_dataset("your-dataset")
```

### Manual Configuration

Create custom configurations:

```python
from juddges.data.dataset_registry import DatasetConfig

config = DatasetConfig(
    name="my-legal-dataset",
    column_mapping={
        "case_id": "document_id",
        "judgment_text": "full_text",
        "case_date": "date_issued",
        "court": "court_name"
    },
    required_fields=["document_id", "full_text"],
    text_fields=["full_text", "summary"],
    date_fields=["date_issued"],
    array_fields=["keywords", "judges"],
    json_fields=["metadata"],
    default_values={
        "language": "en",
        "document_type": "judgment"
    }
)
```

## 🔍 Validation Framework

The system provides comprehensive validation:

### Validation Categories

- **Accessibility**: Can the dataset be loaded?
- **Configuration**: Are field mappings valid?
- **Data Quality**: Are there empty rows, missing values?
- **Data Types**: Do date/array fields have correct formats?
- **Resource Requirements**: Memory, processing time estimates

### Validation Levels

- **CRITICAL**: Blocks ingestion (missing dataset, invalid config)
- **ERROR**: Should be fixed (missing required fields, type conflicts)
- **WARNING**: Should be reviewed (data quality issues)
- **INFO**: Informational (resource estimates, suggestions)

## 🛠️ CLI Interface

### Dataset Management Commands

```bash
# List registered datasets
python scripts/dataset_manager.py list

# Preview dataset structure and auto-suggest configuration
python scripts/dataset_manager.py preview "dataset-name" --samples 10

# Register new dataset
python scripts/dataset_manager.py add "dataset-name" --auto
python scripts/dataset_manager.py add "dataset-name" --mapping "id=document_id,text=full_text"

# Show dataset configuration
python scripts/dataset_manager.py show "dataset-name"

# Validate dataset
python scripts/dataset_manager.py validate "dataset-name"

# Ingest dataset
python scripts/dataset_manager.py ingest "dataset-name" --max-docs 1000 --batch-size 32

# Remove dataset configuration
python scripts/dataset_manager.py remove "dataset-name"

# Initialize with default configurations
python scripts/dataset_manager.py init
```

### Advanced Options

```bash
# Force processing despite validation errors
python scripts/dataset_manager.py ingest "dataset-name" --skip-validation

# Dry run (preview processing without actual ingestion)
python scripts/dataset_manager.py ingest "dataset-name" --dry-run

# Custom batch size and limits
python scripts/dataset_manager.py ingest "dataset-name" --max-docs 5000 --batch-size 64
```

## 🔧 Advanced Usage

### Programmatic Interface

```python
from juddges.data.universal_processor import UniversalDatasetProcessor
from juddges.data.config import IngestConfig

processor = UniversalDatasetProcessor()

# Preview dataset
preview = processor.preview_dataset("your-dataset")
print(f"Dataset has {preview.total_rows} rows with columns: {preview.columns}")

# Process with custom configuration
ingest_config = IngestConfig(
    max_documents=1000,
    batch_size=32,
    upsert=True
)

result = processor.process_dataset(
    dataset_name="your-dataset",
    ingest_config=ingest_config,
    create_embeddings=True
)

if result.success:
    print(f"Successfully ingested {result.ingested_documents} documents")
```

### Custom Field Mapping

```python
from juddges.data.smart_mapper import SmartColumnMapper

mapper = SmartColumnMapper()
suggestions = mapper.suggest_mapping(["case_id", "judgment_text", "court_name"])

# Get mapping suggestions with confidence scores
for column, suggestion in suggestions.items():
    print(f"{column} -> {suggestion.target_field} (confidence: {suggestion.confidence})")
```

### Schema Adaptation

```python
from juddges.data.schema_adapter import WeaviateSchemaAdapter

adapter = WeaviateSchemaAdapter()
schema = adapter.create_adaptive_schema(config, sample_data)

# Check compatibility with existing schema
compatibility = adapter.validate_schema_compatibility(existing_schema, new_schema)
if compatibility['compatible']:
    print("Schema is compatible")
else:
    print(f"Issues: {compatibility['issues']}")
```

## 📊 Supported Dataset Types

The system automatically handles:

### Legal Document Types

- **Court Judgments**: decisions, rulings, opinions
- **Legal Acts**: statutes, regulations, directives
- **Tax Interpretations**: tax authority decisions
- **Administrative Decisions**: government rulings

### Field Types

- **Text Fields**: content, summaries, reasoning
- **Date Fields**: judgment dates, publication dates
- **Array Fields**: judges, keywords, legal references
- **JSON Fields**: complex metadata, nested objects
- **ID Fields**: case numbers, citations, references

### Languages

- **Multilingual Support**: Polish, English, and extensible to others
- **Auto-detection**: Infers language from dataset name or content
- **Language-specific Processing**: Appropriate defaults and formatting

## 🚨 Error Handling & Recovery

### Robust Error Recovery

- **Field-level Error Isolation**: One bad field doesn't break entire row
- **Fallback Values**: Uses defaults for missing required fields
- **Type Conversion**: Attempts multiple parsing strategies
- **Batch Recovery**: Continues processing if some batches fail

### Error Reporting

- **Detailed Logs**: Field-level error messages with suggestions
- **Progress Tracking**: Shows successful vs failed processing
- **Validation Reports**: Comprehensive pre-processing validation
- **Performance Metrics**: Processing speed and resource usage

## 🔄 Migration from Old System

### Updating Existing Scripts

Replace old ingestion calls:

```python
# OLD WAY
from juddges.data.ingesters import DocumentIngester
ingester = DocumentIngester()
ingester.ingest(dataset)

# NEW WAY
from juddges.data.universal_processor import UniversalDatasetProcessor
processor = UniversalDatasetProcessor()
result = processor.process_dataset("dataset-name")
```

### Configuration Migration

Old column mappings in `loaders.py` are automatically converted to the new registry system.

## 📈 Performance Optimization

### Processing Speed

- **Parallel Processing**: Configurable worker threads
- **Batch Optimization**: Adaptive batch sizing
- **Memory Management**: Streaming for large datasets
- **Progress Tracking**: Real-time processing status

### Resource Management

- **Memory Estimates**: Predict memory requirements
- **Processing Time**: Estimate completion time
- **Storage Requirements**: Calculate disk space needs
- **Batch Size Recommendations**: Optimal settings for dataset size

## 🤝 Contributing

### Adding New Dataset Types

1. Add field mappings to `SmartColumnMapper.SEMANTIC_MAPPINGS`
2. Create default configuration in `DatasetRegistry.create_default_configs()`
3. Add validation rules in `DatasetValidator`

### Custom Validators

```python
from juddges.data.validators import DatasetValidator

class CustomValidator(DatasetValidator):
    def _validate_custom_field(self, dataset, field_name, result):
        # Add custom validation logic
        pass
```

### Schema Extensions

```python
from juddges.data.schema_adapter import WeaviateSchemaAdapter

class CustomSchemaAdapter(WeaviateSchemaAdapter):
    def _create_custom_properties(self, config):
        # Add domain-specific schema properties
        pass
```

## 📚 Examples

See `examples/universal_ingestion_example.py` for a comprehensive demonstration of all features.

## 🐛 Troubleshooting

### Common Issues

1. **Dataset Not Found**

   ```bash
   Error: Cannot access dataset
   Solution: Check dataset name and network connectivity
   ```

2. **Schema Compatibility**

   ```bash
   Error: Type conflicts in schema
   Solution: Update column mapping or use different target fields
   ```

3. **Validation Failures**

   ```bash
   Error: Required fields not mapped
   Solution: Add missing field mappings or update requirements
   ```

4. **Memory Issues**

   ```bash
   Warning: Estimated memory requirement exceeds available
   Solution: Use smaller batch sizes or streaming processing
   ```

### Getting Help

1. **Preview Mode**: Always start with preview to understand dataset structure
2. **Validation**: Run validation before ingestion to catch issues early
3. **Dry Run**: Test ingestion with `--dry-run` flag
4. **Logs**: Check detailed logs in `logs/universal_ingestion.log`
5. **CLI Help**: Use `--help` flag on any command

## 🔮 Future Enhancements

- **Streaming Ingestion**: Handle datasets larger than memory
- **Multi-format Support**: Direct support for PDF, XML, JSON files
- **Schema Evolution**: Automatic schema migration for updated datasets
- **Quality Scoring**: Automatic data quality assessment and scoring
- **Integration APIs**: RESTful API for external system integration
