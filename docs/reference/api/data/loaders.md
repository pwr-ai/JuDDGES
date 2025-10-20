# Data Loaders

Dataset loading utilities for Weaviate ingestion with column remapping support.

## Overview

The `juddges.data.loaders` module provides the `DatasetLoader` class for loading and preparing datasets for Weaviate ingestion. It handles:

- Loading chunk embeddings from parquet files
- Loading document embeddings with dataset metadata
- Column mapping between dataset schemas and Weaviate schemas
- Dataset validation and error handling

## Key Features

- **Automatic Column Remapping**: Maps dataset columns to Weaviate schema fields
- **Multi-Dataset Support**: Predefined mappings for multiple datasets
- **Validation**: Checks for embeddings existence and dataset integrity
- **Efficient Loading**: Uses Polars for fast data processing
- **Error Handling**: Comprehensive error messages and validation

## Usage Example

```python
from juddges.config import EmbeddingConfig
from juddges.data.loaders import DatasetLoader

# Create configuration
config = EmbeddingConfig(
    dataset_name="juddges/pl-court-raw",
    agg_embeddings_dir="data/embeddings/agg",
    chunk_embeddings_dir="data/embeddings/chunks"
)

# Initialize loader
loader = DatasetLoader(config)

# Load document dataset with remapped columns
doc_dataset = loader.load_document_dataset()
# Columns are automatically remapped from dataset schema to Weaviate schema

# Load chunk dataset
chunk_dataset = loader.load_chunk_dataset()
```

## Dataset Column Mappings

The module includes predefined column mappings for datasets to ensure compatibility with Weaviate schema:

### Polish Court Dataset (`juddges/pl-court-raw`)

Maps raw dataset columns to Weaviate fields:

| Dataset Column | Weaviate Field |
|----------------|----------------|
| `document_id` | `judgment_id` |
| `document_number` | `docket_number` |
| `date_issued` | `judgment_date` |
| `source_url` | `source` |
| `full_text` | `full_text` |

### English Court Dataset (`en-court-raw`)

Maps English court data to Weaviate schema:

| Dataset Column | Weaviate Field |
|----------------|----------------|
| `document_id` | `judgment_id` |
| `issued_on` | `date_issued` |
| `case_number` | `document_number` |
| `content` | `full_text` |

## API Reference

::: juddges.data.loaders.DatasetLoader
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: juddges.data.loaders.remap_row
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: juddges.data.loaders.DATASET_COLUMN_MAPPINGS
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## Related

- [Judgments Database](judgments_weaviate_db.md) - Database operations for judgments
- [Stream Ingester](stream_ingester.md) - Production ingestion pipeline
- [Dataset Mapper](dataset_mapper.md) - Additional mapping utilities
- [How-To: Embed and Ingest](../../../how-to/embeddings/embeddings_embed_and_ingest_weaviate.md) - End-to-end workflow

## Common Patterns

### Adding New Dataset Mapping

To add a new dataset mapping, update `DATASET_COLUMN_MAPPINGS`:

```python
DATASET_COLUMN_MAPPINGS = {
    "your-org/your-dataset": {
        "weaviate_field": "dataset_column",
        "judgment_id": "doc_id",
        "full_text": "content",
        # Add more mappings
    }
}
```

### Custom Column Remapping

For runtime column remapping:

```python
from juddges.data.loaders import remap_row

# Define custom mapping
custom_mapping = {
    "judgment_id": "custom_id_field",
    "full_text": "custom_text_field"
}

# Remap a row
remapped = remap_row(row_dict, custom_mapping)
```

## Error Handling

The loader validates datasets and provides clear error messages:

```python
try:
    loader = DatasetLoader(config)
    dataset = loader.load_document_dataset()
except AssertionError as e:
    # Embeddings directory doesn't exist
    print(f"Configuration error: {e}")
except ValueError as e:
    # Dataset is empty or not loaded correctly
    print(f"Dataset error: {e}")
except Exception as e:
    # Other loading errors
    print(f"Loading failed: {e}")
```
