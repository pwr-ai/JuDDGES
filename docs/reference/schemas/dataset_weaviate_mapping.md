# Dataset to Weaviate Mapping

This document describes how to map HuggingFace dataset records to Weaviate documents and manage raw text versions of legal documents.

## Overview

The JuDDGES project now supports:

1. **Bidirectional mapping** between HuggingFace datasets and Weaviate documents
2. **Raw text storage** in Weaviate (`raw_content` field) for unprocessed versions of judgments/tax interpretations
3. **Fast lookups** using `document_id` or `document_number` as keys
4. **Bulk updates** to populate Weaviate documents with raw text from datasets

## Architecture

### Key Components

- **`DatasetToWeaviateMapper`** ([dataset_mapper.py](../juddges/data/dataset_mapper.py)) - Core mapping utility
- **`raw_content` field** - New Weaviate property for storing raw unprocessed text
- **Column mappings** - Defined in [loaders.py](../juddges/data/loaders.py) for each dataset

### Data Flow

```
HuggingFace Dataset → DatasetToWeaviateMapper → Weaviate Document
     (judgment_id)         (index lookup)           (document_id)
     (text field)                                   (raw_content field)
```

## Schema Changes

### Weaviate LegalDocuments Collection

New property added:

```python
wvcc.Property(
    name="raw_content",
    data_type=wvcc.DataType.TEXT,
    description="Raw unprocessed text version of judgment or tax interpretation",
    skip_vectorization=True,  # Not vectorized to save resources
)
```

### LegalDocument Schema

Updated in [schemas.py](../juddges/data/schemas.py):

```python
class LegalDocument(BaseModel):
    # ...
    full_text: Optional[str]  # Processed/cleaned text
    raw_content: Optional[str]   # Raw unprocessed text (NEW)
    # ...
```

## Column Mappings

### Polish Court Judgments (`juddges/pl-court-raw`)

| Dataset Field | Weaviate Property | Description |
|--------------|-------------------|-------------|
| `judgment_id` | `document_id` | Primary identifier |
| `docket_number` | `document_number` | Secondary identifier |
| `full_text` | `full_text` | Processed text |
| `text` | `raw_content` | **Raw unprocessed text** |
| `excerpt` | `summary` | Abstract/summary |

### Tax Interpretations (`AI-TAX/pl-eureka-raw`)

| Dataset Field | Weaviate Property | Description |
|--------------|-------------------|-------------|
| `id` | `document_id` | Primary identifier |
| `docker_number` | `document_number` | Secondary identifier |
| `html_content` | `full_text` | Processed HTML content |
| `text` | `raw_content` | **Raw unprocessed text** |
| `introduction` | `summary` | Introduction section |
| `question` | `thesis` | Main question |

See [loaders.py](../juddges/data/loaders.py) for complete mappings.

## Usage

### 1. Basic Mapping Example

```python
from juddges.data import DatasetToWeaviateMapper, WeaviateLegalDocumentsDatabase

# Connect to Weaviate
db = WeaviateLegalDocumentsDatabase(
    host="localhost",
    port=8222,
    grpc_port=8085,
)

# Initialize mapper
mapper = DatasetToWeaviateMapper(
    db=db,
    dataset_name="juddges/pl-court-raw",
)

# Build index for fast lookups
mapper.build_index(
    id_field="judgment_id",
    secondary_id_field="docket_number",
)

# Get dataset record by document_id
dataset_record = mapper.get_dataset_record(
    document_id="150000000000503_I_C_001234_2020_Uz_2021-05-15_001"
)

# Get Weaviate document by document_id
weaviate_doc = mapper.get_weaviate_document(
    document_id="150000000000503_I_C_001234_2020_Uz_2021-05-15_001"
)
```

### 2. Update Raw Text in Weaviate

#### Using the Script

```bash
# Dry run to see what would be updated
python scripts/embed/update_raw_content.py \
    --dataset-name juddges/pl-court-raw \
    --raw-text-field text \
    --dry-run

# Actually update documents
python scripts/embed/update_raw_content.py \
    --dataset-name juddges/pl-court-raw \
    --raw-text-field text \
    --batch-size 100
```

#### Using the API

```python
# Update all documents with raw_content from dataset
updated_count = mapper.update_raw_content_from_dataset(
    raw_content_field="text",  # Field in dataset
    batch_size=100,
    document_id_field="judgment_id",
)

print(f"Updated {updated_count} documents")
```

### 3. Find Missing Raw Text

```python
# Find documents missing raw_content
missing = mapper.get_missing_raw_content_documents()

print(f"Found {len(missing)} documents without raw_content")
for doc in missing[:5]:
    print(f"  - {doc['document_id']}: {doc.get('title', 'No title')}")
```

### 4. Join Dataset with Weaviate

```python
# Get Weaviate documents
collection = db.legal_documents_collection
response = collection.query.fetch_objects(limit=10)
weaviate_docs = [obj.properties for obj in response.objects]

# Enrich with dataset fields
enriched = mapper.join_dataset_to_weaviate(
    weaviate_documents=weaviate_docs,
    dataset_fields=["text", "excerpt", "judgment_date"],
)

# Access enriched data
for doc in enriched:
    print(f"Document: {doc['document_id']}")
    print(f"Raw text: {doc.get('dataset_text', 'N/A')[:100]}...")
```

## Docker Usage

### Run Update Script in Container

```bash
docker compose run --rm juddges \
    python scripts/embed/update_raw_content.py \
        --dataset-name juddges/pl-court-raw \
        --raw-text-field text \
        --weaviate-host weaviate \
        --weaviate-port 8080
```

## API Reference

### `DatasetToWeaviateMapper`

#### Constructor

```python
DatasetToWeaviateMapper(
    db: WeaviateLegalDocumentsDatabase,
    dataset_name: Optional[str] = None,
    dataset: Optional[Dataset] = None,
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `build_index(id_field, secondary_id_field)` | Build lookup index |
| `get_dataset_record(document_id, document_number)` | Get dataset record |
| `get_weaviate_document(document_id, document_number)` | Get Weaviate doc |
| `join_dataset_to_weaviate(weaviate_documents, dataset_fields)` | Enrich Weaviate docs |
| `update_raw_content_from_dataset(raw_content_field, batch_size, document_id_field)` | Bulk update raw_content |
| `get_missing_raw_content_documents()` | Find docs missing raw_content |

### `WeaviateLegalDocumentsDatabase` (Extended)

#### New Filtering & Analysis Methods

| Method | Description |
|--------|-------------|
| `filter_by_raw_content_presence(has_raw_content, limit)` | Filter documents by raw_content presence |
| `get_raw_content_statistics()` | Get coverage statistics (total, with/without raw_content, %) |
| `filter_by_document_type_and_raw_content(document_type, has_raw_content, limit)` | Combined filter by type and raw_content |
| `compare_text_fields(document_id)` | Compare full_text vs raw_content for a document |
| `get_weaviate_document(document_id)` | Get document properties by ID |

#### Example Usage

```python
# Get statistics
stats = db.get_raw_content_statistics()
# Returns: {
#     "total_documents": 1000,
#     "with_raw_content": 850,
#     "without_raw_content": 150,
#     "coverage_percentage": 85.0
# }

# Filter by raw_content presence
docs_with_raw = db.filter_by_raw_content_presence(has_raw_content=True, limit=100)
docs_without_raw = db.filter_by_raw_content_presence(has_raw_content=False, limit=100)

# Combined filtering
judgments_missing_raw = db.filter_by_document_type_and_raw_content(
    document_type="judgment",
    has_raw_content=False,
    limit=50
)

# Compare text fields
comparison = db.compare_text_fields(document_id="doc-123")
# Returns: {
#     "document_id": "doc-123",
#     "has_full_text": True,
#     "has_raw_content": True,
#     "full_text_length": 5000,
#     "raw_content_length": 5500,
#     "length_difference": 500,
#     "length_ratio": 0.909
# }
```

## Performance Considerations

### Indexing

- **Build index once** at initialization for fast lookups
- Index stores all dataset records in memory
- For large datasets (>100k records), consider batch processing

### Batch Updates

- Default batch size: **100 documents**
- Adjust based on Weaviate server capacity
- Monitor memory usage during bulk updates

### Vectorization

- `raw_content` field has `skip_vectorization=True`
- Only `full_text` is vectorized for semantic search
- This saves storage and computational resources

## Common Patterns

### Pattern 1: Incremental Updates

```python
# Update only documents missing raw_content
missing = mapper.get_missing_raw_content_documents()
print(f"Updating {len(missing)} documents...")

updated = mapper.update_raw_content_from_dataset(
    raw_content_field="text",
    batch_size=50,
)
```

### Pattern 2: Dataset-Specific Mapping

```python
# Tax interpretations use different ID field
mapper_tax = DatasetToWeaviateMapper(
    db=db,
    dataset_name="AI-TAX/pl-eureka-raw",
)

mapper_tax.build_index(
    id_field="id",  # Different from judgments
    secondary_id_field="docker_number",
)
```

### Pattern 3: Cross-Reference Validation

```python
# Ensure all Weaviate docs have dataset source
collection = db.legal_documents_collection
response = collection.query.fetch_objects(limit=1000)

for obj in response.objects:
    doc_id = obj.properties.get("document_id")
    dataset_record = mapper.get_dataset_record(document_id=doc_id)

    if not dataset_record:
        print(f"Warning: {doc_id} not found in dataset")
```

## Troubleshooting

### Issue: Index Not Built

```
RuntimeError: Index not built. Call build_index() first.
```

**Solution:** Always call `build_index()` before using lookup methods.

### Issue: Dataset Record Not Found

**Causes:**
1. Incorrect ID field names
2. Mismatch between dataset and Weaviate IDs
3. Dataset not fully loaded

**Solution:** Verify ID fields and rebuild index.

### Issue: Slow Updates

**Solution:** Increase batch size or use parallel processing:

```python
mapper.update_raw_content_from_dataset(
    raw_content_field="text",
    batch_size=500,  # Increase batch size
)
```

## Migration Guide

### Updating Existing Weaviate Collections

If you have existing collections without `raw_content`:

1. **Add the field** by recreating the collection (or use Weaviate schema migration)
2. **Run update script** to populate raw_content from datasets
3. **Verify** all documents have raw_content

```bash
# Step 1: Recreate collection (optional, field is added automatically)
python scripts/embed/create_weaviate_collections.py

# Step 2: Update raw_content
python scripts/embed/update_raw_content.py \
    --dataset-name juddges/pl-court-raw \
    --raw-text-field text

# Step 3: Verify
python scripts/embed/analyze_raw_content_coverage.py
```

## Filtering and Analysis

### Analyze Coverage

```bash
# Get overall statistics
python scripts/embed/analyze_raw_content_coverage.py

# Show coverage by document type
python scripts/embed/analyze_raw_content_coverage.py --by-type

# Compare text fields for specific document
python scripts/embed/analyze_raw_content_coverage.py \
    --compare-document "150000000000503_I_C_001234_2020_Uz_2021-05-15_001"
```

### Filter Documents by raw_content Presence

```python
from juddges.data import WeaviateLegalDocumentsDatabase

db = WeaviateLegalDocumentsDatabase(host="localhost", port=8222, grpc_port=8085)

# Get documents WITH raw_content
with_raw = db.filter_by_raw_content_presence(has_raw_content=True, limit=100)
print(f"Found {len(with_raw)} documents with raw_content")

# Get documents WITHOUT raw_content
without_raw = db.filter_by_raw_content_presence(has_raw_content=False, limit=100)
print(f"Found {len(without_raw)} documents missing raw_content")
```

### Get Coverage Statistics

```python
stats = db.get_raw_content_statistics()
print(f"Coverage: {stats['coverage_percentage']}%")
print(f"With raw_content: {stats['with_raw_content']}/{stats['total_documents']}")
```

### Filter by Document Type and raw_content

```python
# Get judgments WITH raw_content
judgments_with_raw = db.filter_by_document_type_and_raw_content(
    document_type="judgment",
    has_raw_content=True,
    limit=100
)

# Get tax interpretations WITHOUT raw_content
tax_without_raw = db.filter_by_document_type_and_raw_content(
    document_type="tax_interpretation",
    has_raw_content=False,
    limit=100
)
```

### Compare Text Fields

```python
# Analyze differences between full_text and raw_content
comparison = db.compare_text_fields(document_id="your-doc-id")

if comparison:
    print(f"Full text: {comparison['full_text_length']} chars")
    print(f"Raw text: {comparison['raw_content_length']} chars")
    print(f"Ratio: {comparison['length_ratio']}")
```

## Examples

Full examples available in:
- [map_dataset_to_weaviate.py](../scripts/examples/map_dataset_to_weaviate.py)
- [update_raw_content.py](../scripts/embed/update_raw_content.py)
- [analyze_raw_content_coverage.py](../scripts/embed/analyze_raw_content_coverage.py)

## See Also

- [Weaviate Schema Documentation](../juddges/data/documents_weaviate_db.py)
- [Dataset Loaders](../juddges/data/loaders.py)
- [Embedding Pipeline](./embedding_pipeline.md)
