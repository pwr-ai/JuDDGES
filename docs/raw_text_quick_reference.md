# raw_content Quick Reference

Quick commands and snippets for working with the `raw_content` field in Weaviate.

## Quick Commands

### Analyze Coverage

```bash
# Basic statistics
python scripts/embed/analyze_raw_content_coverage.py

# With document type breakdown
python scripts/embed/analyze_raw_content_coverage.py --by-type

# Show first 20 missing documents
python scripts/embed/analyze_raw_content_coverage.py --show-missing 20
```

### Update raw_content

```bash
# Dry run (preview changes)
python scripts/embed/update_raw_content.py \
    --dataset-name juddges/pl-court-raw \
    --dry-run

# Update Polish court judgments
python scripts/embed/update_raw_content.py \
    --dataset-name juddges/pl-court-raw \
    --raw-text-field text

# Update tax interpretations
python scripts/embed/update_raw_content.py \
    --dataset-name AI-TAX/pl-eureka-raw \
    --raw-text-field text
```

## Quick Snippets

### Python - Get Statistics

```python
from juddges.data import WeaviateLegalDocumentsDatabase

db = WeaviateLegalDocumentsDatabase(host="localhost", port=8222, grpc_port=50051)
stats = db.get_raw_content_statistics()

print(f"Coverage: {stats['coverage_percentage']}%")
print(f"{stats['with_raw_content']} / {stats['total_documents']} documents")
```

### Python - Find Missing Documents

```python
missing = db.filter_by_raw_content_presence(has_raw_content=False, limit=10)

for doc in missing:
    print(f"{doc['document_id']}: {doc.get('title', 'No title')}")
```

### Python - Update from Dataset

```python
from juddges.data import DatasetToWeaviateMapper

mapper = DatasetToWeaviateMapper(db=db, dataset_name="juddges/pl-court-raw")
mapper.build_index(id_field="judgment_id", secondary_id_field="docket_number")

# Update all missing
updated = mapper.update_raw_content_from_dataset(raw_content_field="text")
print(f"Updated {updated} documents")
```

### Python - Compare Text Fields

```python
# Single document
comparison = db.compare_text_fields(document_id="your-doc-id")
print(f"Full: {comparison['full_text_length']} chars")
print(f"Raw:  {comparison['raw_content_length']} chars")
print(f"Ratio: {comparison['length_ratio']}")

# Batch comparison
docs = db.filter_by_raw_content_presence(has_raw_content=True, limit=100)
for doc in docs:
    comp = db.compare_text_fields(doc['document_id'])
    if comp and comp['length_ratio'] and comp['length_ratio'] < 0.8:
        print(f"Large difference in {doc['document_id']}")
```

## Docker Commands

```bash
# Run analysis in container
docker compose run --rm juddges \
    python scripts/embed/analyze_raw_content_coverage.py \
        --weaviate-host weaviate \
        --weaviate-port 8080

# Update raw_content in container
docker compose run --rm juddges \
    python scripts/embed/update_raw_content.py \
        --dataset-name juddges/pl-court-raw \
        --weaviate-host weaviate \
        --weaviate-port 8080
```

## Common Workflows

### Initial Setup

```bash
# 1. Analyze current state
python scripts/embed/analyze_raw_content_coverage.py

# 2. Update judgments
python scripts/embed/update_raw_content.py \
    --dataset-name juddges/pl-court-raw \
    --raw-text-field text

# 3. Update tax interpretations
python scripts/embed/update_raw_content.py \
    --dataset-name AI-TAX/pl-eureka-raw \
    --raw-text-field text

# 4. Verify coverage
python scripts/embed/analyze_raw_content_coverage.py --by-type
```

### Quality Check

```python
from juddges.data import WeaviateLegalDocumentsDatabase

db = WeaviateLegalDocumentsDatabase(host="localhost", port=8222, grpc_port=50051)

# Check judgments
judgments_missing = db.filter_by_document_type_and_raw_content(
    document_type="judgment",
    has_raw_content=False,
    limit=1000
)

# Check tax interpretations
tax_missing = db.filter_by_document_type_and_raw_content(
    document_type="tax_interpretation",
    has_raw_content=False,
    limit=1000
)

print(f"Judgments missing raw_content: {len(judgments_missing)}")
print(f"Tax interpretations missing raw_content: {len(tax_missing)}")
```

### Incremental Updates

```python
# Only update documents without raw_content
from juddges.data import DatasetToWeaviateMapper, WeaviateLegalDocumentsDatabase

db = WeaviateLegalDocumentsDatabase(host="localhost", port=8222, grpc_port=50051)
mapper = DatasetToWeaviateMapper(db=db, dataset_name="juddges/pl-court-raw")
mapper.build_index(id_field="judgment_id", secondary_id_field="docket_number")

# Get missing count before
stats_before = db.get_raw_content_statistics()
print(f"Before: {stats_before['without_raw_content']} missing")

# Update
updated = mapper.update_raw_content_from_dataset(raw_content_field="text", batch_size=100)

# Get missing count after
stats_after = db.get_raw_content_statistics()
print(f"After: {stats_after['without_raw_content']} missing")
print(f"Updated: {updated} documents")
```

## Troubleshooting

### No documents updated

```python
# Check if dataset has the field
from datasets import load_dataset
ds = load_dataset("juddges/pl-court-raw", split="train")
print(ds.column_names)  # Should include "text"
print(ds[0].keys())     # Check first record
```

### Index build fails

```python
# Verify ID fields exist in dataset
mapper = DatasetToWeaviateMapper(db=db, dataset_name="juddges/pl-court-raw")
print(mapper.dataset[0].keys())  # Check available fields

# Use correct ID field
mapper.build_index(
    id_field="judgment_id",  # Must exist in dataset
    secondary_id_field="docket_number"
)
```

### Slow updates

```python
# Increase batch size
mapper.update_raw_content_from_dataset(
    raw_content_field="text",
    batch_size=500,  # Larger batches = faster (but more memory)
)
```

## Dataset-Specific Field Names

| Dataset | ID Field | Secondary ID | raw_content Source |
|---------|----------|--------------|-----------------|
| `juddges/pl-court-raw` | `judgment_id` | `docket_number` | `text` |
| `AI-TAX/pl-eureka-raw` | `id` | `docker_number` | `text` |
| `AI-TAX/pl-eureka-raw-sample` | `id` | `docker_number` | `text` |
| `en-court-raw` | `judgment_id` | `case_number` | `text` |

## See Also

- [Full Documentation](./dataset_weaviate_mapping.md)
- [Weaviate Schema](../juddges/data/documents_weaviate_db.py)
- [Dataset Loaders](../juddges/data/loaders.py)
