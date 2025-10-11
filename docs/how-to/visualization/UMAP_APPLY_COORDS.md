# UMAP Coordinate Application Guide

## Overview

The script `scripts/embed/apply_umap_to_missing_coords.py` efficiently applies a saved UMAP model to documents with missing x,y coordinates in Weaviate using an optimized O(n) single-pass algorithm.

## Quick Start

### Prerequisites

```bash
# Start Weaviate (if not already running)
cd weaviate/
docker compose up -d
cd ..
```

### Basic Usage

**Test mode (1000 documents):**

```bash
docker compose run --rm web python scripts/embed/apply_umap_to_missing_coords.py \
    --model-path models/umap/umap_model_LegalDocuments.pkl \
    --collection LegalDocuments \
    --test-mode \
    --test-sample-size 1000
```

**Production mode (all documents):**

```bash
docker compose run --rm web python scripts/embed/apply_umap_to_missing_coords.py \
    --model-path models/umap/umap_model_LegalDocuments.pkl \
    --collection LegalDocuments \
    --process-batch-size 1000
```

**Dry run (preview without updating):**

```bash
docker compose run --rm web python scripts/embed/apply_umap_to_missing_coords.py \
    --model-path models/umap/umap_model_LegalDocuments.pkl \
    --collection LegalDocuments \
    --test-mode \
    --dry-run
```

## Performance

- **Algorithm**: O(n) single-pass iterator
- **Expected Speed**: ~50 minutes for 100K documents
- **Memory**: Configurable via `--process-batch-size` (default: 1000)

## Configuration

### Required

- `--model-path`: Path to saved UMAP pickle
- `--collection`: `LegalDocuments` or `DocumentChunks`

### Optional

- `--process-batch-size`: Docs per batch (default: 1000)
- `--test-mode`: Process limited sample
- `--test-sample-size`: Sample size (default: 1000)
- `--dry-run`: Preview only

## Monitoring Progress

Check how many documents have coordinates:

```bash
docker compose run --rm web python scripts/embed/query_coords.py
```

This will show:

- Sample documents with coordinates from each collection
- Total count of documents with x,y coordinates set

You can also use it programmatically:

```python
from scripts.embed.query_coords import count_documents_with_coordinates
from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase

with WeaviateLegalDocumentsDatabase() as db:
    count = count_documents_with_coordinates(db, "LegalDocuments")
    print(f"Documents with coords: {count}")
```

## Related Scripts

- `scripts/embed/sample_and_calculate_umap.py` - Create UMAP models
- `scripts/embed/query_coords.py` - Query and count documents with coordinates
