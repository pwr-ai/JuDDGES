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

## Related Scripts

- `scripts/embed/sample_and_calculate_umap.py` - Create UMAP models
- `scripts/embed/query_coords.py` - Query documents with coordinates
