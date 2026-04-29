# UMAP Embeddings Sampling

## Overview

This document describes the process of sampling embeddings from Weaviate for UMAP calculation and visualization. The sampling strategy uses **stratified sampling** by `country` and `source_url` to ensure representative coverage across different data sources.

## Sampling Process

### Script: `sample_and_calculate_umap.py`

The sampling script queries Weaviate collections directly and creates stratified samples for UMAP computation:

**Features:**

- Samples up to 2,500 documents per unique combination of stratification fields (configurable)
- Queries documents directly from Weaviate with their embeddings
- Processes both LegalDocuments and DocumentChunks collections
- Saves sampled data with vectors to parquet files
- Fits UMAP model on sampled data and saves it for later use
- Updates Weaviate documents with calculated (x, y) coordinates

### Running the Sampler

**Step 1: Sample and Calculate UMAP Coordinates**

```bash
# Sample and fit UMAP for both collections (uses default sample-size of 2500)
docker compose run --rm web python scripts/embed/sample_and_calculate_umap.py \
    --output-dir data/embeddings_samples \
    --collection both \
    --n-neighbors 15 \
    --min-dist 0.1

# Or for a specific collection with custom sample size
docker compose run --rm web python scripts/embed/sample_and_calculate_umap.py \
    --output-dir data/embeddings_samples \
    --sample-size 5000 \
    --collection LegalDocuments
```

**Step 2: Apply UMAP Model to All Remaining Documents**

After fitting UMAP on sampled data, apply it to all documents:

```bash
# Apply saved UMAP model to all documents
docker compose run --rm web python scripts/embed/sample_and_calculate_umap.py \
    --apply-saved-model data/embeddings_samples/umap_model.pkl \
    --collection LegalDocuments

docker compose run --rm web python scripts/embed/sample_and_calculate_umap.py \
    --apply-saved-model data/embeddings_samples/umap_model.pkl \
    --collection DocumentChunks
```

### Output Structure

Sampled data is saved to `data/embeddings_samples/`:

```
data/embeddings_samples/
├── LegalDocuments_sampled.parquet    # Sampled documents with vectors
├── DocumentChunks_sampled.parquet    # Sampled chunks with vectors
└── umap_model.pkl                     # Fitted UMAP model (saved as pickle)
```

Each parquet file contains:

- `uuid`: Document/chunk UUID
- `vector`: Embedding vector (array)
- For LegalDocuments: `country`, `source_url`
- For DocumentChunks: `language`, `document_type`

### Stratified Sampling Strategy

The script uses different stratification fields depending on the collection:

#### LegalDocuments Collection

Samples up to 2,500 documents (default) for each unique combination of:

- **Country**: `country` field (e.g., "Poland", "Switzerland")
- **Source URL**: `source_url` field (different court systems)

**Example strata:**

- `Poland|saos-api` → up to 2,500 documents
- `Poland|court-decisions-api` → up to 2,500 documents
- `Switzerland|federal-court` → up to 2,500 documents

#### DocumentChunks Collection

Samples up to 2,500 chunks (default) for each unique combination of:

- **Language**: `language` field (e.g., "pl", "en")
- **Document Type**: `document_type` field (e.g., "judgment", "tax_interpretation")

**Example strata:**

- `pl|judgment` → up to 2,500 chunks
- `en|judgment` → up to 2,500 chunks
- `pl|tax_interpretation` → up to 2,500 chunks

This ensures:

- ✅ Balanced representation across countries/languages
- ✅ Balanced representation across data sources/document types
- ✅ Coverage of different court systems
- ✅ Prevents dominance by any single source

## UMAP Configuration

Default UMAP parameters:

- `n_neighbors`: 15 (balances local vs global structure)
- `min_dist`: 0.1 (minimum distance between points)
- `metric`: cosine (suitable for embeddings)
- `n_components`: 2 (2D visualization)
- `random_state`: 42 (reproducibility)

## Complete Workflow

### 1. Sample and Fit UMAP (First Time)

```bash
# Sample documents from Weaviate, fit UMAP, and update sampled documents
docker compose run --rm web python scripts/embed/sample_and_calculate_umap.py \
    --output-dir data/embeddings_samples \
    --collection both \
    --n-neighbors 15 \
    --min-dist 0.1 \
    --batch-size 500
```

This will:

1. ✅ Query Weaviate and sample up to 2.5k documents per stratum (default)
2. ✅ Save sampled data with vectors to `data/embeddings_samples/`
3. ✅ Normalize vectors using L2 normalization
4. ✅ Fit UMAP model on sampled data
5. ✅ Save UMAP model to `data/embeddings_samples/umap_model.pkl`
6. ✅ Calculate (x, y) coordinates for sampled documents
7. ✅ Update sampled documents in Weaviate with coordinates

### 2. Apply UMAP to All Remaining Documents

```bash
# Apply saved UMAP model to all LegalDocuments
docker compose run --rm web python scripts/embed/sample_and_calculate_umap.py \
    --apply-saved-model data/embeddings_samples/umap_model.pkl \
    --collection LegalDocuments \
    --batch-size 500

# Apply saved UMAP model to all DocumentChunks
docker compose run --rm web python scripts/embed/sample_and_calculate_umap.py \
    --apply-saved-model data/embeddings_samples/umap_model.pkl \
    --collection DocumentChunks \
    --batch-size 500
```

This will:

1. ✅ Load the saved UMAP model
2. ✅ Fetch ALL documents from Weaviate with their vectors
3. ✅ Normalize vectors
4. ✅ Transform vectors to 2D coordinates using the saved UMAP model
5. ✅ Update ALL documents in Weaviate with (x, y) coordinates

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir` | Directory to save sampled data and UMAP model | `data/embeddings_samples` |
| `--sample-size` | Max documents per stratum | `2500` |
| `--collection` | Collection to process (`LegalDocuments`, `DocumentChunks`, `both`) | `both` |
| `--vector-name` | Name of vector to use | `base` |
| `--batch-size` | Batch size for Weaviate updates | `500` |
| `--n-neighbors` | UMAP n_neighbors parameter | `15` |
| `--min-dist` | UMAP min_dist parameter | `0.1` |
| `--apply-saved-model` | Path to saved UMAP model (skip sampling) | None |

## Benefits of This Approach

1. **Stratified Sampling**: Ensures balanced representation across countries and sources
2. **Saved UMAP Model**: Can apply the same transformation to new documents later
3. **Two-Stage Process**:
   - Fit UMAP on representative sample (faster)
   - Apply to all documents (consistent coordinates)
4. **Direct Weaviate Integration**: No need for intermediate parquet files
5. **Automatic Updates**: Directly updates Weaviate documents with coordinates

## Notes

- UMAP fitting uses `fit_transform()` on sampled data
- Applying to all documents uses `transform()` on the saved model
- The same UMAP model ensures consistent coordinate space
- Files created by Docker may have root ownership; use `sudo rm -rf` if needed
- The default 2.5k per-stratum limit balances efficiency with representativeness
- Increase `--sample-size` for larger samples if needed
- Pagination batch size is limited to 100 to avoid Weaviate query errors
- L2 normalization is applied before UMAP transformation
