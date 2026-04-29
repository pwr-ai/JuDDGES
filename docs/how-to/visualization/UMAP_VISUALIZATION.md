# UMAP 2D Visualization Coordinates

This document describes how to calculate and update UMAP 2D coordinates for documents and chunks stored in Weaviate for visualization purposes.

## Overview

The UMAP (Uniform Manifold Approximation and Projection) algorithm reduces high-dimensional embeddings to 2D coordinates suitable for visualization. This allows you to:

- Visualize document clusters and relationships
- Explore semantic similarity in 2D space
- Identify patterns and groupings in legal documents
- Create interactive visualizations for exploration

## Prerequisites

1. **Install dependencies**:

   ```bash
   uv pip install -e ".[full]"
   ```

2. **Generated embeddings** in parquet format:
   - Run the embedding pipeline to create `chunk_embeddings` and `agg_embeddings` directories
   - These should be in a path like `data/embeddings/{dataset-name}/`

3. **Weaviate running** (for updating coordinates):

   ```bash
   cd weaviate
   docker compose up -d
   ```

## Two-Step Workflow (Recommended)

The recommended approach is to calculate and save coordinates first, then update Weaviate separately. This allows you to:

- Review coordinates before updating
- Reuse coordinates for multiple updates
- Separate computation from database updates

### Step 1: Calculate and Save Coordinates

Calculate UMAP coordinates from embeddings and save to parquet files:

```bash
# Calculate for both collections and save
docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
    --embeddings-dir data/embeddings/pl-court-raw-sample \
    --output-dir umap_coords \
    --skip-update

# This creates:
# - umap_coords/LegalDocuments_coords.parquet
# - umap_coords/DocumentChunks_coords.parquet
```

Each parquet file contains three columns:

- `uuid`: The deterministic UUID for the document/chunk
- `x`: The x coordinate from UMAP
- `y`: The y coordinate from UMAP

### Step 2: Update Weaviate

Load the saved coordinates and update Weaviate:

```bash
# Update LegalDocuments collection
docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
    --load-from umap_coords/LegalDocuments_coords.parquet \
    --collection LegalDocuments

# Update DocumentChunks collection
docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
    --load-from umap_coords/DocumentChunks_coords.parquet \
    --collection DocumentChunks
```

## One-Step Workflow (Alternative)

Calculate and update in a single step:

```bash
# Calculate and update both collections
docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
    --embeddings-dir data/embeddings/pl-court-raw-sample

# Calculate and update specific collection
docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
    --embeddings-dir data/embeddings/pl-court-raw-sample \
    --collection LegalDocuments
```

## Additional Options

### Adjust UMAP Parameters

```bash
# More local structure (smaller neighborhoods)
docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
    --embeddings-dir data/embeddings/pl-court-raw-sample \
    --output-dir umap_coords \
    --skip-update \
    --n-neighbors 5 --min-dist 0.01

# More global structure (larger neighborhoods)
docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
    --embeddings-dir data/embeddings/pl-court-raw-sample \
    --output-dir umap_coords \
    --skip-update \
    --n-neighbors 50 --min-dist 0.5
```

### Dry Run Mode

Preview changes without updating Weaviate:

```bash
docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
    --load-from umap_coords/LegalDocuments_coords.parquet \
    --collection LegalDocuments \
    --dry-run
```

### Custom Batch Size

```bash
# Larger batches for faster processing (if memory allows)
docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
    --load-from umap_coords/LegalDocuments_coords.parquet \
    --collection LegalDocuments \
    --batch-size 1000
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--embeddings-dir` | str | - | Path to directory containing embedding parquet files (required for calculation mode) |
| `--load-from` | str | - | Load previously calculated coordinates from parquet file (alternative to --embeddings-dir) |
| `--output-dir` | str | - | Directory to save calculated coordinates (creates `{collection}_coords.parquet` files) |
| `--collection` | str | `both` | Collection to process: `LegalDocuments`, `DocumentChunks`, or `both` |
| `--skip-update` | flag | False | Only calculate and save coordinates, don't update Weaviate |
| `--batch-size` | int | 500 | Number of documents to update per batch |
| `--n-neighbors` | int | 15 | UMAP n_neighbors parameter (controls local vs global structure) |
| `--min-dist` | float | 0.1 | UMAP min_dist parameter (controls point spacing) |
| `--dry-run` | flag | False | Preview changes without updating Weaviate |

## How It Works

### Mode 1: Calculate from Embeddings

The script performs these phases:

1. **Load Embeddings from Parquet Files**
   - Loads embeddings from `chunk_embeddings/` or `agg_embeddings/` subdirectories
   - Generates deterministic UUIDs based on document IDs (and chunk IDs for chunks)
   - Extracts embedding vectors from the parquet files

2. **Normalize Vectors**
   - All vectors are L2-normalized before UMAP computation
   - Ensures consistent cosine distance interpretation
   - Better UMAP clustering results

3. **Compute UMAP Coordinates**

   ```python
   from umap import UMAP

   umap_model = UMAP(
       n_neighbors=15,      # Controls local vs global structure
       min_dist=0.1,        # Controls point spacing
       metric='cosine',     # Appropriate for normalized embeddings
       random_state=42,     # For reproducibility
       n_components=2       # 2D output
   )

   coordinates = umap_model.fit_transform(normalized_embeddings)
   ```

4. **Save Coordinates (Optional)**
   - If `--output-dir` is specified, saves to `{collection}_coords.parquet`
   - Format: `uuid`, `x`, `y` columns

5. **Update Weaviate (Optional)**
   - If `--skip-update` is not specified, updates documents in Weaviate
   - Only updates the `x` and `y` properties

### Mode 2: Load from Saved Coordinates

1. **Load Coordinates**
   - Reads parquet file with `uuid`, `x`, `y` columns
   - Validates file exists and contains required columns

2. **Update Weaviate**
   - Updates documents with loaded coordinates
   - Uses deterministic UUIDs to match documents

## UMAP Parameter Tuning

### n_neighbors

Controls the balance between local and global structure:

- **Small values (5-10)**: Focus on local structure, tighter clusters
- **Medium values (15-30)**: Balanced view (recommended)
- **Large values (50+)**: Focus on global structure, broader patterns

### min_dist

Controls how tightly UMAP packs points:

- **Small values (0.01-0.05)**: Tight clusters, minimal spacing
- **Medium values (0.1-0.3)**: Balanced spacing (recommended)
- **Large values (0.5+)**: Looser clusters, more spread out

### metric

We use `cosine` distance because:

- Embeddings represent semantic meaning
- After L2 normalization, cosine distance is most appropriate
- Works well for high-dimensional text embeddings

## Weaviate Schema

The script updates these properties in both collections:

```python
# In LegalDocuments and DocumentChunks collections
wvcc.Property(
    name="x",
    data_type=wvcc.DataType.NUMBER,
    description="X coordinate for visualization",
    skip_vectorization=True,
),
wvcc.Property(
    name="y",
    data_type=wvcc.DataType.NUMBER,
    description="Y coordinate for visualization",
    skip_vectorization=True,
),
```

## Example Workflow

Complete workflow from embedding generation to visualization:

```bash
# 1. Generate embeddings (using DVC or embedding scripts)
dvc repro embed

# 2. Ingest documents to Weaviate (if not already done)
docker compose run --rm web python scripts/embed/simple_ingest.py \
    configs/datasets/JuDDGES_pl-court-raw-sample.yaml

# 3. Calculate UMAP coordinates and save
docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
    --embeddings-dir data/embeddings/pl-court-raw-sample \
    --output-dir umap_coords \
    --skip-update

# 4. Review saved coordinates (optional)
python -c "
import polars as pl
df = pl.read_parquet('umap_coords/LegalDocuments_coords.parquet')
print(df.head())
print(f'X range: [{df[\"x\"].min():.2f}, {df[\"x\"].max():.2f}]')
print(f'Y range: [{df[\"y\"].min():.2f}, {df[\"y\"].max():.2f}]')
"

# 5. Update Weaviate with saved coordinates
docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
    --load-from umap_coords/LegalDocuments_coords.parquet \
    --collection LegalDocuments

docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
    --load-from umap_coords/DocumentChunks_coords.parquet \
    --collection DocumentChunks

# 6. Query documents with coordinates
docker compose run --rm web python -c "
from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase

with WeaviateLegalDocumentsDatabase() as db:
    collection = db.legal_documents_collection

    # Fetch first 10 documents with coordinates
    response = collection.query.fetch_objects(
        limit=10,
        return_properties=['document_id', 'title', 'x', 'y']
    )

    for obj in response.objects:
        props = obj.properties
        print(f'{props[\"title\"][:50]}: ({props[\"x\"]:.2f}, {props[\"y\"]:.2f})')
"
```

## Performance Considerations

### Memory Usage

- Loading embeddings loads all vectors into memory
- For large collections (>100k documents), ensure sufficient RAM
- Embedding dimension × number of documents = memory required
  - Example: 768 dims × 100,000 docs × 4 bytes = ~300MB

### Processing Time

Approximate times (on standard CPU):

| Documents | Load Embeddings | UMAP | Save/Update | Total |
|-----------|----------------|------|-------------|-------|
| 10,000 | ~30 sec | ~2 min | ~30 sec | ~3 min |
| 50,000 | ~2 min | ~10 min | ~2 min | ~14 min |
| 100,000 | ~4 min | ~30 min | ~4 min | ~38 min |

### Optimization Tips

1. **Use two-step workflow**: Calculate once, update multiple times if needed
2. **Process one collection at a time**: Use `--collection` to avoid memory issues
3. **Save coordinates**: Use `--output-dir` to preserve calculations
4. **Adjust batch size**: Default (500) is optimized for most cases
5. **Run in Docker**: Ensures consistent environment and resource allocation

## Troubleshooting

### No embeddings extracted

**Problem**: Script reports "No embeddings extracted"

**Solutions**:

- Verify embeddings directory exists and contains parquet files
- Check that `chunk_embeddings/` or `agg_embeddings/` subdirectories exist
- Ensure embedding generation was successful (check DVC pipeline or embedding scripts)
- Verify parquet files contain `embedding` or `embeddings` field

### Memory errors

**Problem**: Out of memory during UMAP computation

**Solutions**:

- Process collections separately: `--collection LegalDocuments`
- Use `--skip-update` to save coordinates without immediate Weaviate update
- Increase Docker memory limit
- Consider processing in smaller batches (requires script modification)

### Update failures

**Problem**: Some documents fail to update

**Solutions**:

- Check Weaviate logs for errors
- Verify UUIDs in saved coordinates match documents in Weaviate
- Ensure sufficient Weaviate resources
- Check network connectivity
- Use `--dry-run` to preview before actual update

### Coordinates file not found

**Problem**: Cannot load coordinates from saved file

**Solutions**:

- Verify the file path is correct
- Ensure Step 1 (calculation) completed successfully
- Check that the parquet file contains `uuid`, `x`, `y` columns
- Verify file permissions

### Inconsistent coordinates

**Problem**: Running calculation multiple times gives different coordinates

**Solutions**:

- This is expected - UMAP has random initialization
- The script uses `random_state=42` for reproducibility within a single run
- Coordinates are relative, not absolute
- Save coordinates after calculation to reuse the same projection

## Visualization Examples

After calculating coordinates, you can visualize them using various tools:

### Python with Matplotlib

```python
import matplotlib.pyplot as plt
from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase

with WeaviateLegalDocumentsDatabase() as db:
    collection = db.legal_documents_collection

    response = collection.query.fetch_objects(
        limit=1000,
        return_properties=['document_type', 'x', 'y']
    )

    # Extract coordinates and types
    points = [(obj.properties['x'], obj.properties['y'], obj.properties['document_type'])
              for obj in response.objects if obj.properties.get('x') is not None]

    x = [p[0] for p in points]
    y = [p[1] for p in points]
    types = [p[2] for p in points]

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, c=pd.Categorical(types).codes, alpha=0.5, cmap='tab10')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('Legal Documents - UMAP Projection')
    plt.colorbar(label='Document Type')
    plt.savefig('umap_visualization.png', dpi=300, bbox_inches='tight')
```

### Interactive with Plotly

```python
import plotly.express as px
import pandas as pd
from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase

with WeaviateLegalDocumentsDatabase() as db:
    collection = db.legal_documents_collection

    response = collection.query.fetch_objects(
        limit=5000,
        return_properties=['document_id', 'title', 'document_type', 'x', 'y']
    )

    # Convert to DataFrame
    data = []
    for obj in response.objects:
        props = obj.properties
        if props.get('x') is not None:
            data.append({
                'x': props['x'],
                'y': props['y'],
                'type': props['document_type'],
                'title': props['title'][:100],
                'id': props['document_id']
            })

    df = pd.DataFrame(data)

    # Create interactive plot
    fig = px.scatter(
        df, x='x', y='y', color='type',
        hover_data=['title', 'id'],
        title='Legal Documents UMAP Visualization'
    )

    fig.update_traces(marker=dict(size=5, opacity=0.6))
    fig.write_html('umap_interactive.html')
```

### Analyze Saved Coordinates

```python
import polars as pl

# Load saved coordinates
df = pl.read_parquet('umap_coords/LegalDocuments_coords.parquet')

# Basic statistics
print(f"Total coordinates: {len(df)}")
print(f"X range: [{df['x'].min():.2f}, {df['x'].max():.2f}]")
print(f"Y range: [{df['y'].min():.2f}, {df['y'].max():.2f}]")

# Preview
print(df.head())
```

## Advanced Topics

### Incremental Updates

For new documents, you have several options:

1. **Recalculate all coordinates**: Ensures consistent projection but requires full recalculation
2. **Calculate separately**: Create separate UMAP for new batch (coordinates won't align with existing)
3. **Save UMAP model**: Use a saved UMAP model to transform new documents (requires script modification)

### Custom Coordinate Scaling

Normalize coordinates to specific range:

```python
import polars as pl
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load coordinates
df = pl.read_parquet('umap_coords/LegalDocuments_coords.parquet')

# Scale to 0-100 range
coords = np.column_stack([df['x'].to_numpy(), df['y'].to_numpy()])
scaler = MinMaxScaler(feature_range=(0, 100))
scaled_coords = scaler.fit_transform(coords)

# Save scaled coordinates
df_scaled = pl.DataFrame({
    'uuid': df['uuid'],
    'x': scaled_coords[:, 0],
    'y': scaled_coords[:, 1]
})
df_scaled.write_parquet('umap_coords/LegalDocuments_coords_scaled.parquet')
```

### Multiple UMAP Projections

Calculate different projections for different use cases:

```bash
# Local structure (for detailed exploration)
docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
    --embeddings-dir data/embeddings/pl-court-raw-sample \
    --output-dir umap_coords/local \
    --skip-update \
    --n-neighbors 5 --min-dist 0.01

# Global structure (for overview)
docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
    --embeddings-dir data/embeddings/pl-court-raw-sample \
    --output-dir umap_coords/global \
    --skip-update \
    --n-neighbors 50 --min-dist 0.5
```

## References

- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [Understanding UMAP](https://pair-code.github.io/understanding-umap/)
- [Weaviate Named Vectors](https://weaviate.io/developers/weaviate/config-refs/schema/multi-vector)
- [Scikit-learn Normalization](https://scikit-learn.org/stable/modules/preprocessing.html#normalization)
