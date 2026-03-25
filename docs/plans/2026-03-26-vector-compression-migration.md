# Vector Compression Migration: 50-100x Storage Reduction

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace current 1024-dim float32 vectors (~260 GB) with 128-dim binary-quantized vectors (~3-6 GB) while maintaining hybrid search quality.

**Architecture:** Switch from `sdadas/mmlw-roberta-large` (1024-dim, no Matryoshka) to `intfloat/multilingual-e5-small` (384-dim, Matryoshka-capable, truncated to 128-dim). Enable Weaviate Binary Quantization (BQ). Increase chunk size from 512 to 2048 chars to reduce DocumentChunks count by ~4x. Remove named vectors (base/dev/fast) in favor of a single vector. Keep BM25 as primary search, vector as fallback.

**Tech Stack:** Weaviate 1.30.2, sentence-transformers, intfloat/multilingual-e5-small, Binary Quantization, Python 3.13

---

## Pre-Migration Checklist

- [x] Native backup on NAS: `backup-20260324-224720` (345 GB, verified SUCCESS)
- [x] Parquet dump on NAS: `LegalDocuments_20260107.parquet` + `DocumentChunks_20260107.parquet` (67 GB)
- [ ] Verify NAS mount is accessible: `ls /mnt/readynas/datasets/legal-ai-weaviate/native-backups/`

---

### Task 1: Add new embedding model config

**Files:**
- Create: `configs/embedding_model/multilingual-e5-small.yaml`
- Modify: `juddges/settings.py`

**Step 1: Create embedding model config**

```yaml
name: intfloat/multilingual-e5-small
max_seq_length: 512
embedding_dim: 384
matryoshka_dim: 128
```

**Step 2: Update settings.py — replace model and simplify VectorName**

In `juddges/settings.py`, change:

```python
TEXT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
TEXT_EMBEDDING_DIM = 128  # Matryoshka truncation from 384

class VectorName:
    DEFAULT = "default"
```

Remove old `BASE`, `DEV`, `FAST` constants.

**Step 3: Run grep to find all VectorName.BASE/DEV/FAST references**

Run: `grep -rn "VectorName\." juddges/ scripts/ --include="*.py" | grep -v __pycache__`

Update each reference to use `VectorName.DEFAULT`.

**Step 4: Commit**

```bash
git add configs/embedding_model/multilingual-e5-small.yaml juddges/settings.py
git commit -m "feat: switch to multilingual-e5-small with 128-dim Matryoshka"
```

---

### Task 2: Update chunking config for larger chunks

**Files:**
- Modify: `configs/embedding.yaml`
- Modify: `configs/embedding_model/multilingual-e5-small.yaml`

**Step 1: Update embedding.yaml chunk_config**

Change `configs/embedding.yaml`:

```yaml
defaults:
  - embedding_model: multilingual-e5-small
  - _self_
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

dataset_name: ???
chunk_config:
  chunk_size: 2048
  min_split_chars: 100
  take_n_first_chunks: 8
  chunk_overlap: 256

batch_size: 4096
num_output_shards: 10
ingest_batch_size: 64
upsert: false

output_dir: null

hydra:
  output_subdir: null
  run:
    dir: .

default_column_values:
  country: "Poland"
  language: "pl"
  document_type: "judgment"
```

Key changes:
- `chunk_size`: 512 -> 2048 (4x larger chunks = ~4x fewer DocumentChunks)
- `chunk_overlap`: 128 -> 256 (proportional increase)
- `take_n_first_chunks`: 16 -> 8 (fewer but larger chunks per doc)
- `min_split_chars`: 50 -> 100
- `ingest_batch_size`: 32 -> 64 (smaller vectors = bigger batches)
- `embedding_model`: multilingual-e5-small

**Step 2: Commit**

```bash
git add configs/embedding.yaml configs/embedding_model/multilingual-e5-small.yaml
git commit -m "feat: increase chunk size to 2048 chars for vector compression"
```

---

### Task 3: Rewrite collection schema with single vector + BQ

**Files:**
- Modify: `juddges/data/documents_weaviate_db.py` (lines 251-789)

**Step 1: Replace vectorizer_config for LegalDocuments collection**

In `create_collections()`, replace the `vectorizer_config` for LegalDocuments (around line 629-640):

OLD:
```python
vectorizer_config=[
    wvcc.Configure.NamedVectors.text2vec_transformers(
        name=VectorName.BASE,
        vectorize_collection_name=False,
        source_properties=["full_text"],
        vector_index_config=wvcc.Configure.VectorIndex.hnsw(
            ef_construction=128,
            max_connections=32,
            distance_metric=wvcc.VectorDistances.COSINE,
        ),
    ),
],
```

NEW:
```python
vectorizer_config=[
    wvcc.Configure.NamedVectors.text2vec_transformers(
        name=VectorName.DEFAULT,
        vectorize_collection_name=False,
        source_properties=["full_text"],
        vector_index_config=wvcc.Configure.VectorIndex.hnsw(
            ef_construction=64,
            max_connections=16,
            distance_metric=wvcc.VectorDistances.COSINE,
            quantizer=wvcc.Configure.VectorIndex.Quantizer.bq(
                rescore_limit=200,
            ),
        ),
    ),
],
```

**Step 2: Replace vectorizer_config for DocumentChunks collection**

Replace the 3 named vectors (around line 756-778) with a single vector:

NEW:
```python
vectorizer_config=[
    wvcc.Configure.NamedVectors.text2vec_transformers(
        name=VectorName.DEFAULT,
        vectorize_collection_name=False,
        source_properties=["chunk_text"],
        vector_index_config=wvcc.Configure.VectorIndex.hnsw(
            ef_construction=64,
            max_connections=16,
            distance_metric=wvcc.VectorDistances.COSINE,
            quantizer=wvcc.Configure.VectorIndex.Quantizer.bq(
                rescore_limit=200,
            ),
        ),
    ),
],
```

**Step 3: Update search methods to use VectorName.DEFAULT**

In `semantic_search()` (line 843): change `target_vector: str = "base"` to `target_vector: str = "default"`

In `hybrid_search()` (line 927): change `target_vector="base"` to `target_vector="default"`

**Step 4: Reduce HNSW env vars in weaviate/.env**

Update `weaviate/.env`:
```
HNSW_EF_CONSTRUCTION=64
HNSW_MAX_CONNECTIONS=16
HNSW_DYNAMIC_EF_MAX=200
VECTOR_CACHE_MAX_OBJECTS=10000000
```

**Step 5: Commit**

```bash
git add juddges/data/documents_weaviate_db.py
git commit -m "feat: single vector with BQ quantization, reduced HNSW params"
```

---

### Task 4: Update transformer Docker service

**Files:**
- Modify: `weaviate/docker-compose.yaml`
- Modify: `weaviate/.env`

**Step 1: Update docker-compose.yaml transformer service**

Change `t2v-transformers-base` to use new model:

```yaml
  t2v-transformers-base:
    build:
      context: .
      dockerfile: hf_transformers.dockerfile
      args:
        MODEL_NAME: 'intfloat/multilingual-e5-small'
        ENABLE_CUDA: ${ENABLE_CUDA}
    ports:
      - "8082:8080"
    environment:
      ENABLE_CUDA: ${ENABLE_CUDA}
      MODEL_NAME: 'intfloat/multilingual-e5-small'
    restart: always
    deploy:
      restart_policy:
        condition: on-failure
        max_attempts: 3
      resources:
        limits:
          cpus: '4'
          memory: '4G'
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

Key changes:
- `MODEL_NAME`: `sdadas/mmlw-roberta-large` -> `intfloat/multilingual-e5-small`
- `memory`: `16G` -> `4G` (model is ~4x smaller)
- `cpus`: `8` -> `4`

**Step 2: Update .env model names**

```
MODEL_NAME_BASE='intfloat/multilingual-e5-small'
```

Remove `MODEL_NAME_DEV` and `MODEL_NAME_FAST`.
Remove `TRANSFORMERS_INFERENCE_API_DEV` and `TRANSFORMERS_INFERENCE_API_FAST`.

**Step 3: Commit**

```bash
git add weaviate/docker-compose.yaml
git commit -m "feat: switch transformer service to multilingual-e5-small"
```

---

### Task 5: Update stream_ingester.py for single vector + truncation

**Files:**
- Modify: `juddges/data/stream_ingester.py`

**Step 1: Update _generate_embeddings to truncate to 128 dims**

Replace `_generate_embeddings()` method (around line 790):

```python
def _generate_embeddings(self, texts: List[str]) -> Dict[str, List[List[float]]]:
    """Generate embeddings and truncate to Matryoshka dimension."""
    from juddges.settings import TEXT_EMBEDDING_DIM

    embeddings_dict = {}
    for vector_name, transformer in self.transformers.items():
        try:
            embeddings = transformer.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            # Matryoshka truncation + re-normalize
            truncated = embeddings[:, :TEXT_EMBEDDING_DIM]
            norms = np.linalg.norm(truncated, axis=1, keepdims=True)
            norms[norms == 0] = 1
            truncated = truncated / norms
            embeddings_dict[vector_name] = truncated.tolist()
        except Exception as e:
            logger.error(f"Failed to generate {vector_name} embeddings: {e}")
            embeddings_dict[vector_name] = [[0.0] * TEXT_EMBEDDING_DIM for _ in texts]
    return embeddings_dict
```

**Step 2: Update _aggregate_embeddings to use new dimension**

```python
def _aggregate_embeddings(self, embeddings_dict: Dict[str, List[List[float]]]) -> Dict[str, List[float]]:
    """Aggregate chunk embeddings into document embeddings."""
    from juddges.settings import TEXT_EMBEDDING_DIM

    aggregated_dict = {}
    for vector_name, embeddings in embeddings_dict.items():
        if not embeddings:
            aggregated_dict[vector_name] = [0.0] * TEXT_EMBEDDING_DIM
        else:
            agg = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(agg)
            if norm > 0:
                agg = agg / norm
            aggregated_dict[vector_name] = agg.tolist()
    return aggregated_dict
```

**Step 3: Update transformer initialization to only load single model**

Find where `self.transformers` dict is populated and ensure it only loads one model (`VectorName.DEFAULT`).

**Step 4: Add `normalize_embeddings=True` and `query: ` prefix for e5 models**

Note: `intfloat/multilingual-e5-small` requires `query: ` prefix for queries and `passage: ` prefix for documents. Update encoding calls accordingly:
- In ingestion: prefix texts with `"passage: "`
- In search (if using custom embeddings): prefix query with `"query: "`

If using Weaviate's text2vec-transformers module (which handles this internally via the HF model's tokenizer), this may not be needed — verify during testing.

**Step 5: Commit**

```bash
git add juddges/data/stream_ingester.py
git commit -m "feat: truncate embeddings to 128-dim Matryoshka with normalization"
```

---

### Task 6: Delete old collections and recreate

**DANGER ZONE — requires backup verification first**

**Step 1: Verify backup is accessible**

```bash
python weaviate/backup_native.py --list
# Must show: backup-20260324-224720 with SUCCESS status
```

**Step 2: Delete old collections**

```python
# Run from project root
WEAVIATE_HOST=localhost WEAVIATE_API_KEY="..." python -c "
import weaviate
client = weaviate.connect_to_custom(
    http_host='localhost', http_port=8084, http_secure=False,
    grpc_host='localhost', grpc_port=8085, grpc_secure=False,
    auth_credentials=weaviate.auth.AuthApiKey(api_key='...'),
)
print('Before:', [c.name for c in client.collections.list_all().values()])
client.collections.delete('DocumentChunks')
client.collections.delete('LegalDocuments')
print('After:', [c.name for c in client.collections.list_all().values()])
client.close()
"
```

**Step 3: Rebuild transformer service with new model**

```bash
cd weaviate/
docker compose build t2v-transformers-base
docker compose up -d t2v-transformers-base
# Wait for model download and startup
docker compose logs -f t2v-transformers-base  # Watch until "Application startup complete"
```

**Step 4: Restart Weaviate to pick up any env changes**

```bash
docker compose restart weaviate
# Wait for ready
curl -s http://localhost:8084/v1/.well-known/ready
```

**Step 5: Create new collections with BQ schema**

```python
# Use the updated WeaviateLegalDocumentsDatabase.create_collections()
WEAVIATE_HOST=localhost WEAVIATE_API_KEY="..." python -c "
from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase
db = WeaviateLegalDocumentsDatabase(
    host='localhost', port=8084, grpc_port=8085,
    api_key='...',
)
db.create_collections()
db.close()
print('Collections created with BQ')
"
```

**Step 6: Commit all config changes**

```bash
git add -A
git commit -m "feat: delete old collections, recreate with BQ + single 128-dim vector"
```

---

### Task 7: Re-ingest documents from Parquet backup

**Files:**
- Create: `scripts/migration/reingest_from_parquet.py`

**Step 1: Write re-ingestion script**

This script reads the Parquet backup from NAS and re-ingests with new chunking + embeddings:

```python
#!/usr/bin/env python3
"""Re-ingest documents from Parquet backup with new embedding model and chunk size."""

import os
from pathlib import Path

import pyarrow.parquet as pq
from loguru import logger
from rich.console import Console
from rich.progress import Progress

from juddges.data.stream_ingester import StreamIngester

console = Console()

PARQUET_PATH = Path("/mnt/readynas/datasets/legal-ai-weaviate/LegalDocuments_20260107_175321.parquet")

def main():
    # StreamIngester handles chunking, embedding, and ingestion
    # Configure it with the new model and chunk params
    ingester = StreamIngester(
        weaviate_host=os.getenv("WEAVIATE_HOST", "localhost"),
        weaviate_port=int(os.getenv("WEAVIATE_PORT", "8084")),
        weaviate_grpc_port=8085,
        weaviate_api_key=os.getenv("WEAVIATE_API_KEY", ""),
        embedding_model="intfloat/multilingual-e5-small",
        chunk_size=2048,
        chunk_overlap=256,
    )

    # Read parquet in batches and ingest
    parquet_file = pq.ParquetFile(PARQUET_PATH)
    total_rows = parquet_file.metadata.num_rows
    console.print(f"Total documents to re-ingest: {total_rows:,}")

    ingester.ingest_from_parquet(PARQUET_PATH)

if __name__ == "__main__":
    main()
```

Note: The exact implementation depends on StreamIngester's API. This may need adaptation — the key point is to read from the Parquet backup and feed into the new pipeline.

**Step 2: Run re-ingestion**

```bash
WEAVIATE_HOST=localhost WEAVIATE_API_KEY="..." python scripts/migration/reingest_from_parquet.py
```

Expected time: With smaller model + fewer chunks, this should take 2-4 hours (vs 6+ hours for the old pipeline).

**Step 3: Verify counts**

```bash
WEAVIATE_HOST=localhost WEAVIATE_API_KEY="..." python -c "
import weaviate
client = weaviate.connect_to_custom(
    http_host='localhost', http_port=8084, http_secure=False,
    grpc_host='localhost', grpc_port=8085, grpc_secure=False,
    auth_credentials=weaviate.auth.AuthApiKey(api_key='...'),
)
for name in ['LegalDocuments', 'DocumentChunks']:
    col = client.collections.get(name)
    count = col.aggregate.over_all(total_count=True).total_count
    print(f'{name}: {count:,}')
client.close()
"
```

Expected:
- LegalDocuments: ~3,185,832 (same as before)
- DocumentChunks: ~9-10M (was 37.8M — ~4x reduction from larger chunks)

---

### Task 8: Validate search quality

**Files:**
- Create: `scripts/migration/validate_search_quality.py`

**Step 1: Write validation script**

Test hybrid search with sample legal queries in Polish and English:

```python
#!/usr/bin/env python3
"""Validate search quality after vector compression migration."""

from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase

SAMPLE_QUERIES = [
    "kredyt frankowy CHF",
    "odszkodowanie za wypadek",
    "kara umowna",
    "prawo do informacji publicznej",
    "divorce proceedings",
    "tax deduction for business expenses",
]

def main():
    db = WeaviateLegalDocumentsDatabase(
        host="localhost", port=8084, grpc_port=8085,
        api_key="...",
    )

    for query in SAMPLE_QUERIES:
        print(f"\n{'='*60}")
        print(f"Query: {query}")

        # BM25 only
        bm25_results = db.bm25_search(query, limit=5)
        print(f"  BM25: {len(bm25_results)} results")

        # Hybrid (BM25 + vector)
        hybrid_results = db.hybrid_search(query, alpha=0.25, limit=5)
        print(f"  Hybrid (alpha=0.25): {len(hybrid_results)} results")

        # Show top result
        if hybrid_results:
            top = hybrid_results[0]
            print(f"  Top: {top.get('title', 'N/A')[:80]}")
            print(f"  Score: {top.get('_score', 'N/A')}")

    db.close()

if __name__ == "__main__":
    main()
```

**Step 2: Run validation**

```bash
WEAVIATE_HOST=localhost WEAVIATE_API_KEY="..." python scripts/migration/validate_search_quality.py
```

**Step 3: Measure storage**

```bash
# Check Weaviate data size
docker exec legal_ai-weaviate-1 du -sh /var/lib/weaviate/
```

Expected: ~3-6 GB (was ~260+ GB)

---

### Task 9: Create new backup of compressed data

**Step 1: Run native backup**

```bash
WEAVIATE_HOST=localhost WEAVIATE_API_KEY="..." python weaviate/backup_native.py --backup-id post-compression-v1
```

Expected: Should complete in minutes (not hours) due to much smaller data.

**Step 2: Commit migration script**

```bash
git add scripts/migration/
git commit -m "feat: add re-ingestion and validation scripts for vector compression"
```

---

## Expected Results

| Metric | Before | After | Reduction |
|---|---|---|---|
| Embedding model | mmlw-roberta-large (1024-dim) | multilingual-e5-small (128-dim) | 8x dims |
| Quantization | float32 (32 bits/dim) | BQ (1 bit/dim) | 32x bits |
| DocumentChunks count | 37.8M | ~9.5M | 4x fewer |
| Vector storage | ~145 GB (chunks) + 12 GB (docs) | ~0.15 GB + 0.05 GB | **~800x** |
| HNSW index | ~100+ GB | ~2-5 GB | **~30x** |
| Total Weaviate size | ~300+ GB | ~3-6 GB | **50-100x** |
| Transformer service RAM | 16 GB | 4 GB | 4x |
| Search quality (hybrid) | Baseline | Minimal degradation | BM25 carries |

## Rollback Plan

If search quality is unacceptable:

1. Delete new collections
2. Restore from native backup: `python weaviate/backup_native.py --restore backup-20260324-224720`
3. Revert git changes: `git revert HEAD~N`
4. Rebuild old transformer: update docker-compose back to mmlw-roberta-large
