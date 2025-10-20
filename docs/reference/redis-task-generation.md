# Redis Task Generation Reference

Quick reference for `generate_redis_tasks.py` - a tool for creating Redis extraction tasks with automatic deduplication.

## Quick Start

```bash
# Process 1M documents (10 docs per task)
python scripts/extraction/generate_redis_tasks.py \
    --force-cursor \
    --max-documents 1000000 \
    --task-size 10
```

## Command Syntax

```bash
python scripts/extraction/generate_redis_tasks.py \
    [--max-documents N] \
    [--task-size N] \
    [--search-queries QUERY ...] \
    [--document-type TYPE] \
    [--search-mode MODE] \
    [--force-cursor] \
    [--skip-documents N] \
    [--sort-by-creation-time] \
    [--redis-url URL] \
    [--queue-name NAME] \
    [--run-name NAME]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--max-documents` | int | 1,000,000 | Maximum documents to process |
| `--task-size` | int | 10 | Documents per Redis task |
| `--search-queries` | str[] | None | Search queries (required unless `--force-cursor`) |
| `--document-type` | enum | None | Filter: `judgment`, `tax_interpretation` |
| `--search-mode` | enum | `hybrid` | Mode: `keyword`, `semantic`, `hybrid` |
| `--force-cursor` | flag | False | Bypass search, fetch ALL documents |
| `--skip-documents` | int | 0 | Skip first N documents |
| `--sort-by-creation-time` | flag | False | Sort by date (limited to 10K) |
| `--redis-url` | str | `$REDIS_URL` | Redis connection URL |
| `--queue-name` | str | `extraction_queue` | Redis queue name |
| `--run-name` | str | None | Optional descriptive name |

## Environment Variables

Required in `.env`:

```bash
# Redis
REDIS_URL=redis://:PASSWORD@host:port/db

# PostgreSQL (for deduplication)
EXTRACTION_POSTGRES_HOST=localhost
EXTRACTION_POSTGRES_PORT=5433
EXTRACTION_POSTGRES_USER=extraction_user
EXTRACTION_POSTGRES_PASSWORD=extraction_pass
EXTRACTION_POSTGRES_DB=legal_extraction

# Weaviate
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8084
```

## Output Structure

### Console Output

```
Redis Task Generator
Run ID: {uuid}
Target documents: {count}
Task size: {size} documents per task

Step 1: Fetching document IDs from Weaviate...
Step 2: Checking already processed documents...
Step 3: Filtering documents...
Step 4: Generating Redis tasks...

✓ Task Generation Complete

[Summary Table]
```

### Statistics File

Location: `data/extraction_runs/{run_id}_task_generation.json`

```json
{
  "run_id": "uuid",
  "total_fetched": 1000000,
  "already_processed": 291234,
  "filtered_count": 291234,
  "new_documents": 708766,
  "tasks_generated": 70877,
  "documents_queued": 708766,
  "duration_seconds": 127.3
}
```

## Usage Patterns

### Pattern 1: Full Corpus Extraction

```bash
python scripts/extraction/generate_redis_tasks.py \
    --force-cursor \
    --max-documents 1000000 \
    --task-size 10
```

**Use case:** Extract all documents in database
**Documents:** Up to 1M
**Filtering:** Automatic deduplication via PostgreSQL

### Pattern 2: Targeted Extraction

```bash
python scripts/extraction/generate_redis_tasks.py \
    --search-queries "kredyt frankowy" \
    --max-documents 50000 \
    --task-size 10 \
    --search-mode keyword
```

**Use case:** Extract documents on specific topic
**Documents:** Matching search query
**Search:** BM25 keyword matching

### Pattern 3: Resume Interrupted Run

```bash
python scripts/extraction/generate_redis_tasks.py \
    --force-cursor \
    --max-documents 1000000 \
    --skip-documents 500000 \
    --task-size 10
```

**Use case:** Continue from where you left off
**Documents:** Skip first 500K, process next 500K
**Note:** Automatic deduplication still applies

### Pattern 4: Document Type Filter

```bash
python scripts/extraction/generate_redis_tasks.py \
    --force-cursor \
    --max-documents 500000 \
    --task-size 10 \
    --document-type judgment
```

**Use case:** Extract only judgments (not tax interpretations)
**Documents:** Filtered by type
**Types:** `judgment`, `tax_interpretation`

## Deduplication Logic

```python
# Pseudocode
fetched_ids = weaviate.fetch_documents(max=1M)           # Step 1
processed_ids = postgres.get_processed_documents()       # Step 2
remaining_ids = fetched_ids - processed_ids              # Step 3
tasks = create_tasks(remaining_ids, task_size=10)        # Step 4
redis.queue_tasks(tasks)                                 # Step 4
```

## Task Structure

Each Redis task contains:

```json
{
  "job_id": "uuid-of-job",
  "run_id": "uuid-of-extraction-run",
  "document_ids": [
    "doc-id-1",
    "doc-id-2",
    "...",
    "doc-id-10"
  ]
}
```

Queue operation: `LPUSH extraction_queue <task_json>`

## Integration Points

### PostgreSQL Tables

**Read from:**
- `extraction_results` - Check which documents already processed

**Write to:**
- `extraction_runs` - Create new extraction run record

### Redis Queues

**Write to:**
- `extraction_queue` (default) - Queue tasks for workers

### Weaviate Collections

**Read from:**
- `LegalDocuments` - Fetch document IDs

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Fetch rate | ~8K docs/sec | From Weaviate (cursor mode) |
| Dedup rate | ~50K docs/sec | PostgreSQL set operations |
| Queue rate | ~15K tasks/sec | Redis LPUSH operations |
| Typical duration | 2-3 minutes | For 1M documents |

## Return Codes

- `0` - Success
- `1` - Redis connection failed
- `1` - Missing required arguments
- `1` - Weaviate connection failed

## Logging

**Console:**
- INFO level with rich formatting
- Progress bars for long operations
- Summary tables for statistics

**Loguru:**
- Detailed debug information
- Error traces
- Connection diagnostics

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `Redis authentication failed` | Wrong password | Check `REDIS_URL` format |
| `Storage not available` | PostgreSQL down | Start PostgreSQL container |
| `No new documents to process` | All processed | Normal - extraction complete |
| `--search-queries is required` | Missing args | Add queries or `--force-cursor` |

## Comparison with coordinator.py

| Feature | generate_redis_tasks.py | coordinator.py |
|---------|-------------------------|----------------|
| **Task size** | 10 docs (configurable) | 2 docs (fixed) |
| **Deduplication** | Built-in, always on | Optional flag |
| **Output** | Rich table + stats file | Console only |
| **Progress** | Real-time progress bar | No progress bar |
| **Use case** | One-time batch generation | Continuous coordination |
| **Monitoring** | Separate tools | Built-in monitoring |

## See Also

- [How-To: Generate Redis Tasks](../how-to/extraction/generate-redis-tasks.md)
- [How-To: Distributed Extraction](../how-to/distributed-extraction-redis.md)
- [Reference: Worker Configuration](./worker-configuration.md)
- [Reference: Extraction Storage Schema](./extraction-storage-schema.md)
