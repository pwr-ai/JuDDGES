# How to Generate Redis Tasks for Large-Scale Extraction

This guide explains how to use `generate_redis_tasks.py` to create Redis tasks for distributed extraction of legal documents with automatic deduplication.

## Overview

The `generate_redis_tasks.py` script:
1. Fetches document IDs from Weaviate (up to 1,000,000 documents)
2. Filters out already-processed documents from PostgreSQL
3. Creates Redis tasks with configurable batch sizes (default: 10 documents per task)
4. Provides detailed statistics on documents added vs already processed

## Prerequisites

- Redis server running and accessible
- PostgreSQL extraction database running
- Weaviate server with legal documents
- Environment variables configured in `.env`:
  - `REDIS_URL` - Redis connection URL
  - `EXTRACTION_POSTGRES_HOST`, `EXTRACTION_POSTGRES_PORT`, etc.
  - `WEAVIATE_HOST`, `WEAVIATE_PORT`

## Basic Usage

### Generate Tasks for 1 Million Documents

```bash
python scripts/extraction/generate_redis_tasks.py \
    --force-cursor \
    --max-documents 1000000 \
    --task-size 10
```

This will:
- Fetch up to 1M document IDs from Weaviate using cursor pagination
- Filter out already processed documents
- Create tasks with 10 documents each
- Queue tasks to Redis

### Generate Tasks with Search Query

```bash
python scripts/extraction/generate_redis_tasks.py \
    --search-queries "kredyt frankowy" "IP Box" \
    --max-documents 100000 \
    --task-size 10
```

This will:
- Search for documents matching "kredyt frankowy" OR "IP Box"
- Process up to 100K documents
- Create tasks with 10 documents each

## Command-Line Options

### Required Options

You must provide either:
- `--search-queries` - One or more search queries
- `--force-cursor` - Fetch ALL documents without search filtering

### Document Selection

- `--max-documents N` - Maximum number of documents to process (default: 1,000,000)
- `--skip-documents N` - Skip first N documents (useful for resuming)
- `--document-type {judgment,tax_interpretation}` - Filter by document type
- `--sort-by-creation-time` - Sort by publication date (oldest first)

### Task Configuration

- `--task-size N` - Documents per Redis task (default: 10)
- `--queue-name NAME` - Redis queue name (default: `extraction_queue`)
- `--run-name NAME` - Optional descriptive name for this run

### Search Configuration

- `--search-mode {keyword,semantic,hybrid}` - Search mode:
  - `keyword` - BM25 keyword search (exact term matching)
  - `semantic` - Vector similarity search (semantic meaning)
  - `hybrid` - Both keyword and semantic (default)

### Connection Configuration

- `--redis-url URL` - Redis URL (format: `redis://:password@host:port/db`)
  - If not provided, uses `REDIS_URL` environment variable

## Examples

### Example 1: Process All Documents

```bash
python scripts/extraction/generate_redis_tasks.py \
    --force-cursor \
    --max-documents 1000000 \
    --task-size 10 \
    --run-name "Full corpus extraction - batch 1"
```

### Example 2: Process Only Judgments

```bash
python scripts/extraction/generate_redis_tasks.py \
    --force-cursor \
    --max-documents 500000 \
    --task-size 10 \
    --document-type judgment \
    --run-name "Judgments only"
```

### Example 3: Process Specific Legal Topics

```bash
python scripts/extraction/generate_redis_tasks.py \
    --search-queries "kredyt frankowy" "umowa kredytu" "spread walutowy" \
    --max-documents 50000 \
    --task-size 10 \
    --search-mode keyword \
    --run-name "Swiss franc loans"
```

### Example 4: Resume Interrupted Task Generation

```bash
python scripts/extraction/generate_redis_tasks.py \
    --force-cursor \
    --max-documents 1000000 \
    --skip-documents 500000 \
    --task-size 10 \
    --run-name "Resume from 500K"
```

### Example 5: Process Oldest Documents First

```bash
python scripts/extraction/generate_redis_tasks.py \
    --force-cursor \
    --max-documents 10000 \
    --task-size 10 \
    --sort-by-creation-time \
    --run-name "Chronological processing"
```

Note: Sorting is limited to 10K documents due to Weaviate offset limitations.

## Output

The script provides:

1. **Real-time progress** - Shows task generation progress with progress bar
2. **Summary table** - Displays statistics:
   - Total fetched from Weaviate
   - Already processed (from PostgreSQL)
   - Filtered out (duplicates)
   - New documents to process
   - Tasks generated
   - Documents queued
   - Average docs per task
   - Duration

3. **Statistics file** - Saved to `data/extraction_runs/{run_id}_task_generation.json`

### Example Output

```
Redis Task Generator
Run ID: 123e4567-e89b-12d3-a456-426614174000
Target documents: 1,000,000
Task size: 10 documents per task
Redis queue: extraction_queue

Step 1: Fetching document IDs from Weaviate...
  Fetched: 1,000,000 document IDs

Step 2: Checking already processed documents...
  Already processed: 291,234 documents

Step 3: Filtering documents...
  Filtered out: 291,234 already processed
  Remaining to process: 708,766 documents

Step 4: Generating Redis tasks...
Queuing tasks... 100% • 70,877/70,877 tasks

✓ Task Generation Complete

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric                     ┃    Count ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Total fetched from         │1,000,000 │
│ Weaviate                   │          │
│ Already processed          │  291,234 │
│ Filtered out (duplicates)  │  291,234 │
│ New documents to process   │  708,766 │
│ Tasks generated            │   70,877 │
│ Documents queued           │  708,766 │
│ Average docs per task      │     10.0 │
│ Duration                   │    127.3s│
└────────────────────────────┴──────────┘

Workers can now poll queue: extraction_queue

Statistics saved to: data/extraction_runs/123e4567-e89b-12d3-a456-426614174000_task_generation.json
```

## Deduplication Strategy

The script automatically filters already-processed documents using PostgreSQL:

1. Queries `extraction_results` table for all documents with `extraction_status = 'success'`
2. Creates a set of processed document IDs
3. Filters fetched document IDs against this set
4. Only queues documents that haven't been processed yet

This ensures:
- No duplicate processing
- Efficient resource usage
- Resumable operations

## Integration with Workers

After generating tasks, start workers to process them:

```bash
# Start single worker
python scripts/extraction/worker.py \
    --worker-id 1 \
    --batch-size 5

# Start multiple workers (multi-region)
./scripts/infrastructure/start_workers_multi_region.sh
```

Workers will:
1. Poll the Redis queue
2. Fetch document batches
3. Extract data using Gemini
4. Save results to PostgreSQL
5. Mark documents as processed (preventing re-extraction)

## Monitoring

Check queue status:

```bash
python scripts/extraction/check_queue.py
```

Check worker health:

```bash
python scripts/extraction/check_worker_health.py
```

## Troubleshooting

### Issue: "Storage not available"

**Cause:** PostgreSQL connection failed

**Solution:**
- Check PostgreSQL is running: `docker compose ps`
- Verify connection parameters in `.env`
- Test connection: `psql -h localhost -p 5433 -U extraction_user -d legal_extraction`

### Issue: "Redis authentication failed"

**Cause:** Redis password incorrect or not provided

**Solution:**
- Ensure `REDIS_URL` includes password: `redis://:YOUR_PASSWORD@host:port/db`
- Or pass via `--redis-url` argument

### Issue: "No new documents to process"

**Cause:** All documents already processed

**Solution:**
- This is normal - all documents have been extracted
- Check PostgreSQL for processed documents:
  ```sql
  SELECT COUNT(*) FROM extraction_results WHERE extraction_status = 'success';
  ```

### Issue: Tasks generated but workers not processing

**Cause:** Workers not running or misconfigured

**Solution:**
- Start workers: `python scripts/extraction/worker.py --worker-id 1`
- Check Redis queue: `python scripts/extraction/check_queue.py`
- Verify Redis connection in workers

## Best Practices

1. **Start small** - Test with `--max-documents 1000` before scaling to 1M
2. **Use force-cursor** - For processing entire corpus, `--force-cursor` bypasses 10K offset limits
3. **Optimal task size** - 10 documents per task balances:
   - Parallelization (more tasks = more workers can process)
   - Efficiency (batching reduces API overhead)
4. **Monitor progress** - Use `check_queue.py` to monitor task consumption
5. **Save run names** - Use descriptive `--run-name` for tracking different extraction runs
6. **Resume safely** - Use `--skip-documents` to resume interrupted task generation

## Related Documentation

- [Distributed Extraction with Redis](../distributed-extraction-redis.md)
- [Avoid Reprocessing Documents](./avoid-reprocessing-documents.md)
