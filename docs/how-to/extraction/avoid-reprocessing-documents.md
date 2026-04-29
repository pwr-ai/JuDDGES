# How to Avoid Reprocessing Already-Extracted Documents

This guide explains how to use the automatic filtering feature to skip documents that have already been successfully extracted, saving API costs and processing time.

## Overview

The extraction system now automatically checks the PostgreSQL database (`extraction_results` table) to identify documents that have already been processed. This prevents duplicate work and reduces costs when:

- Resuming interrupted extraction jobs
- Running overlapping search queries
- Re-running extraction with different parameters on the same dataset

## Quick Start

### 1. Start Multi-Region Workers

```bash
./start_workers_multi_region.sh
```

This launches 54 workers across 18 regions (US, EU, Asia-Pacific).

### 2. Queue Jobs with Automatic Filtering

**For specific search queries:**

```bash
python scripts/extraction/coordinator.py \
    --search-queries "kredyt frankowy" "IP Box" \
    --max-documents 100000 \
    --filter-already-extracted \
    --redis-url "$REDIS_URL"
```

**For ALL documents (using cursor pagination, up to millions):**

```bash
python scripts/extraction/coordinator.py \
    --force-cursor \
    --max-documents 1000000 \
    --filter-already-extracted \
    --redis-url "$REDIS_URL"
```

**For chronological processing (sorted by publication_date, limited to 10K):**

```bash
python scripts/extraction/coordinator.py \
    --max-documents 10000 \
    --filter-already-extracted \
    --sort-by-creation-time \
    --redis-url "$REDIS_URL"
```

> **Note:** Sorting has Weaviate limitations:
> - Maximum 10,000 documents (offset pagination limit)
> - Cannot be combined with `--force-cursor` (cursor pagination)
> - Cannot be used with hybrid/semantic search (only keyword search or no search)
> - For processing millions of documents, use `--force-cursor` without sorting

## How It Works

### Database Query

The `ExtractionStorage.get_processed_document_ids()` method queries the database:

```sql
SELECT DISTINCT document_id
FROM extraction_results
WHERE extraction_status = 'success'
```

### Filtering Process

1. **Fetch documents** from Weaviate based on search criteria
2. **Query database** for all successfully extracted document IDs (291K+ documents)
3. **Filter out** documents that already exist in the database
4. **Queue only** unprocessed documents to Redis

### Example Output

```
Step 1: Fetching document IDs from Weaviate...
  Found: 44 documents

Step 2: Filtering already-extracted documents...
  Found 291,172 processed document IDs
  Filtered 42 already-extracted documents. Remaining: 2 documents

Step 3: Queuing jobs to Redis...
✓ Queued 1 jobs to Redis
✓ Total documents: 2
```

## Advanced Usage

### Filter by Specific Run ID

If you want to check only documents from a specific extraction run:

```python
from juddges.extraction import ExtractionStorage

storage = ExtractionStorage()
processed_ids = storage.get_processed_document_ids(
    status="success",
    run_id="your-run-id-here"
)
```

### Include Failed Extractions

To get all processed documents regardless of status:

```python
processed_ids = storage.get_processed_document_ids(status=None)
```

## Benefits

1. **Cost Savings**: Avoid redundant API calls to Gemini
2. **Time Efficiency**: Skip already-processed documents
3. **Resume Support**: Easily resume interrupted extraction jobs
4. **Idempotent Operations**: Safe to re-run extraction commands

## Monitoring

Check how many documents are already processed:

```bash
# Connect to PostgreSQL
psql -h 127.0.0.1 -p 5434 -U extraction_user -d legal_extraction

# Count processed documents
SELECT
    extraction_status,
    COUNT(*) as count
FROM extraction_results
GROUP BY extraction_status;

# Check specific search query results
SELECT COUNT(DISTINCT document_id)
FROM extraction_results
WHERE extraction_status = 'success';
```

## Troubleshooting

### Storage Connection Issues

If filtering fails, the coordinator will log a warning and continue without filtering:

```
Failed to filter already-extracted documents: <error>
Returning all documents without filtering
```

Check your PostgreSQL connection settings in `.env`:

```bash
EXTRACTION_POSTGRES_HOST=localhost
EXTRACTION_POSTGRES_PORT=5434
EXTRACTION_POSTGRES_USER=extraction_user
EXTRACTION_POSTGRES_PASSWORD=extraction_pass
EXTRACTION_POSTGRES_DB=legal_extraction
```

### Performance Considerations

The database query is optimized with:
- Index on `extraction_results.extraction_status`
- Index on `extraction_results.document_id`
- Set-based filtering in Python for O(1) lookup

Large result sets (300K+ documents) complete in under 1 second.

## See Also

- [Distributed Extraction with Redis](../distributed-extraction-redis.md)
- [Extraction Database Schema](../../reference/api/extraction_storage_schema.md)
