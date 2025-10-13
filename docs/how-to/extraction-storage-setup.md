# Extraction Storage Setup

## Overview

The extraction storage system uses a dedicated PostgreSQL database to store all extraction inputs, outputs, and metadata. This provides:

- ✅ **Parallel execution safety** - ACID transactions prevent data loss when running multiple extraction jobs
- ✅ **Complete audit trail** - All prompts, parameters, and timestamps are preserved
- ✅ **Easy analysis** - SQL queries instead of parsing JSONL files
- ✅ **Ingestion support** - Full traceability from extraction to Weaviate ingestion

## Architecture

### Components

1. **PostgreSQL Database** (`extraction-postgres` container)
   - Port: **5434** (avoiding conflicts with other postgres instances)
   - Database: `legal_extraction`
   - User: `extraction_user`

2. **Storage Module** (`juddges/extraction/extraction_storage.py`)
   - Python API for database operations
   - Connection pooling and transaction management
   - Export utilities (JSONL, CSV)

3. **Database Schema** (`weaviate/init_extraction_db.sql`)
   - `extraction_runs` - Run metadata and configuration
   - `extraction_results` - Document inputs and outputs
   - `ingestion_logs` - Weaviate ingestion tracking
   - `field_coverage` - Extraction quality metrics

## Quick Start

### 1. Start the Extraction PostgreSQL Container

```bash
cd weaviate
docker compose up -d extraction-postgres
```

### 2. Verify Database Setup

```bash
# Check container is running
docker ps | grep extraction-postgres

# Connect to database
docker exec -it legal_ai_extraction_postgres psql -U extraction_user -d legal_extraction

# List tables
\dt

# Exit
\q
```

### 3. Configuration

The extraction storage is configured via environment variables:

**Main `.env` file:**
```bash
EXTRACTION_POSTGRES_USER=extraction_user
EXTRACTION_POSTGRES_PASSWORD=extraction_pass
EXTRACTION_POSTGRES_DB=legal_extraction
EXTRACTION_POSTGRES_HOST=localhost
EXTRACTION_POSTGRES_PORT=5434
EXTRACTION_POSTGRES_URL=postgresql+psycopg://extraction_user:extraction_pass@localhost:5434/legal_extraction
```

**Weaviate `.env` file:**
```bash
# Same variables for docker-compose
EXTRACTION_POSTGRES_USER=extraction_user
EXTRACTION_POSTGRES_PASSWORD=extraction_pass
EXTRACTION_POSTGRES_DB=legal_extraction
EXTRACTION_POSTGRES_HOST=localhost
EXTRACTION_POSTGRES_PORT=5434
```

## Database Schema

### extraction_runs

Stores metadata about each extraction run:

| Column | Type | Description |
|--------|------|-------------|
| run_id | UUID | Primary key |
| search_query | TEXT | Optional search query |
| document_type_filter | TEXT | Document type filter |
| model_name | TEXT | Gemini model used |
| prompt_template | TEXT | Full prompt template |
| extraction_schema | JSONB | Complete schema definition |
| sample_size | INTEGER | Number of documents |
| batch_size | INTEGER | Batch size |
| max_workers | INTEGER | Parallel workers |
| total_documents | INTEGER | Results: total docs |
| successful_extractions | INTEGER | Results: successful |
| failed_extractions | INTEGER | Results: failed |
| started_at | TIMESTAMP | Start time |
| completed_at | TIMESTAMP | Completion time |
| duration_seconds | FLOAT | Total duration |

### extraction_results

Stores individual document extraction results:

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | Primary key |
| run_id | UUID | FK to extraction_runs |
| **document_id** | TEXT | Weaviate UUID |
| **document_number** | TEXT | Case number |
| **document_type** | TEXT | Document type |
| **full_text** | TEXT | Complete document |
| extraction_status | TEXT | success/failed/skipped |
| **extracted_data** | JSONB | Extracted fields |
| error_message | TEXT | Error if failed |
| extracted_at | TIMESTAMP | Extraction time |

### ingestion_logs

Tracks Weaviate ingestion operations:

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | Primary key |
| run_id | UUID | FK to extraction_runs |
| batch_size | INTEGER | Ingestion batch size |
| overwrite_existing | BOOLEAN | Overwrite mode |
| successful_updates | INTEGER | Success count |
| failed_updates | INTEGER | Failed count |
| errors | JSONB | Error details |
| status | TEXT | running/completed/failed |

### field_coverage

Tracks extraction quality:

| Column | Type | Description |
|--------|------|-------------|
| run_id | UUID | FK to extraction_runs |
| field_name | TEXT | Field name |
| populated_count | INTEGER | Non-empty values |
| empty_count | INTEGER | Empty values |
| coverage_percentage | FLOAT | Auto-calculated % |

## Python API Usage

### Initialize Storage

```python
from juddges.extraction.extraction_storage import ExtractionStorage

# Using environment variables
storage = ExtractionStorage()

# Or explicit connection
storage = ExtractionStorage(
    host="localhost",
    port=5434,
    user="extraction_user",
    password="extraction_pass",
    database="legal_extraction"
)
```

### Create Extraction Run

```python
from uuid import UUID

run_id: UUID = storage.create_extraction_run(
    model_name="gemini-2.5-pro",
    sample_size=100,
    batch_size=10,
    max_workers=3,
    weaviate_host="localhost",
    weaviate_port=8084,
    search_query="kredyt frankowy",
    document_type_filter="judgment",
    vertex_project="insbay-b32351",
    vertex_location="us-central1",
    temperature=0.0,
    prompt_template="...",  # Full template
    extraction_schema={...},  # Complete schema
    random_seed=42,
    notes="Production run with new schema"
)

print(f"Created run: {run_id}")
```

### Save Extraction Results

```python
# Save single result
storage.save_extraction_result(
    run_id=run_id,
    document_id="/doc/ABC123",
    document_number="I ACa 123/23",
    document_type="judgment",
    full_text="...",
    extraction_status="success",
    extracted_data={"title": "...", "summary": "..."},
    processing_time_seconds=2.5,
    source_language="pl"
)

# Save batch of results (recommended for parallel jobs)
results = [
    {
        "document_id": "/doc/ABC123",
        "document_number": "I ACa 123/23",
        "document_type": "judgment",
        "full_text": "...",
        "extraction_status": "success",
        "extracted_data": {...},
        "full_text_length": 5000,
        "source_language": "pl"
    },
    # ... more results
]

successful, failed = storage.save_extraction_results_batch(run_id, results)
print(f"Saved: {successful} successful, {failed} failed")
```

### Complete Run

```python
storage.complete_extraction_run(
    run_id=run_id,
    total_documents=100,
    successful_extractions=95,
    failed_extractions=5,
    duration_seconds=120.5
)

# Save field coverage
field_coverage = {
    "title": {"populated": 95, "empty": 5},
    "summary": {"populated": 90, "empty": 10},
    # ... more fields
}
storage.save_field_coverage(run_id, field_coverage)
```

### Get Results for Ingestion

```python
# Get all successful extractions for Weaviate ingestion
results = storage.get_extraction_results_for_ingestion(
    run_id=run_id,
    status="success"
)

# Results are ready for Weaviate PATCH requests
for result in results:
    print(result["document_id"], result["extracted_data"])
```

### Log Ingestion

```python
ingestion_id = storage.log_ingestion(
    run_id=run_id,
    batch_size=50,
    overwrite_existing=False,
    total_documents=100,
    successful_updates=95,
    failed_updates=5,
    skipped_documents=0,
    duration_seconds=30.2,
    errors=[...],
    status="completed"
)
```

### Export to JSONL

```python
# Export extraction results to JSONL
storage.export_to_jsonl(
    run_id=run_id,
    output_path="data/extraction_results/run_2025_01_15.jsonl",
    include_full_text=True
)
```

## SQL Queries

### Find Extractions by Query

```sql
-- Find all extractions for a specific search query
SELECT er.*, res.*
FROM extraction_runs er
JOIN extraction_results res ON er.run_id = res.run_id
WHERE er.search_query = 'kredyt frankowy'
ORDER BY res.extracted_at DESC;
```

### Latest Successful Extraction per Document

```sql
-- Get the latest successful extraction for each document
SELECT * FROM v_latest_extraction_by_document
WHERE extraction_status = 'success';
```

### Failed Extractions

```sql
-- Find documents that failed extraction
SELECT document_id, document_number, error_message, error_type
FROM extraction_results
WHERE extraction_status = 'failed'
ORDER BY extracted_at DESC;
```

### Extraction Quality Metrics

```sql
-- Get extraction quality metrics by run
SELECT * FROM v_extraction_quality_metrics
ORDER BY avg_field_coverage DESC;
```

### Run Statistics

```sql
-- Get comprehensive run summary
SELECT * FROM v_extraction_run_summary
WHERE run_id = 'your-run-id';

-- Or use the helper function
SELECT * FROM get_extraction_stats('your-run-id');
```

### Export for Ingestion

```sql
-- Export successful extractions for Weaviate ingestion
SELECT
    document_id,
    document_number,
    extracted_data
FROM extraction_results
WHERE run_id = 'your-run-id'
  AND extraction_status = 'success'
ORDER BY extracted_at;
```

## Parallel Execution Safety

The database ensures safe parallel execution through:

1. **UNIQUE constraint** on `(run_id, document_id)` prevents duplicates
2. **ACID transactions** ensure atomic writes
3. **ON CONFLICT DO UPDATE** handles race conditions gracefully
4. **Connection pooling** manages concurrent connections

### Example: Running 10 Parallel Jobs

```bash
# Each job gets a unique run_id
for query in "kredyt frankowy" "VAT" "CIT" "odliczenie" "umowa"; do
    python scripts/extraction/run_extraction_rest.py \
        --search-query "$query" \
        --sample-size 100 \
        --max-workers 3 &
done

# All jobs write safely to the same database
wait
```

## Troubleshooting

### Check Database Connection

```bash
# Test connection from host
psql -h localhost -p 5434 -U extraction_user -d legal_extraction

# Test from Python
python -c "from juddges.extraction.extraction_storage import ExtractionStorage; storage = ExtractionStorage(); print('Connected!')"
```

### View Recent Runs

```sql
SELECT
    run_id,
    search_query,
    document_type_filter,
    model_name,
    sample_size,
    successful_extractions,
    failed_extractions,
    started_at,
    duration_seconds
FROM extraction_runs
ORDER BY started_at DESC
LIMIT 10;
```

### Database Size

```sql
-- Check database size
SELECT pg_size_pretty(pg_database_size('legal_extraction'));

-- Check table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Clean Up Old Runs

```sql
-- Delete runs older than 30 days (cascades to results)
DELETE FROM extraction_runs
WHERE started_at < NOW() - INTERVAL '30 days';
```

## Migration from JSONL Files

If you have existing JSONL extraction results:

```python
import json
from juddges.extraction.extraction_storage import ExtractionStorage

storage = ExtractionStorage()

# Create run for historical data
run_id = storage.create_extraction_run(
    model_name="gemini-2.5-pro",
    sample_size=100,
    batch_size=10,
    max_workers=1,
    weaviate_host="localhost",
    weaviate_port=8084,
    notes="Migrated from JSONL files"
)

# Load and save results
results = []
with open("data/extraction_results/sample_documents_extracted.jsonl") as f:
    for line in f:
        result = json.loads(line)
        results.append(result)

storage.save_extraction_results_batch(run_id, results)
storage.complete_extraction_run(run_id, len(results), ...)
```

## Next Steps

1. **Update `run_extraction_rest.py`** to use ExtractionStorage
2. **Add export commands** to CLI for JSONL/CSV export
3. **Create analytics dashboard** querying the database
4. **Set up automated backups** of extraction database
