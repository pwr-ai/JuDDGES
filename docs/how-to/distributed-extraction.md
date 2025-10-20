# Distributed Extraction for Large-Scale Processing

This guide explains how to run extraction at scale (100K - 1M+ documents) using a distributed worker architecture.

## Architecture Overview

```
┌─────────────┐
│ Coordinator │  Fetches document IDs and queues jobs
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Redis    │  Job queue (documents grouped in batches of 50)
└──────┬──────┘
       │
       ├───────┬───────┬───────┬─────────┐
       ▼       ▼       ▼       ▼         ▼
   Worker1  Worker2  Worker3  ...  WorkerN
       │       │       │       │         │
       └───────┴───────┴───────┴─────────┘
                      │
                      ▼
              ┌──────────────┐
              │  PostgreSQL  │  Results storage + checkpoints
              └──────────────┘
```

### Key Design Decisions

- **Small batch size (3 docs)**: Polish legal documents are very long (5K-50K tokens)
- **Many parallel workers (20-200)**: Horizontal scaling instead of larger batches
- **Redis queue**: Simple, reliable job distribution
- **Independent workers**: Each worker polls queue independently
- **Checkpoint storage**: All results saved to PostgreSQL for resume capability

## Performance Estimates

| Workers | Docs/Worker/Min | Total Throughput | Time for 100K | Time for 1M |
|---------|-----------------|------------------|---------------|-------------|
| **5**   | 6               | 30 docs/min      | 55 hours      | 23 days     |
| **20**  | 6               | 120 docs/min     | 14 hours      | 5.8 days    |
| **50**  | 6               | 300 docs/min     | 5.6 hours     | 2.3 days    |
| **100** | 6               | 600 docs/min     | 2.8 hours     | 28 hours    |
| **200** | 6               | 1,200 docs/min   | 1.4 hours     | **14 hours** ⚡ |

## Prerequisites

### Start Redis (Required)

Redis is required for distributed extraction to manage the job queue. Start it using docker-compose:

```bash
# Start Redis via docker-compose (recommended)
cd weaviate
docker compose up -d extraction-redis

# Verify Redis is running
redis-cli -p 6381 ping
# Expected: PONG

# Alternative: Start standalone Redis (testing only)
docker run -d --name extraction-redis -p 6381:6379 redis:7-alpine
```

**Redis Configuration:**

- **Port**: 6381 (to avoid conflicts with other Redis instances)
- **Container**: `legal_ai_extraction_redis`
- **Documentation**: See `weaviate/README_REDIS.md` for details

## Quick Start

### Option 1: Local Multi-Worker Setup (Recommended for Testing)

```bash
# 1. Start Redis (if not already running)
cd weaviate && docker compose up -d extraction-redis

# 2. Create logs directory
mkdir -p logs

# 3. Run distributed extraction with 20 workers
./scripts/extraction/run_distributed.sh \
    --num-workers 20 \
    --sample-size 1000 \
    --search-queries "kredyt frankowy" \
    --redis-url redis://localhost:6381

# Monitor progress in real-time
watch -n 5 'redis-cli -p 6381 LLEN extraction_queue'
```

### Option 2: Docker Compose (Recommended for Production)

```bash
# 1. Start Redis and 20 workers
docker compose -f docker-compose.extraction.yml up -d --scale extraction-worker=20

# 2. Queue extraction jobs (run once)
SEARCH_QUERIES="kredyt frankowy" \
SAMPLE_SIZE=100000 \
docker compose -f docker-compose.extraction.yml --profile coordinator run coordinator

# 3. Monitor progress
docker compose -f docker-compose.extraction.yml logs -f extraction-worker

# 4. Scale workers dynamically (add more workers while running)
docker compose -f docker-compose.extraction.yml up -d --scale extraction-worker=50

# 5. Stop all
docker compose -f docker-compose.extraction.yml down
```

### Option 3: Manual Worker Management (Advanced)

```bash
# 1. Start Redis
redis-server

# 2. Queue jobs (run coordinator once)
python scripts/extraction/coordinator.py \
    --search-queries "kredyt frankowy" "IP Box" \
    --sample-size 100000 \
    --job-batch-size 50 \
    --redis-url redis://localhost:6379

# 3. Start workers (run N times in parallel)
for i in {1..20}; do
    python scripts/extraction/worker.py \
        --worker-id $i \
        --redis-url redis://localhost:6379 \
        --batch-size 3 \
        --model gemini-2.5-pro \
        > logs/worker_${i}.log 2>&1 &
done

# 4. Monitor queue
watch -n 10 'redis-cli LLEN extraction_queue'
```

## Configuration Options

### Coordinator Options

```bash
python scripts/extraction/coordinator.py \
    --search-queries "kredyt frankowy" "IP Box" \    # Multiple queries supported
    --sample-size 100000 \                           # Total documents to extract
    --document-type judgment \                       # Optional: filter by type
    --filter-already-extracted \                     # Skip docs with data
    --job-batch-size 50 \                            # Documents per job
    --redis-url redis://localhost:6379 \             # Redis connection
    --queue-name extraction_queue \                  # Queue name
    --run-name "Swiss Franc Loans" \                 # Optional run label
    --skip-documents 50000 \                         # Skip first N documents (resume capability)
    --monitor                                        # Monitor after queuing
```

### Worker Options

```bash
python scripts/extraction/worker.py \
    --worker-id 1 \                                  # Unique worker ID
    --redis-url redis://localhost:6379 \             # Redis connection
    --queue-name extraction_queue \                  # Queue name
    --batch-size 3 \                                 # Extraction batch size (KEEP SMALL!)
    --model gemini-2.5-pro \                         # Gemini model
    --use-langfuse \                                 # Enable tracing
    --langfuse-sample-rate 0.01                      # Trace 1% of requests
```

## Optimization Tips

### 1. **Batch Size vs Parallelization**

✅ **DO**: Use small batches (3) and many workers (20-200)

```bash
--batch-size 3 --num-workers 100  # GOOD
```

❌ **DON'T**: Use large batches with few workers

```bash
--batch-size 50 --num-workers 5   # BAD - will hit token limits
```

**Why?** Polish legal documents are very long. A single document can be 10K-50K tokens. With the extraction schema and instructions, 3 documents per batch is optimal.

### 2. **Langfuse Tracing**

For large-scale runs, disable Langfuse or use heavy sampling:

```bash
# Option A: Disable completely (fastest)
# Don't pass --use-langfuse flag

# Option B: Use 1% sampling (recommended)
--use-langfuse --langfuse-sample-rate 0.01

# Option C: Use 0.1% sampling (for 1M+ docs)
--use-langfuse --langfuse-sample-rate 0.001
```

This eliminates the "413 Request Entity Too Large" errors from Langfuse.

### 3. **Redis Configuration**

For large-scale extraction (1M+ docs), tune Redis:

```bash
# redis.conf
maxmemory 8gb
maxmemory-policy allkeys-lru
save ""  # Disable RDB snapshots during extraction
appendonly yes
```

### 4. **Worker Placement**

**Best Practice**: Distribute workers across multiple machines

```bash
# Machine 1: 50 workers
for i in {1..50}; do
    python scripts/extraction/worker.py --worker-id $i ... &
done

# Machine 2: 50 workers
for i in {51..100}; do
    python scripts/extraction/worker.py --worker-id $i ... &
done

# Machine 3: 50 workers
for i in {101..150}; do
    python scripts/extraction/worker.py --worker-id $i ... &
done
```

This avoids:

- CPU bottlenecks on a single machine
- Memory pressure
- Network saturation

### 5. **Checkpoint and Resume**

Workers automatically save results to PostgreSQL. If extraction is interrupted, you can resume from where you left off using the `--skip-documents` option:

```bash
# Check how many documents were processed
psql $EXTRACTION_DB_URL -c "
    SELECT COUNT(*) FROM extraction_results
    WHERE run_id='<your_run_id>' AND extraction_status='success'
"

# Option A: Resume by skipping already processed documents
python scripts/extraction/coordinator.py \
    --force-cursor \                    # Important: use cursor pagination to bypass 10K limit
    --skip-documents 50000 \            # Skip first 50K documents already processed
    --sample-size 50000 \               # Extract next 50K documents
    --redis-url redis://localhost:6379

# Option B: Queue only remaining documents (alternative approach)
python scripts/extraction/coordinator.py \
    --filter-already-extracted \
    ...
```

**When to use `--skip-documents`:**

- ✅ **Resume interrupted extraction**: If you've processed 50K documents and need to continue from document 50,001
- ✅ **Distributed processing**: Split work across multiple coordinator runs (e.g., Run 1: docs 0-100K, Run 2: docs 100K-200K)
- ✅ **Testing later documents**: Skip early documents to test extraction on documents further in the collection
- ✅ **Bypass offset limit**: Use with `--force-cursor` to skip beyond Weaviate's 10K offset limit

**Example: Resume from 50K documents**

```bash
# First run (processed 0-50K)
python scripts/extraction/coordinator.py \
    --force-cursor \
    --sample-size 50000 \
    --redis-url redis://localhost:6379

# ... extraction runs ...

# Resume run (process 50K-100K)
python scripts/extraction/coordinator.py \
    --force-cursor \
    --skip-documents 50000 \    # Skip first 50K
    --sample-size 50000 \       # Extract next 50K
    --redis-url redis://localhost:6379
```

## Monitoring

### Check Queue Status

```bash
# Queue length (jobs remaining)
redis-cli LLEN extraction_queue

# Queue length with formatting
redis-cli LLEN extraction_queue | xargs -I{} echo "Jobs remaining: {}"

# Continuous monitoring
watch -n 5 'redis-cli LLEN extraction_queue'
```

### Check Worker Progress

```bash
# View worker logs
tail -f logs/worker_1.log

# Count processed documents per worker
grep "✓ Extracted" logs/worker_*.log | wc -l

# Find errors
grep "ERROR\|Failed" logs/worker_*.log
```

### Check Database Statistics

```sql
-- Total extractions by status
SELECT extraction_status, COUNT(*)
FROM extraction_results
WHERE run_id='<your_run_id>'
GROUP BY extraction_status;

-- Progress by field
SELECT
    field_name,
    SUM(populated) as populated,
    SUM(empty) as empty,
    ROUND(100.0 * SUM(populated) / (SUM(populated) + SUM(empty)), 1) as coverage_pct
FROM field_coverage
WHERE run_id='<your_run_id>'
GROUP BY field_name
ORDER BY coverage_pct DESC;

-- Throughput (docs per minute)
SELECT
    COUNT(*) as total_docs,
    EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) / 60.0 as duration_minutes,
    ROUND(COUNT(*) / (EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) / 60.0), 1) as docs_per_minute
FROM extraction_results
WHERE run_id='<your_run_id>' AND extraction_status='success';
```

## Troubleshooting

### Workers Not Processing

**Symptom**: Queue length not decreasing

```bash
# Check if workers are running
ps aux | grep worker.py

# Check worker logs for errors
tail -f logs/worker_*.log

# Test Redis connection
redis-cli -u redis://localhost:6379 ping
```

### "413 Request Entity Too Large" Errors

**Cause**: Langfuse tracing payloads too large

**Solution**: Reduce Langfuse sampling rate or disable

```bash
--langfuse-sample-rate 0.001  # 0.1% sampling
# OR
# Don't use --use-langfuse flag
```

### "NoneType object has no attribute 'dict'" Errors

**Cause**: Gemini API returned None (rate limit or API error)

**Solution**: Workers automatically retry individual documents. Check logs:

```bash
grep "Batch extraction failed" logs/worker_*.log
```

These are expected occasional errors. Workers handle them gracefully.

### Running Out of Memory

**Solution**: Reduce workers per machine

```bash
# Instead of 100 workers on one machine
# Use 50 workers on two machines
```

## Cost Estimation

**Gemini 2.5 Pro Pricing** (as of 2025):

- Input: $0.00125 per 1K tokens
- Output: $0.005 per 1K tokens

**Per Document Cost**:

- Input: ~15K tokens (document) + ~2K tokens (schema) = 17K tokens
- Output: ~500 tokens (extracted data)
- Cost: (17 × $0.00125) + (0.5 × $0.005) = **$0.024 per document**

**Total Cost Estimates**:

- **100K documents**: 100,000 × $0.024 = **$2,400**
- **1M documents**: 1,000,000 × $0.024 = **$24,000**

**Cost Optimization**:

- Use `gemini-2.5-flash` instead: **~50% cheaper** ($0.012/doc)
- Enable LangChain PostgreSQL cache: **~30-50% savings** on repeated docs

## Next Steps: Ingestion

After extraction completes, ingest results back to Weaviate:

```bash
# Option 1: Use existing script with --ingest-to-weaviate
python scripts/extraction/run_extraction_rest.py \
    --sample-size 0 \  # Don't extract, just ingest
    --ingest-to-weaviate \
    --ingest-batch-size 500

# Option 2: Create separate ingestion worker
# TODO: Implement continuous ingestion worker
```

## References

- **Worker Script**: `scripts/extraction/worker.py`
- **Coordinator Script**: `scripts/extraction/coordinator.py`
- **Docker Compose**: `docker-compose.extraction.yml`
- **Helper Script**: `scripts/extraction/run_distributed.sh`
