# Optimize Multi-Region Deployment for 45 Workers

## Current Issues & Solutions

### 1. PostgreSQL Connection Pool Exhaustion

**Problem**: `max_connections=150` is too low for 45 workers
- Each worker creates ~3-5 connections (LangChain cache + extraction storage)
- 45 workers × 4 connections = **180 connections** (exceeds 150 limit)

**Solution**: Increase PostgreSQL max_connections

```yaml
# docker-compose.yml
services:
  llm-postgres:
    image: postgres:15.2
    container_name: llm-postgres
    command:
      - postgres
      - -c
      - max_connections=300  # Increased from 150
      - -c
      - shared_buffers=512MB  # Increased from 256MB
      - -c
      - effective_cache_size=2GB  # Increased from 1GB
      - -c
      - maintenance_work_mem=128MB  # Increased from 64MB
      - -c
      - checkpoint_completion_target=0.9
      - -c
      - wal_buffers=16MB
      - -c
      - default_statistics_target=100
      - -c
      - random_page_cost=1.1
      - -c
      - effective_io_concurrency=200
      - -c
      - work_mem=8MB  # Added for parallel queries
    deploy:
      resources:
        limits:
          memory: 4G  # Increased from 2G
          cpus: '4'   # Increased from 2
        reservations:
          memory: 1G  # Increased from 512M
          cpus: '1'
```

**Apply changes**:
```bash
docker compose down
docker compose up -d llm-postgres
```

### 2. Rate Limiting (429 Errors)

**Problem**: Some regions (especially asia-northeast1) hit rate limits faster

**Solutions**:

#### Option A: Reduce Workers per Region (Recommended)
```bash
# Use 3 workers per region instead of 5
WORKERS_PER_REGION=3  # in start_25_workers_multi_region.sh
```

#### Option B: Add Exponential Backoff
Workers already retry with backoff, but you can increase max retries:

```python
# In worker.py, increase max_retries for rate limit errors
max_retries = 5  # From 3
retry_delay = 3.0  # From 2.0 seconds
```

#### Option C: Use Only US/EU Regions
Remove Asia regions which have lower quotas:

```bash
declare -A REGIONS=(
    ["us-central1"]="1 5"
    ["us-east1"]="6 10"
    ["us-east4"]="11 15"
    ["us-west1"]="16 20"
    ["us-west4"]="21 25"
    ["europe-west1"]="26 30"
    ["europe-west2"]="31 35"
    ["europe-west3"]="36 40"
)
# Total: 40 workers in 8 regions
```

### 3. Gemini API Returning None ('NoneType' Errors)

**Problem**: `'NoneType' object has no attribute 'dict'`
- Gemini API occasionally returns None instead of structured output
- This happens with longer documents (30K-70K characters)

**Solutions**:

#### Option A: Reduce max_text_length
```python
# In worker.py:_extract_documents()
extracted_batch = self.chain.batch_extract(
    document_type=doc_type,
    texts=batch_texts,
    schema=self.schema,
    langfuse_handler=self.langfuse_handler if use_langfuse else None,
    max_text_length=100000,  # Reduced from 150000
)
```

#### Option B: Add Better Error Handling
Add validation in `juddges/extraction/gemini_chain.py`:

```python
def extract(self, document_type, text, schema, **kwargs):
    # ... existing code ...

    result = self.chain.invoke(...)

    # Add validation
    if result is None:
        logger.error("Gemini returned None response")
        return {}  # Return empty dict instead of crashing

    if not hasattr(result, 'dict'):
        logger.error(f"Gemini returned invalid response: {type(result)}")
        return {}

    return result.dict()
```

### 4. Langfuse 413 Request Too Large

**Problem**: Batch uploads exceed Langfuse server limits

**Solutions**:

#### Option A: Reduce Langfuse Sampling (Recommended)
```bash
# In start_25_workers_multi_region.sh
LANGFUSE_SAMPLE_RATE=0.05  # 5% instead of 100%
```

#### Option B: Disable Langfuse for Production Runs
```bash
# Remove --use-langfuse flag from worker launch
python scripts/extraction/worker.py \
    --worker-id $worker_id \
    --region "$region" \
    --batch-size 10 \
    # --use-langfuse  # Commented out
```

#### Option C: Increase Langfuse Server Limits
Update your Langfuse deployment with larger request limits.

## Recommended Configuration for 45 Workers

### Updated docker-compose.yml

```yaml
version: '3.8'

services:
  llm-postgres:
    image: postgres:15.2
    container_name: llm-postgres
    command:
      - postgres
      - -c
      - max_connections=300
      - -c
      - shared_buffers=512MB
      - -c
      - effective_cache_size=2GB
      - -c
      - maintenance_work_mem=128MB
      - -c
      - checkpoint_completion_target=0.9
      - -c
      - wal_buffers=16MB
      - -c
      - default_statistics_target=100
      - -c
      - random_page_cost=1.1
      - -c
      - effective_io_concurrency=200
      - -c
      - work_mem=8MB
      - -c
      - max_parallel_workers_per_gather=4
      - -c
      - max_worker_processes=8
    env_file:
      - .env
    healthcheck:
      interval: 10s
      retries: 60
      start_period: 2s
      test: pg_isready -q -d ${POSTGRES_DB} -U ${POSTGRES_USER} | grep "accepting connections" || exit 1
      timeout: 2s
    ports:
      - 5555:5432
    restart: always
    networks:
      - langchain-network
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '4'
        reservations:
          memory: 1G
          cpus: '1'
    volumes:
      - llm-postgres-data:/var/lib/postgresql/data

networks:
  langchain-network:
    driver: bridge

volumes:
  llm-postgres-data:
```

### Updated Worker Configuration

```bash
# start_25_workers_multi_region.sh

# Configuration
BATCH_SIZE=10
MODEL="gemini-2.5-flash"
LANGFUSE_SAMPLE_RATE=0.05  # 5% sampling instead of 100%
REDIS_QUEUE="extraction_queue"
WORKERS_PER_REGION=3  # Reduced from 5 to avoid rate limits

# Define regions (8 regions × 3 workers = 24 workers)
declare -A REGIONS=(
    ["us-central1"]="1 3"
    ["us-east1"]="4 6"
    ["us-east4"]="7 9"
    ["us-west1"]="10 12"
    ["us-west4"]="13 15"
    ["europe-west1"]="16 18"
    ["europe-west2"]="19 21"
    ["europe-west3"]="22 24"
)
```

## Monitoring & Debugging

### Check PostgreSQL Connections

```bash
# Connect to PostgreSQL
docker exec -it llm-postgres psql -U $POSTGRES_USER -d $POSTGRES_DB

# Check active connections
SELECT count(*) FROM pg_stat_activity;

# Check connections by application
SELECT application_name, count(*)
FROM pg_stat_activity
GROUP BY application_name;

# Check max connections setting
SHOW max_connections;

# Kill idle connections (if needed)
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle'
AND state_change < NOW() - INTERVAL '10 minutes';
```

### Monitor Worker Errors

```bash
# Count rate limit errors
grep "429" logs/worker_*.log | wc -l

# Count None errors
grep "NoneType" logs/worker_*.log | wc -l

# Count Langfuse errors
grep "413" logs/worker_*.log | wc -l

# Show error distribution by region
for region in us-central1 us-east1 europe-west1 asia-northeast1; do
    echo "$region: $(grep -l "429" logs/worker_*_${region}.log | wc -l) workers with rate limits"
done
```

### Monitor Extraction Progress

```bash
# Total documents extracted
grep "✓ Extracted" logs/worker_*.log | wc -l

# Documents failed
grep "Failed.*after 3 attempts" logs/worker_*.log | wc -l

# Success rate by worker
for worker_id in {1..45}; do
    success=$(grep "✓ Extracted" logs/worker_${worker_id}_*.log 2>/dev/null | wc -l)
    failed=$(grep "Failed.*after 3 attempts" logs/worker_${worker_id}_*.log 2>/dev/null | wc -l)
    echo "Worker $worker_id: $success success, $failed failed"
done
```

## Performance Optimization

### 1. Connection Pooling

Add connection pooling to reduce database overhead:

```python
# In juddges/extraction/extraction_storage.py
from sqlalchemy.pool import QueuePool

engine = create_engine(
    connection_string,
    poolclass=QueuePool,
    pool_size=5,  # Base connections per worker
    max_overflow=10,  # Additional connections when busy
    pool_pre_ping=True,  # Check connection health
    pool_recycle=3600,  # Recycle connections every hour
)
```

### 2. Batch Size Tuning

Test different batch sizes:

```bash
# Small batches (better for rate limits)
BATCH_SIZE=5

# Medium batches (balanced)
BATCH_SIZE=10

# Large batches (fewer API calls but higher memory)
BATCH_SIZE=15
```

### 3. Worker Distribution Strategy

**Even Distribution** (current):
- 5 workers per region
- Good for balanced load

**Quota-Based Distribution**:
- 6 workers in US regions (higher quotas)
- 4 workers in EU regions
- 2 workers in Asia regions (lower quotas)

### 4. Disable Features Under Load

For maximum throughput:

```bash
# Disable PostgreSQL cache
export POSTGRES_CACHE_URL=""

# Disable Langfuse
# Don't use --use-langfuse flag

# Reduce extraction threads
--max-extraction-threads 2  # From 3
```

## Recommended Action Plan

1. **Immediate**: Update PostgreSQL max_connections to 300
   ```bash
   docker compose down
   # Edit docker-compose.yml
   docker compose up -d llm-postgres
   ```

2. **Reduce Langfuse Sampling**: Change from 100% to 5%
   ```bash
   # Edit start_25_workers_multi_region.sh
   LANGFUSE_SAMPLE_RATE=0.05
   ```

3. **Restart Workers**:
   ```bash
   ./start_25_workers_multi_region.sh
   ```

4. **Monitor for 30 minutes**:
   ```bash
   watch -n 30 'echo "Rate limits: $(grep -c 429 logs/worker_*.log)"; echo "None errors: $(grep -c NoneType logs/worker_*.log)"; echo "Success: $(grep -c "✓ Extracted" logs/worker_*.log)"'
   ```

5. **If still seeing issues**: Reduce to 3 workers per region (24 total)

## Expected Results

With optimized configuration:
- **PostgreSQL**: No connection errors
- **Rate limits**: <5% of requests
- **NoneType errors**: <2% of documents
- **Throughput**: ~400-600 documents/hour with 24-45 workers
- **Langfuse**: No 413 errors with 5% sampling

## Related Documentation

- [Multi-Region Workers](./multi-region-workers.md)
- [Monitor Extraction Throughput](./monitor-extraction-throughput.md)
- [Weaviate Integration](../../explanation/architecture/WEAVIATE_INTEGRATION.md)
