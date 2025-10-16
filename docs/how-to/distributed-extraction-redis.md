# How to Set Up Distributed Extraction with Redis

This guide explains how to coordinate large-scale document extraction across multiple worker processes using Redis as a job queue. The system enables parallel extraction of millions of documents with optimal throughput.

## Architecture Overview

The distributed extraction system consists of two main components:

1. **Coordinator** (`scripts/extraction/coordinator.py`): Fetches document IDs from Weaviate, splits them into small batches, and queues jobs to Redis
2. **Workers** (to be implemented): Poll Redis for jobs, extract documents, and update Weaviate

## Key Design Principles

### Small Batch Sizes for Maximum Parallelization

- **Default batch size: 2-3 documents per job**
- Smaller batches enable more workers to process jobs concurrently
- Reduces risk of job failure (fewer documents lost if a job fails)
- Better load distribution across workers
- Optimal for Gemini API rate limits and latency

### Redis Job Queue

Jobs are stored as JSON in Redis lists with this structure:

```json
{
  "job_id": "uuid-here",
  "run_id": "extraction-run-uuid",
  "document_ids": ["doc-id-1", "doc-id-2"]
}
```

## Prerequisites

### 1. Redis Installation

**Using Docker (Recommended)**:

```bash
docker run -d \
  --name legal-extraction-redis \
  -p 6379:6379 \
  -v redis-data:/data \
  redis:7-alpine \
  redis-server --requirepass YOUR_SECURE_PASSWORD
```

**Using Docker Compose**:

```yaml
services:
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: always

volumes:
  redis-data:
```

**Native Installation** (Ubuntu/Debian):

```bash
sudo apt-get install redis-server
sudo nano /etc/redis/redis.conf  # Set requirepass
sudo systemctl restart redis-server
```

### 2. Environment Variables

Add to your `.env` file:

```bash
# Redis connection (includes password for authentication)
REDIS_URL=redis://:YOUR_SECURE_PASSWORD@localhost:6379/0

# Weaviate connection
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8084
WEAVIATE_API_KEY=your-api-key

# Google Cloud Vertex AI
VERTEX_PROJECT=your-gcp-project
VERTEX_LOCATION=us-central1

# PostgreSQL for extraction tracking
EXTRACTION_POSTGRES_URL=postgresql://user:pass@localhost:5434/legal_extraction
```

**Redis URL Format**:
```
redis://[:password@]host:port/db

Examples:
- With auth: redis://:mypassword@localhost:6379/0
- No auth: redis://localhost:6379/0
- Remote: redis://:password@redis.example.com:6379/0
```

## Usage

### Step 1: Queue Extraction Jobs

Run the coordinator to fetch documents and queue jobs:

```bash
# Basic usage - queue 1000 documents
python scripts/extraction/coordinator.py \
  --search-queries "ip box" "kredyt frankowy" \
  --sample-size 1000 \
  --redis-url redis://:YOUR_PASSWORD@localhost:6379/0
```

**Advanced Options**:

```bash
# Large-scale extraction with custom settings
python scripts/extraction/coordinator.py \
  --search-queries "vat" "podatek dochodowy" \
  --sample-size 1000000 \
  --document-type tax_interpretation \
  --job-batch-size 2 \
  --queue-name extraction_queue_prod \
  --run-name "Tax interpretation extraction Q1 2025" \
  --monitor
```

**Parameters**:

- `--search-queries`: One or more search queries to find documents in Weaviate
- `--sample-size`: Total number of documents to extract (default: 1000)
- `--document-type`: Filter by type: `judgment` or `tax_interpretation`
- `--job-batch-size`: Documents per job (default: 2, recommended: 2-3)
- `--redis-url`: Redis connection string (or use `REDIS_URL` env var)
- `--queue-name`: Redis queue name (default: `extraction_queue`)
- `--run-name`: Optional descriptive name for this extraction run
- `--monitor`: Monitor queue progress after queuing (polls every 30 seconds)
- `--filter-already-extracted`: Skip documents with existing extracted data (not yet implemented)

### Step 2: Run Worker Processes

**Note**: Worker implementation is coming soon. Workers will:

1. Poll the Redis queue for jobs
2. Fetch documents from Weaviate by document IDs
3. Run Gemini extraction using `GeminiExtractionChain`
4. Update Weaviate with extracted fields using batch API
5. Log results to PostgreSQL extraction storage

**Expected worker command** (to be implemented):

```bash
# Run a single worker
python scripts/extraction/worker.py \
  --redis-url redis://:PASSWORD@localhost:6379/0 \
  --queue-name extraction_queue \
  --model gemini-2.5-flash

# Run multiple workers (in separate terminals/screens)
for i in {1..5}; do
  python scripts/extraction/worker.py \
    --redis-url redis://:PASSWORD@localhost:6379/0 \
    --worker-id worker-$i &
done
```

### Step 3: Monitor Progress

The coordinator provides real-time monitoring:

```bash
# Queue jobs and monitor
python scripts/extraction/coordinator.py \
  --search-queries "ip box" \
  --sample-size 10000 \
  --monitor
```

**Manual monitoring**:

```bash
# Check queue length
redis-cli -a YOUR_PASSWORD llen extraction_queue

# Peek at next job (without removing)
redis-cli -a YOUR_PASSWORD lindex extraction_queue -1

# Get queue statistics
redis-cli -a YOUR_PASSWORD info stats
```

## Configuration Details

### Optimal Batch Size Selection

| Batch Size | Use Case | Pros | Cons |
|-----------|----------|------|------|
| 1 | Maximum parallelization | Best worker distribution | Highest queue overhead |
| **2-3** | **Recommended default** | Optimal balance | Good for most cases |
| 5-10 | Moderate parallelization | Lower queue overhead | Fewer concurrent jobs |
| 50+ | Legacy batch processing | Minimal queue ops | Poor parallelization |

**Why 2-3 is optimal**:
- Each worker processes jobs quickly (~30-60 seconds per job)
- Failure of one job only affects 2-3 documents
- Queue can support 100+ workers efficiently
- Matches Gemini API rate limits and latency patterns

### Redis Authentication

The coordinator supports Redis authentication via URL password:

```python
# Automatic authentication from URL
redis_url = "redis://:mypassword@localhost:6379/0"
client = redis.from_url(redis_url, decode_responses=True)
```

**Error Handling**:

```python
try:
    client.ping()
except redis.exceptions.AuthenticationError:
    print("Error: Redis authentication failed")
    print("Ensure REDIS_URL includes password: redis://:PASSWORD@host:port/db")
```

## Best Practices

### 1. Worker Scaling

- **Start small**: Begin with 1-2 workers to test
- **Scale gradually**: Add workers based on monitoring
- **Monitor rate limits**: Respect Gemini API quotas
- **Recommended**: 5-10 workers for most deployments
- **Maximum**: 50-100 workers (requires proper infrastructure)

### 2. Error Handling

- **Job retries**: Workers should re-queue failed jobs (with max retry count)
- **Dead letter queue**: Move permanently failed jobs to separate queue
- **Logging**: Log all failures to PostgreSQL for analysis
- **Monitoring**: Track success/failure rates in real-time

### 3. Resource Management

**Coordinator**:
- Minimal CPU/memory (can run on small instance)
- Only needs network access to Weaviate and Redis

**Workers**:
- CPU: 1-2 cores per worker
- Memory: 2-4GB per worker
- Network: Low latency to Vertex AI and Weaviate
- Recommended: Run on Google Cloud for minimal Vertex AI latency

### 4. Security

- **Always use passwords** for Redis in production
- **Use TLS** for Redis connections in production: `rediss://...`
- **Rotate credentials** regularly
- **Restrict network access** to Redis (firewall/VPC)
- **Use IAM** for Vertex AI authentication (Application Default Credentials)

## Troubleshooting

### Redis Authentication Errors

**Error**: `redis.exceptions.AuthenticationError: Authentication required.`

**Solutions**:

1. **Check password in URL**:
   ```bash
   # Wrong (missing password)
   redis://localhost:6379

   # Correct (with password)
   redis://:mypassword@localhost:6379/0
   ```

2. **Test connection**:
   ```bash
   redis-cli -a YOUR_PASSWORD ping
   # Should return: PONG
   ```

3. **Check environment variable**:
   ```bash
   echo $REDIS_URL
   # Should show: redis://:password@host:port/db
   ```

4. **Verify Redis requires auth**:
   ```bash
   # Check Redis config
   redis-cli config get requirepass
   ```

### Connection Refused

**Error**: `redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379. Connection refused.`

**Solutions**:

1. **Start Redis**:
   ```bash
   # Docker
   docker start legal-extraction-redis

   # Native
   sudo systemctl start redis-server
   ```

2. **Check port**:
   ```bash
   netstat -tlnp | grep 6379
   ```

3. **Verify host/port**:
   ```bash
   # Test connection
   telnet localhost 6379
   ```

### Queue Not Processing

**Symptoms**: Jobs queued but not being processed

**Solutions**:

1. **Check queue length**:
   ```bash
   redis-cli -a PASSWORD llen extraction_queue
   ```

2. **Verify workers are running**:
   ```bash
   ps aux | grep worker.py
   ```

3. **Check worker logs** for errors

4. **Test queue manually**:
   ```bash
   # Pop a job from queue
   redis-cli -a PASSWORD rpop extraction_queue
   ```

### High Memory Usage (Redis)

**Symptoms**: Redis consuming excessive memory

**Solutions**:

1. **Check queue size**:
   ```bash
   redis-cli -a PASSWORD llen extraction_queue
   ```

2. **Set max memory limit**:
   ```bash
   redis-cli config set maxmemory 2gb
   redis-cli config set maxmemory-policy allkeys-lru
   ```

3. **Clear old jobs** (if applicable):
   ```bash
   # Clear completed jobs from separate completed queue
   redis-cli -a PASSWORD del extraction_queue_completed
   ```

## Performance Metrics

### Expected Throughput

With optimal configuration:

- **Coordinator**: Queues 10,000 jobs/minute
- **Single Worker**: 60-120 documents/hour (batch size 2, ~30-60s per job)
- **10 Workers**: 600-1200 documents/hour
- **100 Workers**: 6,000-12,000 documents/hour

### Bottlenecks

1. **Gemini API rate limits**: Primary bottleneck for most deployments
2. **Weaviate write throughput**: Use batch API and async indexing (see [optimize-weaviate-ingestion.md](./optimize-weaviate-ingestion.md))
3. **Network latency**: Run workers in same region as Vertex AI
4. **Worker CPU**: Usually not a bottleneck (extraction is I/O bound)

## Integration with PostgreSQL Storage

The coordinator automatically tracks extraction runs in PostgreSQL:

```python
# Automatic run tracking
run_id = storage.create_extraction_run(
    model_name="gemini-2.5-pro",
    sample_size=1000000,
    batch_size=2,
    max_workers=0,  # Distributed workers (not fixed)
    search_query="ip box; kredyt frankowy",
    notes="Large-scale Q1 2025 extraction",
)
```

**Query extraction runs**:

```sql
-- Get recent extraction runs
SELECT
    run_id,
    model_name,
    sample_size,
    total_documents,
    successful_extractions,
    duration_seconds,
    created_at
FROM extraction_runs
ORDER BY created_at DESC
LIMIT 10;

-- Get field coverage for a run
SELECT
    field_name,
    populated_count,
    empty_count,
    populated_count::float / (populated_count + empty_count) as coverage_rate
FROM extraction_field_coverage
WHERE run_id = 'your-run-id'
ORDER BY coverage_rate DESC;
```

## Advanced Usage

### Multiple Queues for Priority

Run different queues for different priorities:

```bash
# High priority - court judgments
python scripts/extraction/coordinator.py \
  --search-queries "wyrok sądu najwyższego" \
  --sample-size 10000 \
  --queue-name extraction_priority_high

# Low priority - background processing
python scripts/extraction/coordinator.py \
  --search-queries "interpretacja podatkowa" \
  --sample-size 100000 \
  --queue-name extraction_priority_low
```

### Distributed Workers Across Machines

```bash
# Machine 1 (10 workers)
for i in {1..10}; do
  python scripts/extraction/worker.py \
    --redis-url redis://:PASSWORD@redis.example.com:6379/0 \
    --worker-id machine1-worker-$i &
done

# Machine 2 (10 workers)
for i in {1..10}; do
  python scripts/extraction/worker.py \
    --redis-url redis://:PASSWORD@redis.example.com:6379/0 \
    --worker-id machine2-worker-$i &
done
```

### Graceful Shutdown

When stopping workers, ensure jobs are completed:

```bash
# Send SIGTERM (not SIGKILL) to allow graceful shutdown
kill -TERM $(pgrep -f worker.py)

# Workers should:
# 1. Finish current job
# 2. Re-queue job if interrupted
# 3. Close connections
# 4. Exit cleanly
```

## Monitoring Dashboard (Future)

Recommended monitoring stack:

1. **Redis metrics**: Queue length, throughput, memory
2. **Worker metrics**: Jobs/hour, success rate, errors
3. **Weaviate metrics**: Write throughput, index queue length
4. **Vertex AI metrics**: API calls, latency, rate limits
5. **PostgreSQL metrics**: Extraction run statistics, field coverage

**Tools**:
- **Grafana + Prometheus**: Time-series metrics
- **Redis Insight**: Redis monitoring and debugging
- **Weaviate Console**: Weaviate health and performance
- **Google Cloud Monitoring**: Vertex AI metrics

## Related Documentation

- [Optimize Weaviate Ingestion](./optimize-weaviate-ingestion.md) - Weaviate batch API and async indexing
- [Extraction Schema](../reference/extraction-schema.md) - Field definitions and types
- [Gemini Extraction Chain](../reference/gemini-extraction-chain.md) - Model configuration

## Next Steps

1. **Test the coordinator** with small sample size (100-1000 documents)
2. **Implement worker script** (see `scripts/extraction/worker.py` - coming soon)
3. **Set up monitoring** for production deployments
4. **Scale workers** based on throughput requirements
5. **Optimize batch size** for your specific use case

For questions or issues, check the troubleshooting section or review the coordinator source code at `scripts/extraction/coordinator.py`.
