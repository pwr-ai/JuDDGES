# Multi-Region Worker Deployment

This guide explains how to deploy extraction workers across multiple Vertex AI regions to maximize throughput and avoid rate limiting.

## Overview

Each Vertex AI region has **separate rate limits** for Gemini models. By distributing workers across multiple regions, you can significantly increase your total throughput.

### Benefits

- **5× throughput**: Using 5 regions gives you 5× the rate limit capacity
- **Reduced latency**: Workers automatically fail over between regions
- **No code changes**: Same Redis queue, different regional endpoints
- **Simple scaling**: Add more regions to scale horizontally

## Architecture

```
Redis Queue (Shared)
    │
    ├─► us-central1 (5 workers)    ──► Gemini API (60 RPM)
    ├─► us-east1 (5 workers)       ──► Gemini API (60 RPM)
    ├─► europe-west1 (5 workers)   ──► Gemini API (60 RPM)
    ├─► asia-southeast1 (5 workers)──► Gemini API (60 RPM)
    └─► us-west1 (5 workers)       ──► Gemini API (60 RPM)
                                        ─────────────────
                                        Total: 300 RPM
```

## Available Regions

Gemini models are available in the following regions:

### North America
- `us-central1` (Iowa)
- `us-east1` (South Carolina)
- `us-east4` (Virginia)
- `us-west1` (Oregon)
- `us-west4` (Nevada)

### Europe
- `europe-west1` (Belgium)
- `europe-west2` (London)
- `europe-west3` (Frankfurt)
- `europe-west4` (Netherlands)

### Asia Pacific
- `asia-northeast1` (Tokyo)
- `asia-southeast1` (Singapore)

## Quick Start

### 1. Launch Multi-Region Workers

Use the pre-configured script to launch 25 workers across 5 regions:

```bash
# Ensure Redis is running and REDIS_URL is set
export REDIS_URL="redis://:PASSWORD@localhost:6379/0"

# Launch workers
./start_25_workers_multi_region.sh
```

This starts:
- **Region 1** (us-central1): Workers 1-5
- **Region 2** (us-east1): Workers 6-10
- **Region 3** (europe-west1): Workers 11-15
- **Region 4** (asia-southeast1): Workers 16-20
- **Region 5** (us-west1): Workers 21-25

### 2. Queue Documents for Extraction

```bash
python scripts/extraction/coordinator.py \
    --max-documents 100000 \
    --force-cursor \
    --job-batch-size 10 \
    --redis-url "$REDIS_URL"
```

### 3. Monitor Progress

```bash
# Check running workers
ps aux | grep 'worker.py' | grep -v grep

# Check queue length
redis-cli -u $REDIS_URL LLEN extraction_queue

# View worker logs
tail -f logs/worker_1_us-central1.log
tail -f logs/worker_11_europe-west1.log

# Monitor all workers
./monitor_workers.sh
```

## Configuration Options

### Worker Configuration

Each worker can be configured with:

```bash
python scripts/extraction/worker.py \
    --worker-id 1 \
    --region us-central1 \
    --batch-size 10 \
    --model gemini-2.5-pro \
    --use-langfuse \
    --langfuse-sample-rate 0.01 \
    --max-fetch-threads 10 \
    --max-extraction-threads 3 \
    --redis-url "$REDIS_URL"
```

**Key Parameters:**
- `--region`: Vertex AI region (overrides `VERTEX_LOCATION` env var)
- `--batch-size`: Documents per extraction batch (default: 5, recommended: 10)
- `--model`: Gemini model name (gemini-2.5-pro, gemini-1.5-flash, etc.)
- `--langfuse-sample-rate`: Fraction of requests to trace (0.01 = 1%)
- `--max-extraction-threads`: Concurrent API calls per worker (default: 3)

### Custom Region Configuration

Create a custom launcher script for different regions:

```bash
#!/bin/bash
# start_custom_workers.sh

REGIONS=("us-central1" "europe-west1" "asia-northeast1")
WORKERS_PER_REGION=10

worker_id=1
for region in "${REGIONS[@]}"; do
    for i in $(seq 1 $WORKERS_PER_REGION); do
        python scripts/extraction/worker.py \
            --worker-id $worker_id \
            --region "$region" \
            --batch-size 10 \
            --redis-url "$REDIS_URL" &

        echo "Worker $worker_id started in $region"
        ((worker_id++))
    done
done
```

## Performance Tuning

### Batch Size

Larger batches = fewer API calls but longer processing time per batch:

- **Small (2-5)**: More parallelization, better for rate-limited scenarios
- **Medium (10-15)**: Balanced throughput and latency
- **Large (20+)**: Maximum efficiency but slower individual job completion

**Recommendation**: Use `--batch-size 10` for optimal balance.

### Workers Per Region

Calculate optimal workers based on rate limits:

```
Max Workers per Region = (Rate Limit RPM) / (Requests per Minute per Worker)

Example:
- Rate limit: 60 RPM
- Worker throughput: ~12 RPM (5 docs/batch, 0.4 batches/min)
- Max workers: 60 / 12 = 5 workers per region
```

**Recommendation**: Start with 5 workers per region, monitor rate limit errors.

### Thread Configuration

Each worker has two thread pools:

1. **Fetch threads** (`--max-fetch-threads`): Fetch documents from Weaviate
   - Default: 10
   - Higher = faster document loading

2. **Extraction threads** (`--max-extraction-threads`): Concurrent Gemini API calls
   - Default: 3
   - Higher = more API calls but may trigger rate limits

**Recommendation**: Keep defaults unless you see bottlenecks in logs.

## Monitoring & Troubleshooting

### Check Worker Status

```bash
# List all workers with PIDs
ps aux | grep 'worker.py' | grep -v grep

# Count running workers
ps aux | grep 'worker.py' | grep -v grep | wc -l

# Kill all workers
pkill -f 'worker.py'
```

### Monitor Queue

```bash
# Queue length
redis-cli -u $REDIS_URL LLEN extraction_queue

# Peek at next job (without removing)
redis-cli -u $REDIS_URL LINDEX extraction_queue -1
```

### Check Logs

Worker logs are saved to `logs/worker_<id>_<region>.log`:

```bash
# View live logs
tail -f logs/worker_1_us-central1.log

# Search for errors
grep -i error logs/worker_*.log

# Check rate limit errors
grep -i "429\|rate limit" logs/worker_*.log

# View statistics
grep "Final Statistics" logs/worker_*.log
```

### Common Issues

**Rate Limit Errors (429)**
- Reduce `--max-extraction-threads` (try 2 instead of 3)
- Reduce workers per region (try 3-4 instead of 5)
- Add more regions to distribute load

**Worker Timeout**
- Check network connectivity to Vertex AI
- Verify credentials: `gcloud auth application-default print-access-token`
- Check Weaviate connectivity

**Queue Not Draining**
- Verify workers are running: `ps aux | grep worker.py`
- Check worker logs for errors
- Verify Redis connectivity: `redis-cli -u $REDIS_URL PING`

## Cost Optimization

### Langfuse Sampling

Reduce tracing overhead with sampling:

```bash
--use-langfuse --langfuse-sample-rate 0.01  # Trace 1% of requests
```

### Regional Pricing

Some regions may have different pricing:
- **US regions**: Generally standard pricing
- **Europe/Asia**: May have slight variations

Check [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) for details.

## Advanced Usage

### Dynamic Region Selection

Workers automatically use the region specified in `--region` parameter. You can implement dynamic region selection based on:

1. **Latency**: Measure API response times per region
2. **Availability**: Rotate through regions on errors
3. **Load balancing**: Distribute based on queue length

### Graceful Shutdown

Workers handle `SIGINT` and `SIGTERM` gracefully:

```bash
# Stop specific worker
kill <PID>

# Stop all workers gracefully
pkill -SIGTERM -f 'worker.py'
```

### Resume Interrupted Extractions

If workers crash or are stopped:

1. Check completed documents in PostgreSQL
2. Re-queue incomplete jobs using coordinator with `--skip-documents`
3. Restart workers

```bash
# Skip first 50K documents (already processed)
python scripts/extraction/coordinator.py \
    --max-documents 100000 \
    --skip-documents 50000 \
    --force-cursor
```

## Best Practices

1. **Start with 5 workers per region** (25 total across 5 regions)
2. **Use batch size of 10** for balanced throughput
3. **Enable Langfuse with 1% sampling** for debugging
4. **Monitor rate limit errors** in first 10 minutes
5. **Scale horizontally** by adding more regions, not more workers per region
6. **Use `--force-cursor`** for coordinator to bypass 10K search limit
7. **Check logs regularly** for rate limit or timeout issues

## Example Deployment

Full end-to-end extraction:

```bash
# 1. Start 25 workers across 5 regions
./start_25_workers_multi_region.sh

# 2. Queue 500K documents
python scripts/extraction/coordinator.py \
    --max-documents 500000 \
    --force-cursor \
    --job-batch-size 10 \
    --redis-url "$REDIS_URL"

# 3. Monitor progress
watch -n 5 'redis-cli -u $REDIS_URL LLEN extraction_queue'

# 4. Check worker statistics
grep "documents extracted" logs/worker_*.log

# 5. Stop workers when done
pkill -f 'worker.py'
```

## Related Documentation

- [Extraction Architecture](../../explanation/architecture/extraction-pipeline.md)
- [Monitor Extraction Throughput](./monitor-extraction-throughput.md)
- [Distributed Extraction](../distributed-extraction.md)
- [Weaviate Integration](../../explanation/architecture/WEAVIATE_INTEGRATION.md)
