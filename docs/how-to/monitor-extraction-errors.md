# How to Monitor and Troubleshoot Extraction Errors

This guide explains how to monitor extraction jobs and troubleshoot common errors using the enhanced error logging system.

## Overview

The extraction system includes comprehensive error logging and automatic retry mechanisms to handle transient API failures gracefully.

## Error Types

The system automatically detects and categorizes the following error types:

### 1. Rate Limit Errors (429)
**Symptoms:**
- HTTP 429 status code
- Error message contains "rate limit"
- Multiple workers hitting API limits simultaneously

**System Response:**
- Automatically retries with exponential backoff (2s, 4s, 8s)
- Up to 3 attempts per document
- Falls back to individual document processing

**Example Log:**
```
ERROR | Batch extraction failed: {'error_type': 'ValueError', 'likely_cause': 'Rate limit exceeded', 'http_code': 429, 'batch_size': 3}
WARNING | [Worker 5] Rate limit detected - will retry individual docs with backoff
WARNING | [Worker 5] Attempt 1/3 failed for doc_123: Rate limit exceeded. Retrying in 2.0s...
```

### 2. API Server Errors (500, 502, 503)
**Symptoms:**
- HTTP 5xx status codes
- Temporary Gemini API unavailability
- Server-side processing errors

**System Response:**
- Automatically retries with exponential backoff
- Falls back to individual processing
- Logs error details for monitoring

**Example Log:**
```
ERROR | Batch extraction failed: {'error_type': 'InternalServerError', 'likely_cause': 'API server error', 'http_code': 503}
WARNING | [Worker 12] API server error - will retry individual docs
```

### 3. Timeout Errors
**Symptoms:**
- Request timeout message
- Very long documents (>100K tokens)
- Slow API responses

**System Response:**
- Retries with longer timeouts
- May need to truncate very long documents
- Check `max_text_length` parameter (default: 150,000 chars)

**Example Log:**
```
ERROR | Extraction failed: {'error_type': 'TimeoutError', 'likely_cause': 'Request timeout', 'text_length': 158217}
WARNING | Text length 158217 exceeds max 150000, truncating
```

### 4. None Response Errors
**Symptoms:**
- Error: `'NoneType' object has no attribute 'dict'`
- API returns None instead of structured response
- Usually indicates rate limits or API errors

**System Response:**
- Explicit None check before processing response
- Detailed error logging with context
- Automatic retry with backoff

**Example Log:**
```
ERROR | Extraction failed: API returned None - likely rate limit, timeout, or API error
WARNING | [Worker 8] Attempt 2/3 failed for doc_456. Retrying in 4.0s...
```

## Monitoring Commands

### Check Worker Status
```bash
# Count active workers
pgrep -f "worker.py" | wc -l

# View real-time logs
tail -f logs/worker_*.log | grep ERROR
```

### Check Redis Queue
```bash
# Queue length
redis-cli -p 6381 LLEN extraction_queue

# Queue status
redis-cli -p 6381 INFO stats
```

### Analyze Error Rates
```bash
# Count total errors
grep "ERROR" logs/worker_*.log | wc -l

# Count failed documents
grep "Failed.*:" logs/worker_*.log | wc -l

# Count unique failed documents
grep "Failed.*:" logs/worker_*.log | cut -d: -f3- | sort -u | wc -l

# Show error breakdown by type
grep "likely_cause" logs/worker_*.log | cut -d: -f3- | sort | uniq -c | sort -rn
```

### Check Database Progress
```bash
# Total extracted documents
psql -d legal_extraction -c "SELECT COUNT(*) FROM extraction_results WHERE extraction_status = 'success';"

# Failed extractions
psql -d legal_extraction -c "SELECT COUNT(*) FROM extraction_results WHERE extraction_status = 'failed';"

# Success rate by run_id
psql -d legal_extraction -c "
  SELECT
    run_id,
    COUNT(*) FILTER (WHERE extraction_status = 'success') as successful,
    COUNT(*) FILTER (WHERE extraction_status = 'failed') as failed,
    ROUND(100.0 * COUNT(*) FILTER (WHERE extraction_status = 'success') / COUNT(*), 2) as success_rate
  FROM extraction_results
  GROUP BY run_id
  ORDER BY created_at DESC
  LIMIT 5;
"
```

## Troubleshooting

### High Failure Rate (>10%)

**Likely Causes:**
1. Rate limiting - too many workers
2. API quota exhausted
3. Network issues
4. Very long documents

**Solutions:**
```bash
# Reduce number of workers
./scripts/extraction/run_distributed.sh --num-workers 10  # instead of 50

# Increase batch processing delay
# Edit worker.py: time.sleep(5) between batches

# Check API quota
gcloud alpha billing accounts describe BILLING_ACCOUNT_ID

# Review failed document lengths
psql -d legal_extraction -c "
  SELECT document_id, LENGTH(full_text) as text_length
  FROM extraction_results
  WHERE extraction_status = 'failed'
  ORDER BY text_length DESC
  LIMIT 10;
"
```

### Workers Stuck or Not Processing

**Check:**
```bash
# Worker process status
ps aux | grep worker.py

# Redis connection
redis-cli -p 6381 PING

# Database connection
psql -d legal_extraction -c "SELECT 1;"

# Weaviate connection
curl http://localhost:8080/v1/meta
```

**Solutions:**
```bash
# Restart workers
pkill -f worker.py
./scripts/extraction/run_distributed.sh --num-workers 20

# Clear Redis queue (if needed)
redis-cli -p 6381 DEL extraction_queue

# Check PostgreSQL connections
psql -d legal_extraction -c "SELECT * FROM pg_stat_activity WHERE datname = 'legal_extraction';"
```

### Retry Logic Not Working

**Verify Code Version:**
```bash
# Check if enhanced error handling is present
grep -n "exponential backoff" scripts/extraction/worker.py
grep -n "likely_cause" juddges/extraction/gemini_chain.py

# If not found, pull latest changes
git pull origin feat/umap-calc
```

**Restart Workers:**
```bash
# Stop old workers
pkill -f worker.py

# Start new workers with updated code
./scripts/extraction/run_distributed.sh --num-workers 20 --redis-url redis://localhost:6381
```

## Performance Optimization

### Recommended Settings by Scale

| Documents | Workers | Batch Size | Expected Time |
|-----------|---------|------------|---------------|
| 1K        | 10      | 3          | 15 minutes    |
| 10K       | 20      | 3          | 2 hours       |
| 100K      | 50      | 3          | 8-12 hours    |
| 1M        | 100     | 3          | 3-5 days      |

### Tuning Parameters

**Increase throughput:**
```bash
# More workers (watch for rate limits)
--num-workers 50

# Parallel batch processing (if API allows)
--batch-size 5  # instead of 3
```

**Reduce errors:**
```bash
# Fewer workers to avoid rate limits
--num-workers 10

# Enable Langfuse tracing for debugging
--use-langfuse --langfuse-sample-rate 0.1
```

**Cost optimization:**
```bash
# Use Gemini Flash instead of Pro
--model gemini-2.5-flash

# Sample rate for tracing
--langfuse-sample-rate 0.01  # 1% of requests
```

## Error Log Format

### Enhanced Error Log Structure

The system logs errors with structured context:

```python
{
    "error_type": "ValueError",           # Exception type
    "error_message": "API returned None", # Error description
    "document_type": "judgment",          # Document type
    "text_length": 45000,                 # Document size
    "batch_size": 3,                      # Batch size (if applicable)
    "http_code": 429,                     # HTTP status code (if available)
    "status_code": 429,                   # Alternative status code field
    "likely_cause": "Rate limit exceeded" # Automatic error categorization
}
```

### Example Full Error Trace

```
2025-10-13 11:15:23.456 | ERROR | juddges.extraction.gemini_chain:batch_extract:180 -
Batch extraction failed: ValueError - API returned None for 1/3 documents - likely rate limit or API error |
Details: {
  'error_type': 'ValueError',
  'error_message': 'API returned None for 1/3 documents - likely rate limit or API error',
  'document_type': 'judgment',
  'batch_size': 3,
  'likely_cause': 'Rate limit exceeded'
}

2025-10-13 11:15:23.457 | WARNING | __main__:_extract_documents:371 -
[Worker 5] Rate limit detected - will retry individual docs with backoff

2025-10-13 11:15:23.458 | WARNING | __main__:_extract_documents:104 -
[Worker 5] Attempt 1/3 failed for doc_123: API returned None. Retrying in 2.0s...

2025-10-13 11:15:25.500 | INFO | juddges.extraction.gemini_chain:extract:74 -
Successfully extracted 16 fields from judgment using structured output

2025-10-13 11:15:25.501 | DEBUG | __main__:_extract_documents:79 -
[Worker 5] ✓ Extracted doc_123
```

## Best Practices

1. **Start Small**: Test with `--sample-size 100` before scaling to thousands
2. **Monitor Continuously**: Use `tail -f logs/worker_*.log | grep ERROR` during runs
3. **Check Success Rate**: Aim for >90% success rate; investigate if lower
4. **Use Checkpointing**: Results are saved to PostgreSQL; jobs can be resumed
5. **Enable Tracing Selectively**: Use low `--langfuse-sample-rate` (1-5%) to reduce overhead
6. **Scale Gradually**: Increase workers incrementally to find optimal throughput
7. **Budget Rate Limits**: Google Vertex AI has per-project rate limits; coordinate with team

## Related Documentation

- [Distributed Extraction Guide](./distributed-extraction.md) - Setup and architecture
- [Weaviate Integration](../explanation/architecture/WEAVIATE_INTEGRATION.md) - Document storage
- [Extraction Schema Reference](../reference/schemas/extraction_schema.md) - Field definitions
