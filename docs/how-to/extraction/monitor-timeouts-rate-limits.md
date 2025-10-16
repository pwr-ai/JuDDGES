# Monitor Extraction Timeouts and Rate Limits

This guide explains how to detect and troubleshoot timeouts and rate limiting issues during large-scale extraction operations.

## Overview

During extraction, you may encounter:
- **Rate Limiting (429 errors)**: Too many API requests to Vertex AI
- **Timeouts**: Requests taking too long to complete
- **Server Errors (5xx)**: Vertex AI service issues

The extraction system includes **automatic retry logic** with exponential backoff to handle these issues gracefully.

## Quick Health Check

### Check Current Worker Health

```bash
# Single snapshot
python scripts/extraction/check_worker_health.py

# Live monitoring
python scripts/extraction/check_worker_health.py --live

# Monitor queue consumption rate
python scripts/extraction/check_worker_health.py --monitor-rate --duration 60
```

### Check for Errors and Rate Limits

```bash
# Analyze recent logs
python scripts/extraction/check_errors_and_limits.py

# Analyze specific log file
python scripts/extraction/check_errors_and_limits.py --log-file logs/extraction.log

# Check database for specific run
python scripts/extraction/check_errors_and_limits.py --run-id <run_id>

# Check last 48 hours
python scripts/extraction/check_errors_and_limits.py --hours 48
```

## Built-in Error Handling

### Automatic Retry Logic

The worker includes automatic retry with exponential backoff:

```python
# From scripts/extraction/worker.py
max_retries = 3
retry_delay = 2.0  # seconds

for attempt in range(max_retries):
    try:
        # Extract document
        extracted = chain.extract(...)
        break  # Success!
    except Exception as e:
        if attempt < max_retries - 1:
            # Exponential backoff: 2s, 4s, 8s
            backoff_time = retry_delay * (2 ** attempt)
            time.sleep(backoff_time)
```

### Error Detection

The system automatically detects and categorizes errors:

```python
# Rate limiting detection
is_rate_limit = "429" in error_msg or "rate limit" in error_msg.lower()

# Timeout detection
is_timeout = "timeout" in error_msg.lower()

# Server error detection
is_server_error = "500" in error_msg or "503" in error_msg
```

## Understanding the Output

### Worker Health Check Output

```
Redis Queue Status
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Metric                       ┃ Value        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━┫
┃ Status                       ┃ ● Online     ┃
┃ Queue Length                 ┃ 2,345        ┃
┃ Response Time                ┃ 2.34 ms      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┛

Worker Statistics
┏━━━━━━━━━━┳━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Worker   ┃ Jobs ┃ Success┃ Failed ┃ Rate (d/min) ┃ Status  ┃
┣━━━━━━━━━━╋━━━━━━╋━━━━━━━━╋━━━━━━━━╋━━━━━━━━━━━━━━╋━━━━━━━━━┫
┃ Worker 1 ┃  125 ┃  1,200 ┃     15 ┃         48.5 ┃ ● Active┃
┃ Worker 2 ┃  118 ┃  1,150 ┃     22 ┃         45.2 ┃ ● Active┃
┃ Worker 3 ┃   12 ┃    115 ┃      3 ┃          4.8 ┃ ● Slow  ┃
┗━━━━━━━━━━┻━━━━━━┻━━━━━━━━┻━━━━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━━┛

Health Assessment
✓ All systems appear healthy
```

**Interpretation**:
- **● Active**: Worker processing > 10 docs/min
- **● Slow**: Worker processing 1-10 docs/min (possible issues)
- **● Idle**: Worker processing < 1 doc/min (likely stalled)

### Error Check Output

```
Log File Analysis: extraction.log

┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Error Type              ┃ Count ┃ Status         ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━╋━━━━━━━━━━━━━━━━┫
┃ Rate Limiting (429)     ┃    15 ┃ ⚠ DETECTED     ┃
┃ Timeouts                ┃     8 ┃ ⚠ DETECTED     ┃
┃ Server Errors (5xx)     ┃     2 ┃ ⚠ DETECTED     ┃
┃ Retry Attempts          ┃    45 ┃ ↻ Active       ┃
┃ Total Errors            ┃    87 ┃                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━┻━━━━━━━━━━━━━━━━┛

Recommendations:
⚠ Rate Limiting Detected
  • Reduce --max-workers (try 3-5 instead of 10)
  • Increase --batch-size (process more docs per request)
  • Add delays between batches
  • Check Vertex AI quota limits

⚠ Timeout Errors Detected
  • Reduce max_text_length (current: 150000)
  • Decrease --batch-size (try 3-5 for long documents)
  • Check network connectivity to Vertex AI
```

## Common Issues and Solutions

### 1. Rate Limiting (429 Errors)

**Symptoms**:
- Logs show "429" or "rate limit" errors
- Worker throughput drops significantly
- Many retry attempts

**Solutions**:

```bash
# Reduce parallelization
python scripts/extraction/run_extraction_rest.py \
  --max-workers 3 \
  --batch-size 10 \
  --max-documents 10000
```

**Settings to adjust**:
- `--max-workers`: Reduce from 10 to 3-5
- `--batch-size`: Increase to process more docs per API call
- Add delays: Modify worker to add `time.sleep(1)` between batches

**Check quotas**:
```bash
# Check Vertex AI quotas
gcloud alpha quotas list \
  --service=aiplatform.googleapis.com \
  --project=<your-project>
```

### 2. Timeout Errors

**Symptoms**:
- Logs show "timeout" or "timed out" messages
- Extraction stalls on long documents
- Inconsistent throughput

**Solutions**:

```bash
# Process smaller batches with reduced text length
python scripts/extraction/run_extraction_rest.py \
  --max-workers 5 \
  --batch-size 3 \
  --max-documents 10000
```

**Code changes** (if needed):

Edit `scripts/extraction/worker.py`:
```python
# Reduce max text length
extracted = self.chain.extract(
    ...
    max_text_length=100000,  # Reduced from 150000
)
```

Edit `juddges/extraction/gemini_chain.py`:
```python
# Increase timeout in request
timeout=300,  # 5 minutes instead of default
```

**Network diagnostics**:
```bash
# Test network latency to Vertex AI
ping -c 10 <vertex-ai-endpoint>

# Check DNS resolution
nslookup <vertex-ai-endpoint>
```

### 3. Server Errors (500, 503)

**Symptoms**:
- Logs show "500", "503", or "502" errors
- Intermittent failures
- Multiple retry attempts

**Solutions**:

These are usually temporary Vertex AI issues. The retry logic will handle them automatically.

**Monitoring**:
```bash
# Check Vertex AI status
gcloud status

# Monitor error patterns
python scripts/extraction/check_errors_and_limits.py --hours 6
```

**If persistent**:
- Check [Google Cloud Status Dashboard](https://status.cloud.google.com/)
- Consider switching models temporarily
- Contact GCP support if sustained

### 4. Queue Stalled

**Symptoms**:
- Queue length not decreasing
- Workers show 0 docs/min
- No recent logs

**Solutions**:

```bash
# Check worker processes
ps aux | grep worker

# Check Redis connectivity
redis-cli -u redis://localhost:6379 ping

# Restart workers
# Kill existing workers
pkill -f "extraction/worker.py"

# Start new workers
./scripts/extraction/run_distributed.sh
```

## Monitoring Best Practices

### 1. Set Up Continuous Monitoring

Create a monitoring script that runs periodically:

```bash
#!/bin/bash
# monitor_extraction.sh

while true; do
    echo "=== $(date) ==="

    # Check worker health
    python scripts/extraction/check_worker_health.py

    # Check for errors
    python scripts/extraction/check_errors_and_limits.py --hours 1

    # Wait 5 minutes
    sleep 300
done
```

Run in background:
```bash
nohup ./monitor_extraction.sh > monitoring.log 2>&1 &
```

### 2. Set Up Alerts

Monitor key metrics and alert on thresholds:

```python
# alert_on_errors.py
import time
from juddges.extraction import ExtractionStorage

storage = ExtractionStorage()

while True:
    # Check error rate (last hour)
    stats = check_database_errors(storage, hours=1)

    error_rate = stats["total_failed"] / 100.0  # Assuming ~100 docs/hour

    if error_rate > 0.1:  # > 10% error rate
        send_alert(f"High error rate: {error_rate*100:.1f}%")

    if stats["rate_limit_errors"] > 5:
        send_alert("Rate limiting detected!")

    time.sleep(600)  # Check every 10 minutes
```

### 3. Log Rotation

Ensure logs don't fill up disk:

```bash
# Setup logrotate
sudo cat > /etc/logrotate.d/extraction <<EOF
/home/user/extraction/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 user user
}
EOF
```

## Performance Tuning

### Optimal Settings by Use Case

**High Volume, Fast Processing**:
```bash
--max-workers 8 \
--batch-size 10 \
--model gemini-2.5-flash
```
- Expected: 40-60 docs/min per worker
- Risk: Higher chance of rate limiting
- Use when: Processing large batches quickly

**Stable, Long-Running**:
```bash
--max-workers 5 \
--batch-size 5 \
--model gemini-2.5-pro
```
- Expected: 20-30 docs/min per worker
- Risk: Lower chance of issues
- Use when: Running overnight/weekend jobs

**Long Documents**:
```bash
--max-workers 3 \
--batch-size 3 \
--model gemini-2.5-flash
```
- Expected: 15-25 docs/min per worker
- Risk: Timeouts with large batches
- Use when: Documents > 50KB

### Redis Configuration

Optimize Redis for high-throughput:

```bash
# In redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save ""  # Disable persistence for speed
```

### Vertex AI Quotas

Check and request quota increases:

```bash
# View current quotas
gcloud alpha quotas list \
  --service=aiplatform.googleapis.com \
  --filter="metric:aiplatform.googleapis.com/prediction_requests_per_minute"

# Request increase (via web console)
# https://console.cloud.google.com/iam-admin/quotas
```

## Troubleshooting Workflow

When encountering issues, follow this workflow:

1. **Check Worker Health**
   ```bash
   python scripts/extraction/check_worker_health.py
   ```

2. **Check for Errors**
   ```bash
   python scripts/extraction/check_errors_and_limits.py
   ```

3. **Review Recent Logs**
   ```bash
   tail -100 logs/extraction.log | grep -E "(ERROR|WARNING|429|timeout)"
   ```

4. **Check Database Stats**
   ```bash
   python scripts/extraction/monitor_extraction_throughput.py
   ```

5. **Take Action** based on findings:
   - **Rate limiting** → Reduce max-workers
   - **Timeouts** → Reduce batch-size or max_text_length
   - **Server errors** → Wait and monitor
   - **Queue stalled** → Restart workers

## Related Documentation

- [Monitor Extraction Throughput](monitor-extraction-throughput.md)
- [Distributed Extraction](../distributed-extraction.md)
- [Monitor Extraction Errors](../monitor-extraction-errors.md)

## Debugging Commands Quick Reference

```bash
# Check worker health
python scripts/extraction/check_worker_health.py

# Check for errors
python scripts/extraction/check_errors_and_limits.py

# Monitor throughput
python scripts/extraction/monitor_extraction_throughput.py --live

# Check Redis queue
redis-cli -u redis://localhost:6379 llen extraction_queue

# View recent worker logs
tail -f logs/extraction.log | grep "Worker"

# Check worker processes
ps aux | grep "extraction/worker.py"

# Kill all workers
pkill -f "extraction/worker.py"
```
