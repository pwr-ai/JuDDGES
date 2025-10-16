# Redis Setup for Distributed Extraction

This document describes the Redis configuration for distributed extraction workloads in the Legal AI project.

## Overview

The `extraction-redis` service provides a dedicated Redis instance for managing distributed extraction job queues. This Redis instance is separate from other Redis instances on the system to avoid conflicts and ensure reliable extraction processing.

## Configuration Details

### Service Name: `extraction-redis`

- **Container Name**: `legal_ai_extraction_redis`
- **Image**: `redis:7-alpine`
- **Port**: `6381` (host) → `6379` (container)
- **Network**: `default` (shared with Weaviate and PostgreSQL)

### Key Features

#### 1. No Persistence (Optimized for Queue Operations)

```yaml
command: redis-server --appendonly no --save ""
```

- **No AOF (Append-Only File)**: Disabled for better performance
- **No RDB snapshots**: Queue data is ephemeral and doesn't need persistence
- **Why**: Job queue data is temporary; permanent results are stored in PostgreSQL

#### 2. Memory Management

```yaml
--maxmemory 2gb --maxmemory-policy allkeys-lru
```

- **Max memory**: 2GB limit (4GB container limit with buffer)
- **Eviction policy**: `allkeys-lru` (Least Recently Used)
- **Why**: Prevents memory exhaustion; old jobs are evicted if memory is full

#### 3. Health Checks

```yaml
healthcheck:
  test: ["CMD", "redis-cli", "ping"]
  interval: 5s
  timeout: 3s
  retries: 5
```

- **Automatic monitoring**: Docker checks Redis health every 5 seconds
- **Graceful recovery**: 5 retries before marking as unhealthy
- **Integration**: Other services can wait for Redis to be healthy

#### 4. Resource Limits

```yaml
resources:
  limits:
    cpus: '2'
    memory: '4G'
```

- **CPU**: 2 cores (sufficient for queue operations)
- **Memory**: 4GB container limit (2GB Redis + 2GB buffer)

#### 5. Logging

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

- **Log rotation**: 3 files × 10MB = 30MB max
- **Prevents disk space issues** from unbounded log growth

## Usage

### Start Redis Service

```bash
# Start all services (including Redis)
cd weaviate
docker compose up -d

# Start only Redis
docker compose up -d extraction-redis

# Check Redis status
docker compose ps extraction-redis
docker compose logs extraction-redis
```

### Connect to Redis

```bash
# Using redis-cli
redis-cli -p 6381

# Check if Redis is running
redis-cli -p 6381 ping
# Expected: PONG

# View memory usage
redis-cli -p 6381 INFO memory

# Monitor commands in real-time
redis-cli -p 6381 MONITOR
```

### Queue Operations

```bash
# Check queue length
redis-cli -p 6381 LLEN extraction_queue

# View all keys
redis-cli -p 6381 KEYS '*'

# View specific job data (example)
redis-cli -p 6381 GET "job:abc123"

# Clear all data (use with caution!)
redis-cli -p 6381 FLUSHALL
```

## Integration with Extraction Scripts

All extraction scripts default to using this Redis instance via the URL:

```
redis://localhost:6381
```

### Distributed Extraction

```bash
# Run distributed extraction (automatically uses port 6381)
./scripts/extraction/run_distributed.sh \
    --num-workers 20 \
    --sample-size 1000 \
    --search-queries "kredyt frankowy"
```

### Worker Connection

Workers connect to Redis using:

```python
redis_client = redis.from_url("redis://localhost:6381", decode_responses=True)
```

### Coordinator Connection

The coordinator pushes jobs to Redis using the same URL:

```python
redis_client = redis.from_url("redis://localhost:6381")
```

## Monitoring

### Health Check

```bash
# Check Docker health status
docker compose ps extraction-redis

# Expected output:
# NAME                         STATUS
# legal_ai_extraction_redis    Up (healthy)
```

### Memory Usage

```bash
# Check current memory usage
redis-cli -p 6381 INFO memory | grep used_memory_human

# Check memory statistics
redis-cli -p 6381 MEMORY STATS
```

### Queue Statistics

```bash
# Number of items in queue
redis-cli -p 6381 LLEN extraction_queue

# Total keys in Redis
redis-cli -p 6381 DBSIZE

# Redis stats
redis-cli -p 6381 INFO stats
```

### RedisInsight Web UI

If you have RedisInsight running:

```bash
# Start RedisInsight (if not already running)
docker run -d --name redis-insight \
    -p 5540:5540 \
    redis/redisinsight:latest

# Access via browser
open http://localhost:5540

# Add connection:
# - Host: localhost
# - Port: 6381
# - Name: Legal AI Extraction Redis
```

## Troubleshooting

### Redis Not Starting

**Check logs:**

```bash
docker compose logs extraction-redis
```

**Common issues:**

1. Port 6381 already in use

   ```bash
   # Check what's using the port
   sudo lsof -i :6381

   # Change port in docker-compose.yaml if needed
   ```

2. Volume permission issues

   ```bash
   # Remove volume and recreate
   docker compose down
   docker volume rm legal_ai_extraction_redis_data
   docker compose up -d extraction-redis
   ```

### Memory Issues

**Check if Redis is evicting keys:**

```bash
redis-cli -p 6381 INFO stats | grep evicted_keys
```

**If eviction count is high:**

```yaml
# Increase maxmemory in docker-compose.yaml
command: redis-server --appendonly no --save "" --maxmemory 4gb --maxmemory-policy allkeys-lru

# Then restart
docker compose up -d extraction-redis
```

### Queue Not Draining

**Check queue length:**

```bash
redis-cli -p 6381 LLEN extraction_queue
```

**If queue is stuck:**

1. Check if workers are running:

   ```bash
   pgrep -f "worker.py" | wc -l
   ```

2. Check worker logs:

   ```bash
   tail -f logs/worker_*.log
   ```

3. Manually inspect queue items:

   ```bash
   # Peek at last item without removing
   redis-cli -p 6381 LINDEX extraction_queue -1
   ```

4. Clear queue if needed:

   ```bash
   redis-cli -p 6381 DEL extraction_queue
   ```

### Connection Refused

**Verify Redis is listening:**

```bash
# Check if port is open
nc -zv localhost 6381

# Check Redis container networking
docker compose exec extraction-redis redis-cli ping
```

**From within Docker network:**

```bash
# Other containers can access Redis via service name
redis://extraction-redis:6379
```

## Performance Tuning

### For High-Throughput Workloads (100+ workers)

Increase memory and CPU limits:

```yaml
resources:
  limits:
    cpus: '4'
    memory: '8G'

command: redis-server --appendonly no --save "" --maxmemory 6gb --maxmemory-policy allkeys-lru
```

### For Memory-Constrained Environments

Reduce memory allocation:

```yaml
command: redis-server --appendonly no --save "" --maxmemory 1gb --maxmemory-policy allkeys-lru

resources:
  limits:
    memory: '2G'
```

## Data Persistence (Optional)

If you need to persist queue data across restarts:

```yaml
# Change command to enable RDB snapshots
command: redis-server --save 60 1 --maxmemory 2gb --maxmemory-policy allkeys-lru

# This will save to disk every 60 seconds if at least 1 key changed
```

**Note**: Not recommended for queue operations, as it impacts performance and queue data is typically ephemeral.

## Migration from Temporary Redis

If you were using the temporary `extraction-redis-test` container:

```bash
# Stop old temporary container
docker stop extraction-redis-test
docker rm extraction-redis-test

# Start new managed Redis via docker-compose
cd weaviate
docker compose up -d extraction-redis

# No migration needed - queue data is ephemeral
# Just update scripts to use redis://localhost:6381
```

## Security Considerations

### Current Setup (Development)

- **No authentication**: Redis is accessible without password on localhost
- **Local only**: Port 6381 is bound to localhost, not exposed externally
- **Acceptable for**: Development and internal use

### Production Recommendations

1. **Enable authentication**:

   ```yaml
   command: redis-server --requirepass ${REDIS_PASSWORD} --appendonly no --save ""
   environment:
     REDIS_PASSWORD: ${REDIS_PASSWORD}
   ```

2. **Use internal networking only**:

   ```yaml
   # Remove external port mapping
   # ports:
   #   - "6381:6379"

   # Let other services access via internal network
   ```

3. **Enable TLS** (for production):

   ```yaml
   command: redis-server --tls-port 6379 --port 0 --tls-cert-file /certs/redis.crt --tls-key-file /certs/redis.key
   ```

## Environment Variables

Add to `weaviate/.env`:

```bash
# Redis for extraction queue
EXTRACTION_REDIS_HOST=localhost
EXTRACTION_REDIS_PORT=6381
EXTRACTION_REDIS_URL=redis://localhost:6381

# Optional: if adding authentication
# EXTRACTION_REDIS_PASSWORD=your_secure_password
```

## Related Documentation

- [Distributed Extraction Guide](../docs/how-to/distributed-extraction.md)
- [Error Monitoring Guide](../docs/how-to/monitor-extraction-errors.md)
- [Weaviate Integration](../docs/explanation/architecture/WEAVIATE_INTEGRATION.md)

## Summary

The `extraction-redis` service provides:

- ✅ Dedicated Redis instance for extraction queues
- ✅ Optimized for high-throughput queue operations
- ✅ Automatic health monitoring
- ✅ Memory management with LRU eviction
- ✅ Easy integration with extraction scripts
- ✅ Production-ready configuration with resource limits

Start it with:

```bash
cd weaviate && docker compose up -d extraction-redis
```

Monitor it with:

```bash
redis-cli -p 6381 MONITOR
```
