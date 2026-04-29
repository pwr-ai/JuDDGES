# Scaling PostgreSQL LLM Cache for High-Concurrency Workloads

This guide explains how to scale the PostgreSQL LLM cache to support 45+ concurrent extraction workers.

## Problem

With 45 workers running extraction jobs, PostgreSQL connection limits are quickly exhausted:

```
ERROR: connection failed: FATAL: sorry, too many clients already
```

**Default PostgreSQL settings:**
- `max_connections`: 150
- Active connections with 45 workers: ~225 (150% of limit!)

## Connection Requirements

Each worker requires multiple PostgreSQL connections:

| Connection Type | Count per Worker | Purpose |
|----------------|------------------|---------|
| LangChain Cache | 1 | Cache initialization |
| Extraction Threads | 3 | Concurrent API calls (max_extraction_threads=3) |
| Storage Connection | 1 | Save extraction results |
| **Total** | **~5** | **Per worker** |

**Calculation for 45 workers:**
- Base connections: 45 workers × 5 = **225 connections**
- Overhead + growth: **75 connections**
- **Total required: 300 connections**

## Solution: Scale PostgreSQL

### Option 1: Quick Restart (Recommended)

Use the automated script to restart PostgreSQL with scaled settings:

```bash
# Stop workers first
pkill -f 'worker.py'

# Restart PostgreSQL with 300 max_connections
./scripts/infrastructure/restart_postgres_scaled.sh

# Restart workers
./start_25_workers_multi_region.sh
```

**New settings applied:**
- `max_connections`: 300
- `shared_buffers`: 512MB (increased for better caching)
- `effective_cache_size`: 1GB
- `maintenance_work_mem`: 128MB

### Option 2: Manual Docker Configuration

If you prefer to configure manually:

```bash
# Stop existing container
docker stop llm-postgres
docker rm llm-postgres

# Start with scaled settings
docker run -d \
  --name llm-postgres \
  --restart unless-stopped \
  -p 5555:5432 \
  -e POSTGRES_USER=llm_cache \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=llm_cache \
  postgres:15.2 \
  postgres \
    -c max_connections=300 \
    -c shared_buffers=512MB \
    -c effective_cache_size=1GB \
    -c maintenance_work_mem=128MB
```

### Option 3: Docker Compose (Permanent)

Update your `docker-compose.yml` to make settings permanent:

```yaml
services:
  llm-postgres:
    image: postgres:15.2
    container_name: llm-postgres
    ports:
      - "5555:5432"
    environment:
      POSTGRES_USER: llm_cache
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: llm_cache
    command:
      - postgres
      - -c
      - max_connections=300
      - -c
      - shared_buffers=512MB
      - -c
      - effective_cache_size=1GB
      - -c
      - maintenance_work_mem=128MB
    restart: unless-stopped
```

Then restart:

```bash
docker compose down llm-postgres
docker compose up -d llm-postgres
```

## Verification

Check that the new settings are applied:

```bash
# Check max_connections
docker exec llm-postgres psql -U llm_cache -d llm_cache -c "SHOW max_connections;"

# Check active connections
docker exec llm-postgres psql -U llm_cache -d llm_cache -c "SELECT count(*) FROM pg_stat_activity;"

# Monitor connections over time
watch -n 5 'docker exec llm-postgres psql -U llm_cache -d llm_cache -t -c "SELECT count(*) FROM pg_stat_activity;"'
```

**Expected output:**
```
 max_connections
-----------------
 300
(1 row)
```

## Performance Tuning

### Memory Settings Explained

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_connections` | 300 | Maximum concurrent client connections |
| `shared_buffers` | 512MB | Memory for caching data pages |
| `effective_cache_size` | 1GB | Estimate of OS cache for query planner |
| `maintenance_work_mem` | 128MB | Memory for maintenance operations (VACUUM, CREATE INDEX) |
| `work_mem` | 4MB | Memory per sort/hash operation |

### Connection Pooling (Optional)

For even better performance, consider using PgBouncer:

```yaml
services:
  pgbouncer:
    image: pgbouncer/pgbouncer
    ports:
      - "6432:5432"
    environment:
      DATABASES_HOST: llm-postgres
      DATABASES_PORT: 5432
      DATABASES_USER: llm_cache
      DATABASES_PASSWORD: ${POSTGRES_PASSWORD}
      DATABASES_DBNAME: llm_cache
      POOL_MODE: transaction
      MAX_CLIENT_CONN: 500
      DEFAULT_POOL_SIZE: 25
```

Then update workers to connect to PgBouncer (port 6432) instead of PostgreSQL directly.

## Monitoring

### Check Connection Usage

```bash
# Current connections by state
docker exec llm-postgres psql -U llm_cache -d llm_cache -c "
  SELECT state, count(*)
  FROM pg_stat_activity
  WHERE datname = 'llm_cache'
  GROUP BY state;
"
```

### Check Cache Hit Ratio

```bash
docker exec llm-postgres psql -U llm_cache -d llm_cache -c "
  SELECT
    sum(blks_hit) / (sum(blks_hit) + sum(blks_read)) * 100 AS cache_hit_ratio
  FROM pg_stat_database
  WHERE datname = 'llm_cache';
"
```

**Target: > 95% cache hit ratio**

### Check Slow Queries

```bash
docker exec llm-postgres psql -U llm_cache -d llm_cache -c "
  SELECT
    query,
    calls,
    mean_exec_time,
    max_exec_time
  FROM pg_stat_statements
  ORDER BY mean_exec_time DESC
  LIMIT 10;
"
```

## Troubleshooting

### Still Getting "Too Many Clients"

1. **Check actual setting:**
   ```bash
   docker exec llm-postgres psql -U llm_cache -d llm_cache -c "SHOW max_connections;"
   ```

2. **Count active workers:**
   ```bash
   ps aux | grep 'worker.py' | grep -v grep | wc -l
   ```

3. **Check connection leaks:**
   ```bash
   docker exec llm-postgres psql -U llm_cache -d llm_cache -c "
     SELECT application_name, count(*)
     FROM pg_stat_activity
     WHERE datname = 'llm_cache'
     GROUP BY application_name
     ORDER BY count DESC;
   "
   ```

### Container Won't Start

If PostgreSQL fails to start with new settings:

```bash
# Check logs
docker logs llm-postgres

# Common issue: shared_buffers too high for available memory
# Reduce shared_buffers or increase Docker memory limit
```

### Performance Degradation

If performance decreases after scaling:

1. **Increase shared_buffers** gradually (up to 25% of RAM)
2. **Add connection pooling** with PgBouncer
3. **Monitor cache hit ratio** and adjust accordingly
4. **Consider scaling to dedicated PostgreSQL server**

## Best Practices

1. **Always stop workers before restarting PostgreSQL** to avoid connection errors
2. **Monitor connection usage** regularly with provided queries
3. **Use connection pooling** for 100+ workers
4. **Set up automated backups** before making infrastructure changes
5. **Document your configuration** in docker-compose.yml or infrastructure-as-code

## Related Documentation

- [Distributed Extraction](../distributed-extraction.md)

## Summary

**Quick Start:**
```bash
# 1. Stop workers
pkill -f 'worker.py'

# 2. Scale PostgreSQL
./scripts/infrastructure/restart_postgres_scaled.sh

# 3. Restart workers
./start_25_workers_multi_region.sh

# 4. Monitor
docker exec llm-postgres psql -U llm_cache -d llm_cache -c "SHOW max_connections;"
```

This scales PostgreSQL from 150 → 300 connections, supporting 45 workers with room for growth.
