# Quick Fixes for Weaviate Aggregate Query Performance

## Problem
Aggregate queries with `topOccurrences` take 110+ seconds on 3.1M documents.

## Root Cause
- Single shard configuration (no parallelization)
- Query-time aggregation (no pre-computed indexes)
- Must scan all 3.1M documents to count occurrences

---

## FASTEST FIX: Implement Caching (30 minutes)

### 1. Create Cache Script

Create `/home/laugustyniak/github/legal-ai/JuDDGES/weaviate/cache_aggregates.py`:

```python
#!/usr/bin/env python3
import os
import sys
import json
import redis
import requests
from datetime import datetime

WEAVIATE_URL = "http://legal-ai-weaviate.augustyniak.ai:8084"
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
REDIS_URL = "redis://localhost:6381"

def fetch_aggregates():
    query = """
    {
      Aggregate {
        LegalDocuments {
          meta { count }
          country {
            count
            topOccurrences(limit: 10) { value occurs }
          }
          language {
            count
            topOccurrences(limit: 10) { value occurs }
          }
          document_type {
            count
            topOccurrences(limit: 10) { value occurs }
          }
        }
      }
    }
    """

    response = requests.post(
        f"{WEAVIATE_URL}/v1/graphql",
        json={"query": query},
        headers={"Authorization": f"Bearer {WEAVIATE_API_KEY}"},
        timeout=300
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Query failed: {response.status_code}")

def main():
    r = redis.from_url(REDIS_URL)

    print(f"[{datetime.now()}] Fetching aggregates from Weaviate...")
    data = fetch_aggregates()

    cache_key = "weaviate:aggregates:legal_documents:v1"
    enriched = {
        "data": data,
        "cached_at": datetime.utcnow().isoformat(),
        "ttl_seconds": 1800
    }

    r.setex(cache_key, 1800, json.dumps(enriched))
    print(f"[{datetime.now()}] Cached aggregate statistics")

if __name__ == "__main__":
    main()
```

### 2. Make Executable and Test

```bash
cd /home/laugustyniak/github/legal-ai/JuDDGES/weaviate
chmod +x cache_aggregates.py

# Test it (will take 2+ minutes first time)
export WEAVIATE_API_KEY=<YOUR_WEAVIATE_API_KEY>
./cache_aggregates.py
```

### 3. Add to Crontab

```bash
# Edit crontab
crontab -e

# Add this line (runs every 30 minutes)
*/30 * * * * cd /home/laugustyniak/github/legal-ai/JuDDGES/weaviate && export WEAVIATE_API_KEY=<YOUR_WEAVIATE_API_KEY> && ./cache_aggregates.py >> /tmp/cache_aggregates.log 2>&1
```

### 4. Use Cached Data in Your Application

```python
import redis
import json

def get_cached_aggregates():
    """Get statistics from cache (< 10ms)."""
    r = redis.from_url("redis://localhost:6381")
    cache_key = "weaviate:aggregates:legal_documents:v1"

    cached = r.get(cache_key)
    if not cached:
        return None  # Cache warming, try again in 30s

    return json.loads(cached)

# Usage
stats = get_cached_aggregates()
if stats:
    print(f"Total docs: {stats['data']['data']['Aggregate']['LegalDocuments'][0]['meta']['count']}")
    print(f"Cached at: {stats['cached_at']}")
```

**Result:** Query time goes from 110+ seconds to <10ms (acceptable staleness: 30 minutes)

---

## ALTERNATIVE: Split Query (Immediate, No Code Changes)

Instead of:
```graphql
{
  Aggregate {
    LegalDocuments {
      meta { count }
      country { topOccurrences(limit: 5) { value occurs } }
      language { topOccurrences(limit: 5) { value occurs } }
    }
  }
}
```

Run separately:
```graphql
# Query 1: Count only (fast)
{ Aggregate { LegalDocuments { meta { count } } } }

# Query 2: Just country (~60s)
{ Aggregate { LegalDocuments { country { topOccurrences(limit: 5) { value occurs } } } } }

# Query 3: Just language (~60s)
{ Aggregate { LegalDocuments { language { topOccurrences(limit: 5) { value occurs } } } } }
```

**Result:** Each query faster (~60s instead of 110+), can run in parallel

---

## ALTERNATIVE: Use Known Values (If Applicable)

If you know the countries/languages in advance:

```python
from weaviate.classes.query import Filter

def get_country_counts_fast(collection, countries=["PL", "US", "UK", "DE", "FR"]):
    """Get counts for known countries (~5s total)."""
    results = {}

    for country in countries:
        result = collection.aggregate.over_all(
            filters=Filter.by_property("country").equal(country),
            total_count=True
        )
        results[country] = result.total_count

    return sorted(results.items(), key=lambda x: x[1], reverse=True)
```

**Result:** <5 seconds for top 5 countries

---

## BEST LONG-TERM FIX: Multi-Shard (Requires Migration)

**Performance Gain:** 4-8x faster (15-30 seconds instead of 110+)

**Trade-off:** Requires recreating collection and re-ingesting all data

### Steps:

1. Create new collection with sharding config
2. Update `juddges/data/documents_weaviate_db.py`:

```python
def create_collections(self) -> None:
    self.safe_create_collection(
        name=self.LEGAL_DOCUMENTS_COLLECTION,
        # ... existing properties ...
        sharding_config=wvcc.Configure.sharding(
            desired_count=4,  # 4 shards = 4x parallelization
            virtual_per_physical=128
        )
    )
```

3. Re-ingest data (can be parallelized)
4. Update application to use new collection
5. Delete old collection

**Timeline:** 2-4 weeks for planning, testing, and migration

---

## Quick Configuration Tweaks (Minor Improvements)

Edit `/home/laugustyniak/github/legal-ai/JuDDGES/weaviate/docker-compose.yaml`:

```yaml
environment:
  # Increase cache size
  - VECTOR_CACHE_MAX_OBJECTS=1000000  # was 500000

  # Allow more CPU for aggregations
  - GOMAXPROCS=14  # was 12

deploy:
  resources:
    limits:
      cpus: '18'  # was 16
      memory: '120G'  # was 110G
```

Then restart:
```bash
cd /home/laugustyniak/github/legal-ai/JuDDGES/weaviate
docker compose down
docker compose up -d
```

**Expected improvement:** 10-15% faster (minor)

---

## Comparison

| Solution | Time | Staleness | Effort | Risk |
|----------|------|-----------|--------|------|
| **Caching (RECOMMENDED)** | <10ms | 30min | 30min | None |
| Split Queries | ~60s each | Real-time | 0min | None |
| Known Values | <5s | Real-time | 5min | None |
| Multi-Shard | 15-30s | Real-time | 2-4 weeks | High |
| Config Tweaks | ~95s | Real-time | 5min | Low |

---

## Recommendation

**Implement caching immediately** (30 minutes of work) for 1000x speedup. Most use cases don't need real-time statistics.

Plan multi-shard migration for next quarter if real-time accuracy is critical.

---

## Monitoring

After implementing caching, monitor:

```bash
# Check cache hit rate
redis-cli -p 6381 INFO stats | grep keyspace_hits

# Check cache age
redis-cli -p 6381 TTL weaviate:aggregates:legal_documents:v1

# Check cron job logs
tail -f /tmp/cache_aggregates.log
```

---

## Support

For issues or questions:
- Full report: `/home/laugustyniak/github/legal-ai/JuDDGES/weaviate/AGGREGATE_QUERY_PERFORMANCE_REPORT.md`
- Weaviate docs: https://docs.weaviate.io/weaviate/search/aggregate
- Community forum: https://forum.weaviate.io/
