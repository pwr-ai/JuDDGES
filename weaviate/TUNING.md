# Recommended Weaviate Configuration Changes

## Current vs. Recommended Configuration

### Option 1: Quick Wins (No Migration Required)

Apply these changes to `/home/laugustyniak/github/legal-ai/JuDDGES/weaviate/docker-compose.yaml`:

```yaml
environment:
  # Resource control
  - LIMIT_RESOURCES=true
  - GOMAXPROCS=14  # Increased from 12 for better aggregate performance
  - GOMEMLIMIT=110GiB  # Increased from 100GiB
  - GOGC=75  # Less aggressive GC (was 50) - better for long queries

  # Query limits
  - QUERY_MAXIMUM_RESULTS=10000
  - QUERY_DEFAULTS_LIMIT=25

  # Persistence and LSM optimization
  - PERSISTENCE_DATA_PATH=/var/lib/weaviate
  - PERSISTENCE_LSM_ACCESS_STRATEGY=mmap
  - PERSISTENCE_FLUSH_IDLE_MEMTABLES_AFTER=30
  - PERSISTENCE_HNSW_MAX_LOG_SIZE=1GB

  # Disk management
  - DISK_USE_READONLY_PERCENTAGE=95
  - DISK_USE_WARNING_PERCENTAGE=75

  # Indexing optimization
  - ASYNC_INDEXING=true
  - REINDEX_VECTOR_DIMENSIONS_AT_STARTUP=false
  - TRACK_VECTOR_DIMENSIONS=false

  # Vector cache optimization - INCREASED for better aggregate performance
  - VECTOR_CACHE_MAX_OBJECTS=1000000  # Increased from 500000

  # Batch processing optimization
  - BATCH_DELETE_TIMEOUT=600
  - BATCH_UPDATE_TIMEOUT=600

  # Tombstone cleanup optimization
  - TOMBSTONE_DELETION_INTERVAL=120

  # Connection pool settings
  - GRPC_MAX_MESSAGE_SIZE=104857600

deploy:
  restart_policy:
    condition: on-failure
    max_attempts: 3
  resources:
    limits:
      cpus: '18'  # Increased from 16 for better parallelization
      memory: '120G'  # Increased from 110G to match GOMEMLIMIT
    reservations:
      cpus: '8'
      memory: '64G'
```

**Expected Improvement:** 10-15% faster aggregate queries (~95 seconds instead of 110+)

**Apply changes:**
```bash
cd /home/laugustyniak/github/legal-ai/JuDDGES/weaviate
docker compose down
docker compose up -d
docker compose logs -f weaviate
```

---

### Option 2: Multi-Shard Configuration (Requires Migration)

For maximum aggregate query performance, implement sharding when creating the collection:

**In `juddges/data/documents_weaviate_db.py`:**

```python
def create_collections(self) -> None:
    # Create LegalDocument collection with sharding
    self.safe_create_collection(
        name=self.LEGAL_DOCUMENTS_COLLECTION,
        description="Collection of legal documents",
        properties=[
            # ... all existing properties ...
        ],
        vectorizer_config=[
            # ... existing vectorizer config ...
        ],
        # NEW: Add sharding configuration
        sharding_config=wvcc.Configure.sharding(
            desired_count=4,  # 4 shards for ~800K docs each
            desired_virtual_count=128,
            virtual_per_physical=128
        )
    )
```

**Docker-compose changes for multi-shard:**

```yaml
environment:
  # Resource control - allow more parallelization
  - GOMAXPROCS=20  # Increased to handle 4+ shards
  - GOMEMLIMIT=120GiB
  - GOGC=75

  # Vector cache - distribute across shards
  - VECTOR_CACHE_MAX_OBJECTS=250000  # Per shard, 4 shards = 1M total

  # Other settings same as Option 1

deploy:
  resources:
    limits:
      cpus: '24'  # Increased to handle parallel shard processing
      memory: '128G'
    reservations:
      cpus: '12'
      memory: '80G'
```

**Expected Improvement:** 4-8x faster aggregate queries (15-30 seconds instead of 110+)

**Migration Steps:**

1. Backup current data:
```bash
curl -X POST \
  -H "Authorization: Bearer PQA2.12-**lafqf" \
  http://legal-ai-weaviate.augustyniak.ai:8084/v1/backups/filesystem
```

2. Test on staging first
3. Create new sharded collection
4. Re-ingest data (2-4 hours for 3.1M docs)
5. Verify data integrity
6. Switch application to new collection
7. Delete old collection

---

## Additional Environment Variables to Consider

### For Better Logging During Troubleshooting

```yaml
environment:
  - LOG_LEVEL=info  # or 'debug' for troubleshooting
  - LOG_FORMAT=json
  - QUERY_SLOW_LOG_ENABLED=true
  - QUERY_SLOW_LOG_THRESHOLD=5s  # Log queries taking >5s
```

### For Production Monitoring

```yaml
environment:
  - PROMETHEUS_MONITORING_ENABLED=true
  - PROMETHEUS_MONITORING_PORT=2112
```

---

## Collection Schema Optimizations

### Current Issues

Your LegalDocuments collection has:
- 38 properties (many large JSON strings)
- 3 named vectors (base, dev, fast)
- Many properties with unnecessary indexing

### Recommended Schema Changes

When recreating the collection, optimize:

```python
# Properties that should disable indexing (never filtered/searched)
wvcc.Property(
    name="raw_content",
    data_type=wvcc.DataType.TEXT,
    index_filterable=False,  # Don't index if never filtered
    index_searchable=False,
    skip_vectorization=True
)

# JSON fields that are rarely queried
wvcc.Property(
    name="metadata",
    data_type=wvcc.DataType.TEXT,
    index_filterable=False,  # Disable indexing
    skip_vectorization=True
)

# Consider removing unused named vectors
vectorizer_config=[
    wvcc.Configure.NamedVectors.text2vec_transformers(
        name=VectorName.BASE,  # Keep only the one you actually use
        vectorize_collection_name=False,
        source_properties=["full_text"],
        vector_index_config=wvcc.Configure.VectorIndex.hnsw()
    )
    # Remove 'dev' and 'fast' if not used
]
```

**Benefits:**
- Reduced memory footprint
- Faster indexing during ingestion
- Slightly faster aggregate queries (less data to scan)

---

## Inverted Index Tuning

### For Properties Used in Aggregations

```python
# Optimize country and language properties
wvcc.Property(
    name="country",
    data_type=wvcc.DataType.TEXT,
    index_filterable=True,  # Keep enabled
    index_searchable=False,  # Not needed for aggregations
    tokenization=wvcc.Tokenization.FIELD,  # Change from WORD to FIELD
    # FIELD tokenization treats whole value as single token
    # Better for enum-like values (country codes)
)

wvcc.Property(
    name="language",
    data_type=wvcc.DataType.TEXT,
    index_filterable=True,
    index_searchable=False,
    tokenization=wvcc.Tokenization.FIELD,  # Change from WORD
)
```

**Benefit:** Slightly faster inverted index lookups during aggregation

---

## Summary of Changes by Priority

### Priority 1: Immediate (Apply Today)

**File:** `docker-compose.yaml`

```yaml
- VECTOR_CACHE_MAX_OBJECTS=1000000  # was 500000
- GOMAXPROCS=14  # was 12
- GOGC=75  # was 50

deploy:
  resources:
    limits:
      cpus: '18'  # was 16
      memory: '120G'  # was 110G
```

**Restart:** 5 minutes downtime

**Improvement:** ~10-15% faster

---

### Priority 2: Short-Term (This Week)

**Implement Redis Caching** (see QUICK_FIXES.md)

- Create cache script
- Setup cron job
- Update application to use cache

**No downtime required**

**Improvement:** <10ms response time (with 30min staleness)

---

### Priority 3: Medium-Term (Next Quarter)

**Multi-Shard Migration**

1. Plan migration window (2-4 hours)
2. Update collection creation code
3. Test on staging
4. Execute migration
5. Verify and switch

**Downtime:** 2-4 hours

**Improvement:** 4-8x faster (15-30 seconds)

---

### Priority 4: Long-Term (Ongoing)

**Schema Optimization**

- Remove unused properties
- Disable unnecessary indexing
- Consolidate named vectors
- Optimize tokenization

**Requires:** Major version update with full re-ingestion

**Improvement:** 10-20% resource reduction

---

## Rollback Plan

If any changes cause issues:

```bash
# Restore previous configuration
cd /home/laugustyniak/github/legal-ai/JuDDGES/weaviate
git checkout docker-compose.yaml

# Restart with old config
docker compose down
docker compose up -d
```

For multi-shard migration issues:
- Keep old collection until new one is verified
- Can switch back by updating collection name in application
- Restore from backup if needed

---

## Verification Checklist

After applying changes:

```bash
# 1. Check service health
curl -H "Authorization: Bearer PQA2.12-**lafqf" \
  http://legal-ai-weaviate.augustyniak.ai:8084/v1/.well-known/ready

# 2. Verify collection still accessible
curl -H "Authorization: Bearer PQA2.12-**lafqf" \
  http://legal-ai-weaviate.augustyniak.ai:8084/v1/schema/LegalDocuments

# 3. Test simple count query
curl -X POST \
  -H "Authorization: Bearer PQA2.12-**lafqf" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ Aggregate { LegalDocuments { meta { count } } } }"}' \
  http://legal-ai-weaviate.augustyniak.ai:8084/v1/graphql

# 4. Monitor resource usage
docker stats weaviate --no-stream

# 5. Check logs for errors
docker compose logs weaviate | grep -i error
```

---

## Performance Monitoring

### Before Changes

Capture baseline metrics:

```bash
# Aggregate query time
time curl -X POST [...] # Your aggregate query

# Resource usage
docker stats weaviate --no-stream > baseline_stats.txt

# Memory info
curl -H "Authorization: Bearer PQA2.12-**lafqf" \
  http://legal-ai-weaviate.augustyniak.ai:8084/v1/meta | jq '.memory'
```

### After Changes

Compare metrics:

```bash
# New aggregate query time
time curl -X POST [...] # Same query

# New resource usage
docker stats weaviate --no-stream > after_stats.txt

# Compare
diff baseline_stats.txt after_stats.txt
```

---

## Contact and Support

**Documentation:**
- Full report: `AGGREGATE_QUERY_PERFORMANCE_REPORT.md`
- Quick fixes: `QUICK_FIXES.md`

**Weaviate Resources:**
- Documentation: https://docs.weaviate.io/
- Community Forum: https://forum.weaviate.io/
- GitHub Issues: https://github.com/weaviate/weaviate/issues

**Internal:**
- Schema definition: `juddges/data/documents_weaviate_db.py`
- Configuration: `weaviate/docker-compose.yaml`
- Environment: `weaviate/.env`
