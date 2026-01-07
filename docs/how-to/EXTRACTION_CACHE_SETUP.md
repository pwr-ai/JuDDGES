# Extraction Cache Setup

## Current Environment Setup

### Where Scripts Run
**Scripts run on HOST machine** (not in Docker):
- Python: `/home/laugustyniak/github/legal-ai/JuDDGES/.venv/bin/python`
- Working directory: `/home/laugustyniak/github/legal-ai/JuDDGES`
- .env file: Loaded automatically by `scripts/extraction/run_extraction_rest.py` (line 24)

### PostgreSQL Container
**Running in Docker** on host port `5555`:
```bash
Container: Running on localhost
Port: 0.0.0.0:5555->5432/tcp
Cache User: $POSTGRES_CACHE_USER (from .env)
Cache Password: $POSTGRES_CACHE_PASSWORD (from .env)
Cache DB: $POSTGRES_CACHE_DB (from .env)
```

## LLM Cache Configuration

### Cache Strategy
1. **Primary**: PostgreSQL cache (centralized, persistent)
2. **Fallback**: SQLite cache at `.cache/langchain.db` (local)

### Environment Variables (in `.env`)
```bash
# PostgreSQL LLM Cache Configuration
POSTGRES_CACHE_USER=llm_cache
POSTGRES_CACHE_PASSWORD=<your-secure-password>
POSTGRES_CACHE_DB=llm_cache
POSTGRES_CACHE_HOST=localhost
POSTGRES_CACHE_PORT=5555

# Connection String (psycopg3 driver)
POSTGRES_CACHE_URL=postgresql+psycopg://${POSTGRES_CACHE_USER}:${POSTGRES_CACHE_PASSWORD}@${POSTGRES_CACHE_HOST}:${POSTGRES_CACHE_PORT}/${POSTGRES_CACHE_DB}
```

## Setup PostgreSQL Cache

### 1. Create Cache User and Database

```bash
# Connect to PostgreSQL container
docker exec -it <your-postgres-container> psql -U <admin-user> -d <admin-db>

# Inside psql, run (replace with your password):
CREATE USER llm_cache WITH PASSWORD '<your-secure-password>';
CREATE DATABASE llm_cache OWNER llm_cache;
GRANT ALL PRIVILEGES ON DATABASE llm_cache TO llm_cache;

# Grant necessary permissions for LangChain cache table
\c llm_cache
GRANT CREATE ON SCHEMA public TO llm_cache;
GRANT ALL ON SCHEMA public TO llm_cache;

# Exit
\q
```

### 2. Verify Connection

```bash
# Load credentials from .env
source .env

# Test connection from host
PGPASSWORD=$POSTGRES_CACHE_PASSWORD psql \
  -h $POSTGRES_CACHE_HOST \
  -p $POSTGRES_CACHE_PORT \
  -U $POSTGRES_CACHE_USER \
  -d $POSTGRES_CACHE_DB \
  -c "SELECT version();"
```

### 3. Test Cache with Extraction

```bash
# Run extraction - should use PostgreSQL cache
python scripts/extraction/run_extraction_rest.py \
  --sample-size 1 \
  --model gemini-2.5-flash \
  --output-dir data/test_postgres_cache

# Check logs for: "Enabled LangChain PostgreSQL cache (SQLAlchemy): localhost:5433/llm_cache"
```

## Cache Implementation: SQLAlchemyMd5Cache

### Why MD5 Cache?

The extraction script uses **`SQLAlchemyMd5Cache`** instead of the standard `SQLAlchemyCache` to handle large prompts efficiently.

**Problem with standard cache:**
- `SQLAlchemyCache` stores full prompt text in indexed columns
- Large prompts (extraction schemas) exceed PostgreSQL btree index limit (2704 bytes)
- Results in: `ProgramLimitExceeded: index row size exceeds btree maximum`

**Solution with MD5 cache:**
- `SQLAlchemyMd5Cache` stores MD5 hashes of prompts (fixed 32-character length)
- No index size issues regardless of prompt size
- Built-in LangChain solution for large prompts
- Reference: [LangChain SQLAlchemyMd5Cache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.SQLAlchemyMd5Cache.html)

### Schema

The `SQLAlchemyMd5Cache` creates a different table structure optimized for hashed lookups:

```sql
CREATE TABLE IF NOT EXISTS llm_cache_full_md5 (
    id VARCHAR PRIMARY KEY,
    prompt_md5 VARCHAR,  -- MD5 hash of prompt (compact, indexable)
    llm VARCHAR,         -- LLM configuration
    idx INTEGER,
    prompt VARCHAR,      -- Full prompt (not indexed)
    response VARCHAR     -- LLM response
);
```

**Key differences from standard cache:**
- Table name: `llm_cache_full_md5` (not `full_md5_llm_cache`)
- Indexes only on MD5 hashes (not full text)
- No index size limitations

## How It Works

### Cache Initialization (in `run_extraction_rest.py`)

```python
from langchain_community.cache import SQLAlchemyMd5Cache
from langchain.globals import set_llm_cache
from sqlalchemy import create_engine

# Initialize LangChain PostgreSQL Cache using SQLAlchemyMd5Cache
postgres_cache_url = os.getenv("POSTGRES_CACHE_URL")
if postgres_cache_url:
    engine = create_engine(postgres_cache_url)

    # Use SQLAlchemyMd5Cache to store MD5 hashes of prompts
    # This avoids index size limits with large prompts/schemas
    set_llm_cache(SQLAlchemyMd5Cache(engine=engine))

    logger.info("LangChain PostgreSQL MD5 cache initialized successfully")
```

### How Cache Works

1. **Prompt + LLM Config → MD5 Hash**: When making an LLM call, LangChain computes MD5 hash of (prompt + model config)
2. **Cache Lookup**: Checks if hash exists in `llm_cache_full_md5` table
3. **Cache Hit**: Returns stored response immediately (no API call)
4. **Cache Miss**: Calls LLM API, stores response with hash key
5. **Efficiency**: MD5 hash lookups are fast, no size limits on original prompt

## Benefits

### PostgreSQL Cache
✅ **Centralized**: Shared across multiple machines/users
✅ **Persistent**: Survives container restarts
✅ **Scalable**: Handles large cache sizes efficiently
✅ **Queryable**: Can inspect/manage cache entries via SQL

### SQLite Fallback
✅ **No setup required**: Works out of the box
✅ **Local**: Fast for single-user scenarios
✅ **Portable**: Cache file at `.cache/langchain.db`

## Troubleshooting

### PostgreSQL Connection Issues

**Error**: `password authentication failed for user "llm_cache"`
- **Cause**: User/database not created in PostgreSQL
- **Fix**: Run setup commands above

**Error**: `No module named 'psycopg'`
- **Cause**: Missing psycopg3 driver
- **Fix**: `uv pip install 'psycopg[binary]'`

**Warning**: `Failed to initialize PostgreSQL cache, falling back to SQLite`
- **Effect**: Scripts still work with SQLite cache
- **Fix**: Check PostgreSQL container is running and user exists

### Check Cache Status

```bash
# Load credentials from .env
source .env

# View cache entries count
PGPASSWORD=$POSTGRES_CACHE_PASSWORD psql \
  -h $POSTGRES_CACHE_HOST -p $POSTGRES_CACHE_PORT \
  -U $POSTGRES_CACHE_USER -d $POSTGRES_CACHE_DB \
  -c "SELECT COUNT(*) FROM llm_cache_full_md5;"

# View recent cache entries with MD5 hashes
PGPASSWORD=$POSTGRES_CACHE_PASSWORD psql \
  -h $POSTGRES_CACHE_HOST -p $POSTGRES_CACHE_PORT \
  -U $POSTGRES_CACHE_USER -d $POSTGRES_CACHE_DB \
  -c "SELECT prompt_md5, LEFT(prompt::text, 80) as prompt_preview, LENGTH(response::text) as response_length FROM llm_cache_full_md5 ORDER BY id DESC LIMIT 5;"

# View table structure
PGPASSWORD=$POSTGRES_CACHE_PASSWORD psql \
  -h $POSTGRES_CACHE_HOST -p $POSTGRES_CACHE_PORT \
  -U $POSTGRES_CACHE_USER -d $POSTGRES_CACHE_DB \
  -c "\d llm_cache_full_md5"
```

## Docker vs Host Execution

### Current Setup (HOST execution)
- ✅ Scripts run on host with `.venv` Python
- ✅ Connect to PostgreSQL via `localhost:5433`
- ✅ .env file loaded automatically
- ✅ GCP credentials from host

### If Running in Docker (Future)
Would need to:
- Use `POSTGRES_CACHE_URL_DOCKER` instead (connects to `llm-postgres:5432`)
- Mount .env file or pass environment variables
- Mount GCP credentials
- Update connection string in .env or pass as env var

---

## Status

✅ **PostgreSQL MD5 cache configured and working** (localhost:5555/llm_cache)
✅ **Using `SQLAlchemyMd5Cache`** - LangChain's built-in solution for large prompts
✅ **No index size limits** - MD5 hashes are compact and indexable
✅ **Table**: `llm_cache_full_md5` (auto-created on first use)
📝 **Last Updated**: 2025-10-12

### Key Advantages of MD5 Cache

- ✅ **Built-in solution**: No manual schema fixes needed
- ✅ **Handles large prompts**: MD5 hash is fixed 32 chars regardless of prompt size
- ✅ **Official LangChain class**: Well-tested and maintained
- ✅ **Same performance**: Cache lookups are equally fast with hash keys

### Migration Note

The database may contain old cache tables from previous implementations:
- `full_llm_cache` - Legacy cache table
- `full_md5_llm_cache` - Previous implementation with manual index fix

The new `SQLAlchemyMd5Cache` will create `llm_cache_full_md5` on first use. Old tables can be safely dropped to save space:

```bash
# Load credentials from .env
source .env

# Optional: Clean up old cache tables
PGPASSWORD=$POSTGRES_CACHE_PASSWORD psql \
  -h $POSTGRES_CACHE_HOST -p $POSTGRES_CACHE_PORT \
  -U $POSTGRES_CACHE_USER -d $POSTGRES_CACHE_DB \
  -c "DROP TABLE IF EXISTS full_llm_cache, full_md5_llm_cache;"
```
