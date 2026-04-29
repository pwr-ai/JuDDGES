# Weaviate Cursor-Based Pagination

## Overview

The `WeaviateRestClient` now supports **cursor-based pagination** to fetch unlimited documents from Weaviate, removing the 10,000 document hard limit imposed by offset-based pagination.

## Problem

Weaviate has a hard limit with offset-based pagination:

- Maximum offset: `< 10,000`
- Attempting to fetch documents beyond this limit results in: `"query maximum results exceeded"` error

This limitation prevented large-scale extractions (20K+ documents).

## Important Limitation

**Weaviate's cursor API does not support filters or search queries with the `after` parameter.**

When using cursor pagination with:

- `search_query` (hybrid search)
- `document_type_filter` (where clauses)

The system automatically **falls back to offset-based pagination** (10K limit).

## Solution

Cursor-based pagination uses Weaviate's `after` parameter instead of `offset`:

- No 10K limit - can iterate through millions of documents
- More efficient for large datasets
- Uses document UUID as cursor for next page

## When Cursor Pagination Is Available

Cursor pagination works **only** for:

- ✅ Fetching all documents without filters
- ❌ **NOT** with `search_query` (hybrid search)
- ❌ **NOT** with `document_type_filter` (where clauses)

**Why?** Weaviate's API error: `"cursor api: invalid 'after' parameter: other params cannot be set with after and limit parameters"`

When filters/search are needed, the system automatically uses offset pagination (10K limit) with a warning.

## Usage

### Automatic (Default)

By default, `fetch_documents()` uses cursor-based pagination when possible:

```python
from juddges.extraction import WeaviateRestClient

client = WeaviateRestClient.from_env()

# Fetch 20K documents - will automatically use cursor pagination
documents = client.fetch_documents(
    sample_size=20000,
    chunk_size=1000,  # Fetch 1000 docs per request
    search_query="kredyt frankowy",
    document_type_filter="judgment"
)
```

### Explicit Control

You can explicitly control pagination method:

```python
# Use cursor-based pagination (recommended for large datasets)
documents = client.fetch_documents(
    sample_size=100000,
    use_cursor=True  # Default
)

# Use offset-based pagination (legacy, max 10K limit)
documents = client.fetch_documents(
    sample_size=5000,
    use_cursor=False
)
```

## Implementation Details

### Cursor Extraction

The cursor is extracted from the `_additional.id` field of the last document in each batch:

```python
cursor = documents[-1]["_additional"]["id"]
```

### GraphQL Query

Cursor-based queries include the `after` parameter:

```graphql
{
  Get {
    LegalDocuments(
      hybrid: { query: "search term", alpha: 0.5 }
      limit: 1000
      after: "00000000-0000-0000-0000-000000000000"  # Cursor from previous page
    ) {
      document_id
      document_type
      full_text
      language
      document_number
      _additional {
        id  # Required for cursor extraction
      }
    }
  }
}
```

### Pagination Flow

1. **First request**: Query without `after` parameter
2. **Extract cursor**: Get UUID from last document's `_additional.id`
3. **Next request**: Query with `after: "{cursor}"`
4. **Repeat**: Until no more documents or target size reached

### Stopping Conditions

Pagination stops when:

- Target number of documents reached
- No documents returned (end of collection)
- Fewer documents than requested (last page)
- No cursor found in response

## Performance

### Benchmark Comparison

| Method | Max Documents | Performance | Use Case |
|--------|--------------|-------------|----------|
| Offset | 10,000 | Fast for small datasets | < 10K documents |
| Cursor | Unlimited | Efficient for large datasets | 10K+ documents |

### Recommendations

- **< 10K documents**: Either method works
- **10K-100K documents**: Use cursor pagination
- **100K+ documents**: Use cursor pagination with distributed workers

## Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Main method | `juddges/extraction/weaviate_client.py` | 65-102 |
| Cursor implementation | `juddges/extraction/weaviate_client.py` | 104-220 |
| Offset implementation | `juddges/extraction/weaviate_client.py` | 222-343 |
| GraphQL query builder | `juddges/extraction/weaviate_client.py` | 548-629 |

## Example: Large-Scale Extraction

Extract 50,000 documents using cursor pagination:

```bash
# Using coordinator script (uses cursor pagination by default)
python scripts/extraction/coordinator.py \
    --search-queries "kredyt frankowy" "CHF" \
    --sample-size 50000 \
    --job-batch-size 2 \
    --redis-url redis://localhost:6381
```

## Troubleshooting

### Issue: "No cursor found in response"

**Cause**: Missing `_additional { id }` in GraphQL query

**Solution**: The `_build_graphql_query_cursor()` method automatically includes this field.

### Issue: Pagination stops early

**Cause**: Documents filtered out (e.g., empty `full_text`)

**Solution**: Increase the multiplier in `target_size = sample_size * 5` (line 125 in `weaviate_client.py`)

### Issue: Slow pagination

**Cause**: Small chunk size

**Solution**: Increase `chunk_size` parameter:

```python
documents = client.fetch_documents(
    sample_size=100000,
    chunk_size=1000  # Max recommended: 1000
)
```

## Migration Guide

### For Existing Scripts

No changes needed! The cursor-based pagination is enabled by default and backward compatible.

### For Custom Implementations

If you're using `_fetch_documents_offset()` directly:

```python
# OLD (offset-based, max 10K)
documents = client._fetch_documents_offset(
    sample_size=5000,
    chunk_size=1000,
    search_query="test",
    document_type_filter=None,
    random_seed=42
)

# NEW (cursor-based, unlimited)
documents = client._fetch_documents_cursor(
    sample_size=50000,
    chunk_size=1000,
    search_query="test",
    document_type_filter=None,
    random_seed=42
)
```

## References

- [Weaviate Cursor API Documentation](https://weaviate.io/developers/weaviate/api/graphql/additional-operators#cursor-with-after)
- [Weaviate Client Implementation](../../juddges/extraction/weaviate_client.py)
