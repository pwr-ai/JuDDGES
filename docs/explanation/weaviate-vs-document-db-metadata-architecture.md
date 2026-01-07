# Weaviate vs Document Database for Evolving Metadata Schemas

> Research conducted: 2025-12-08

## The Core Trade-off

| Aspect | Weaviate (Vector DB) | Document DB (MongoDB/Elasticsearch) |
|--------|---------------------|--------------------------------------|
| **Schema Flexibility** | Limited - new properties work, but pre-existing objects won't be re-indexed | Excellent - fully schemaless evolution |
| **Semantic Search** | Native hybrid search (BM25 + vectors + filters) | Requires add-on (Atlas Vector Search, plugins) |
| **Metadata Filtering** | Strong GraphQL filtering, but schema-enforced | Flexible, dynamic key-value filtering |
| **Backfilling** | Problematic - old objects miss new property indexes | Straightforward - just update documents |

---

## Key Finding: Weaviate Has Significant Schema Evolution Limitations

### Critical Issue for Information Extraction Workflows

When you add new properties to a Weaviate collection after data import:

- Pre-existing objects **won't be automatically re-indexed** with the new property
- The new property index only includes objects added **after** the property was created
- This means queries filtering on new metadata fields will **miss existing documents**

This is a **deal-breaker** if you're continuously extracting new metadata fields from legal documents.

### Additional Weaviate Limitations

- **Immutable settings** after collection creation:
  - Vectorizer configuration
  - Generative module settings
- **indexRangeFilters** only available for properties added before data import
- For every filterable/searchable property, Weaviate creates a dedicated inverted index bucket (memory implications at scale)

---

## Recommended Architecture: Hybrid Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                    Document Database Layer                       │
│  (MongoDB / Elasticsearch)                                       │
│  • Primary document storage                                      │
│  • Flexible metadata schema - add fields anytime                 │
│  • Full metadata filtering and search                            │
│  • Schema versioning with temporal tracking                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓ embeddings sync
┌─────────────────────────────────────────────────────────────────┐
│                    Vector Database Layer                         │
│  (Weaviate - existing setup)                                     │
│  • Semantic/hybrid search on document chunks                     │
│  • Core stable metadata only (court, date, document_type)        │
│  • Use for retrieval, not as source of truth for metadata        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Three Practical Options

### Option 1: Keep Weaviate, Accept Limitations

- Define ALL metadata fields upfront before ingestion
- When adding new fields: **re-ingest entire dataset** (expensive but ensures indexing)
- Best if: Schema is relatively stable, new fields are rare

**Pros:**
- No additional infrastructure
- Simpler architecture

**Cons:**
- Re-ingestion required for new filterable fields
- Must plan schema comprehensively upfront

### Option 2: Hybrid Architecture (Recommended)

- **MongoDB/Elasticsearch**: Store documents + all evolving metadata
- **Weaviate**: Store embeddings + stable core metadata for semantic search
- Retrieve semantically similar docs from Weaviate → fetch full metadata from document DB

**Pros:**
- Best of both worlds
- Continuously add new metadata fields without re-indexing
- Document DB handles complex, nested metadata structures

**Cons:**
- Two systems to maintain
- Synchronization complexity
- Higher infrastructure cost

### Option 3: PostgreSQL + pgvector

- Emerging as unified solution (471 QPS at 99% recall on 50M vectors in 2025 benchmarks)
- Flexible JSONB columns for dynamic metadata
- SQL JOINs, ACID compliance, proper schema migrations

**Pros:**
- Single system for everything
- Mature ecosystem
- Proper schema migrations

**Cons:**
- Migration effort from existing Weaviate setup
- Less specialized for vector search than dedicated vector DBs

---

## Vector Database Comparison (2025)

| Aspect | Weaviate | Pinecone | Qdrant |
|--------|----------|----------|--------|
| **Metadata Model** | Schema-enforced (GraphQL-based) | Dynamic JSON metadata | Dynamic JSON metadata |
| **Filtering Strength** | Flexible GraphQL queries | Strong but simpler API | Best-in-class pre-filtering |
| **Best For** | Complex RAG with relationships | Enterprise production velocity | Complex filters at scale |
| **API** | GraphQL | REST/gRPC | REST/gRPC |
| **Scaling** | Knowledge graphs, hybrid search | Billions of vectors | 10M+ vectors with complex filters |

---

## Information Extraction Pipeline Best Practices

### Schema Versioning Strategy

1. **Semantic Versioning**: Use SemVer for schema definitions
2. **Temporal Versioning**: Add `extracted_at` and `schema_version` fields to track processing history
3. **Schema Registry**: Implement centralized schema definitions as source of truth
4. **Compatibility Rules**:
   - **BACKWARD**: Old consumers can read new data
   - **FORWARD**: New consumers can read old data
   - **FULL**: Bidirectional compatibility (safest)

### Backfilling Existing Documents

When adding new metadata fields to existing documents:

1. **Preparation**:
   - Create full backup before starting
   - Define clear data mapping from source to target fields
   - Test in staging environment first
   - Identify unique keys to prevent duplicates

2. **Execution**:
   - Use batch processing with configurable batch sizes
   - Add throttling between batches to avoid database overload
   - Preserve timestamps for audit trails
   - Monitor progress with rollback procedures ready

3. **Reference**: Netlify successfully backfilled 2 million MongoDB documents at 800+ docs/minute

### LLM-Based Extraction Workflow

```
Raw Documents
    → Document Processing
    → Chunking
    → Embedding → Vector DB (semantic search)
    → Metadata Extraction → Document DB (flexible storage)
    → Retrieval → LLM
```

**Key Patterns:**
- Schema-constrained extraction with deterministic output schemas
- Multi-pass extraction (entity recognition → relationships → tables → validation)
- Self-checking loops with correction for anomalies
- Configuration-driven prompts for easy field updates

---

## Recommendations for JuDDGES Legal AI Project

Given the existing Weaviate setup with `legal_documents` and `document_chunks` collections:

### Immediate Actions

1. **Use Weaviate for semantic search with stable metadata only**
   - Court identifier
   - Document date
   - Document type
   - Language

2. **Add MongoDB/Elasticsearch for evolving extracted metadata**
   - Legal concepts
   - Named entities (judges, parties)
   - Case classifications
   - Custom extracted fields

### Implementation Pattern

```python
# Query flow
1. weaviate_results = semantic_search(query, top_k=100)
2. document_ids = [r.id for r in weaviate_results]
3. enriched_results = mongodb.find({"_id": {"$in": document_ids}})
4. return merge(weaviate_results, enriched_results)
```

### Metadata Schema Design

**Weaviate (stable, rarely changing):**
```yaml
legal_documents:
  properties:
    - court_id: text
    - document_date: date
    - document_type: text
    - language: text
    - content_embedding: vector
```

**MongoDB (evolving, frequently extended):**
```yaml
legal_metadata:
  fields:
    - document_id: ObjectId (links to Weaviate)
    - extracted_at: datetime
    - schema_version: string
    - legal_concepts: array
    - named_entities: object
    - case_classification: object
    - custom_fields: object (dynamic)
```

### Backfill Strategy

When adding new metadata fields:
1. Update document DB first (immediate, no re-indexing needed)
2. Consider periodic Weaviate re-ingestion only for critical filter fields
3. Track schema versions to know which documents have which metadata

---

## Sources

- [Weaviate Collection Definition Documentation](https://weaviate.io/developers/weaviate/config-refs/schema)
- [Weaviate Best Practices](https://docs.weaviate.io/weaviate/best-practices)
- [MongoDB Flexible Schema](https://www.mongodb.com/resources/basics/unstructured-data/schemaless)
- [Elasticsearch Self-querying Retrievers](https://www.elastic.co/search-labs/blog/self-querying-retrievers)
- [Vector Database Comparison 2025](https://toolshelf.tech/blog/pinecone-vs-weaviate-vs-qdrant-vector-database-comparison-2025/)
- [PostgreSQL as Vector Database](https://airbyte.com/data-engineering-resources/postgresql-as-a-vector-database)
- [Schema Evolution in Data Pipelines](https://dataengineeracademy.com/module/best-practices-for-managing-schema-evolution-in-data-pipelines/)
- [Backfilling 2 Million MongoDB Documents](https://www.mongodb.com/developer/products/mongodb/how-netlify-backfilled-2-million-documents/)
- [LLM Document Processing 2025](https://algodocs.com/best-llm-models-for-document-processing-in-2025/)
- [Metadata Filtering in Vector Search](https://www.saumilsrivastava.ai/blog/metadata-filtering-in-vector-search-a-comprehensive-guide-for-engineering-leaders)
