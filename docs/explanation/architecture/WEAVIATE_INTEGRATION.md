# Weaviate Vector Database Integration

## Overview

Weaviate serves as the semantic search engine and vector database for JuDDGES, enabling efficient similarity search, context retrieval, and document management. This document details the integration architecture, schema design, and operational workflows.

## Integration Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        Docs[("Legal Documents<br/>Parquet Files")]
        Embeddings[("Generated Embeddings<br/>768-dim vectors")]
    end

    subgraph "Weaviate Infrastructure"
        subgraph "Docker Container"
            WeaviateCore["Weaviate Core<br/>v1.24+"]
            GRPC["gRPC Server<br/>Port 8085"]
            REST["REST API<br/>Port 8080"]
            GraphQL["GraphQL API<br/>Port 8080"]
        end

        subgraph "Storage Backend"
            VectorIndex["Vector Indices<br/>HNSW Algorithm"]
            ObjectStore["Object Storage<br/>Document Metadata"]
            InvertedIndex["Inverted Index<br/>Keyword Search"]
        end

        subgraph "Modules"
            Text2Vec["text2vec-transformers<br/>Embedding Module"]
            QnA["qna-transformers<br/>Question Answering"]
            Reranker["reranker-transformers<br/>Result Reranking"]
        end
    end

    subgraph "Client Applications"
        Ingestion["Ingestion Script<br/>ingest_to_weaviate.py"]
        Inference["Inference Pipeline<br/>Context Retrieval"]
        Analytics["Analytics<br/>Document Analysis"]
    end

    Docs --> Ingestion
    Embeddings --> Ingestion
    Ingestion --> REST
    REST --> ObjectStore
    REST --> VectorIndex

    Inference --> GraphQL
    GraphQL --> VectorIndex
    VectorIndex --> Reranker
    Reranker --> Inference

    Analytics --> GraphQL
    GraphQL --> InvertedIndex

    style WeaviateCore fill:#e8f5e9
    style VectorIndex fill:#e3f2fd
    style ObjectStore fill:#f3e5f5
```

## Schema Design

### Collection: `legal_documents`

```mermaid
classDiagram
    class LegalDocument {
        +UUID document_id
        +String signature
        +String court_name
        +Date judgment_date
        +String document_text
        +String[] judges
        +String[] keywords
        +String legal_basis
        +String decision_type
        +Float[] embedding_vector
        +String metadata_json
        +Timestamp created_at
        +Timestamp updated_at
    }

    class DocumentMetadata {
        +String source_file
        +String language
        +Integer word_count
        +String document_type
        +String jurisdiction
        +Boolean is_published
    }

    LegalDocument "1" --> "1" DocumentMetadata : contains

    style LegalDocument fill:#e3f2fd
    style DocumentMetadata fill:#f3e5f5
```

### Collection: `document_chunks`

```mermaid
classDiagram
    class DocumentChunk {
        +UUID chunk_id
        +UUID document_id
        +String chunk_text
        +Integer chunk_index
        +Integer start_char
        +Integer end_char
        +Float[] embedding_vector
        +String chunk_type
        +Float relevance_score
    }

    class ChunkMetadata {
        +String section_title
        +String paragraph_type
        +Integer token_count
        +String[] entities
        +String[] citations
    }

    DocumentChunk "1" --> "1" ChunkMetadata : contains
    DocumentChunk "n" --> "1" LegalDocument : belongs_to

    style DocumentChunk fill:#e8f5e9
    style ChunkMetadata fill:#fff3e0
```

## Data Ingestion Pipeline

```mermaid
flowchart TD
    subgraph "Ingestion Process"
        Start["Start Ingestion"]

        Load["Load Parquet Files<br/>• Document batches<br/>• Metadata"]

        Validate["Validate Schema<br/>• Required fields<br/>• Data types"]

        GenUUID["Generate UUIDs<br/>• Deterministic<br/>• Based on content hash"]

        Batch["Create Batches<br/>• Size: 100 objects<br/>• Memory efficient"]

        subgraph "Weaviate Operations"
            Check{"Object Exists?"}
            Update["Update Object<br/>• Merge metadata<br/>• Update embedding"]
            Create["Create Object<br/>• New document<br/>• Full metadata"]
        end

        Commit["Commit Batch<br/>• Atomic operation<br/>• Error handling"]

        More{"More Batches?"}
        Complete["Ingestion Complete<br/>• Log statistics"]
    end

    Start --> Load
    Load --> Validate
    Validate --> GenUUID
    GenUUID --> Batch
    Batch --> Check
    Check -->|Yes| Update
    Check -->|No| Create
    Update --> Commit
    Create --> Commit
    Commit --> More
    More -->|Yes| Batch
    More -->|No| Complete

    style Start fill:#e8f5e9
    style Complete fill:#ffebee
    style Check fill:#fff3e0
```

## Query Architecture

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Weaviate
    participant VectorIndex
    participant Reranker

    Client->>API: Query Request
    API->>API: Parse query & extract intent

    alt Semantic Search
        API->>Weaviate: Vector search query
        Weaviate->>VectorIndex: Find similar vectors
        VectorIndex-->>Weaviate: Top-k results
        Weaviate->>Reranker: Rerank results
        Reranker-->>Weaviate: Reranked results
    else Hybrid Search
        API->>Weaviate: Hybrid query (vector + keyword)
        Weaviate->>VectorIndex: Vector search
        Weaviate->>Weaviate: Keyword search
        Weaviate->>Weaviate: Merge & score results
    else Aggregate Query
        API->>Weaviate: Aggregation query
        Weaviate->>Weaviate: Calculate aggregates
    end

    Weaviate-->>API: Query results
    API->>API: Post-process results
    API-->>Client: Formatted response
```

## Query Types and Patterns

### 1. Semantic Search

```mermaid
graph LR
    Query["User Query:<br/>'contracts breach damages'"]
    Embed["Generate Query<br/>Embedding"]
    Search["Vector Similarity<br/>Search"]
    Filter["Apply Filters:<br/>• Date range<br/>• Court type<br/>• Jurisdiction"]
    Rank["Rank by:<br/>• Cosine similarity<br/>• Relevance score"]
    Results["Return Top-K<br/>Documents"]

    Query --> Embed
    Embed --> Search
    Search --> Filter
    Filter --> Rank
    Rank --> Results

    style Query fill:#e3f2fd
    style Results fill:#e8f5e9
```

### 2. Hybrid Search

```mermaid
graph TB
    subgraph "Hybrid Search Components"
        Input["Search Query"]

        subgraph "Vector Search Path"
            VEmbed["Query Embedding"]
            VSearch["Vector Search<br/>α = 0.7"]
        end

        subgraph "Keyword Search Path"
            KTokenize["Tokenization"]
            KSearch["BM25 Search<br/>α = 0.3"]
        end

        Fusion["Score Fusion<br/>weighted combination"]
        Output["Ranked Results"]
    end

    Input --> VEmbed
    Input --> KTokenize
    VEmbed --> VSearch
    KTokenize --> KSearch
    VSearch --> Fusion
    KSearch --> Fusion
    Fusion --> Output

    style Input fill:#e3f2fd
    style Output fill:#e8f5e9
    style Fusion fill:#fff3e0
```

### 3. Context Retrieval for RAG

```mermaid
flowchart LR
    subgraph "RAG Context Pipeline"
        Question["User Question"]

        subgraph "Retrieval"
            ChunkSearch["Search Chunks<br/>Top-20"]
            DocSearch["Search Documents<br/>Top-5"]
            Combine["Combine Results"]
        end

        subgraph "Processing"
            Dedup["Deduplicate"]
            Sort["Sort by Relevance"]
            Truncate["Fit Context Window"]
        end

        Context["Final Context<br/>for LLM"]
    end

    Question --> ChunkSearch
    Question --> DocSearch
    ChunkSearch --> Combine
    DocSearch --> Combine
    Combine --> Dedup
    Dedup --> Sort
    Sort --> Truncate
    Truncate --> Context

    style Question fill:#e3f2fd
    style Context fill:#e8f5e9
```

## Performance Optimization

```mermaid
graph TD
    subgraph "Optimization Strategies"
        subgraph "Indexing"
            HNSW["HNSW Index<br/>• ef_construction: 128<br/>• max_connections: 16"]
            Sharding["Data Sharding<br/>• By date<br/>• By court"]
        end

        subgraph "Caching"
            QueryCache["Query Cache<br/>• TTL: 5 minutes<br/>• LRU eviction"]
            EmbedCache["Embedding Cache<br/>• Pre-computed<br/>• Frequently accessed"]
        end

        subgraph "Batch Operations"
            BatchIngest["Batch Ingestion<br/>• 100 objects/batch<br/>• Parallel processing"]
            BatchQuery["Batch Queries<br/>• Multi-get<br/>• Aggregations"]
        end

        subgraph "Resource Management"
            ConnPool["Connection Pooling<br/>• Max: 100<br/>• Timeout: 30s"]
            RateLimit["Rate Limiting<br/>• 1000 req/min<br/>• Backoff strategy"]
        end
    end

    style HNSW fill:#e3f2fd
    style QueryCache fill:#e8f5e9
    style BatchIngest fill:#fff3e0
    style ConnPool fill:#f3e5f5
```

## Docker Deployment

```mermaid
graph TB
    subgraph "Docker Compose Stack"
        Network["Docker Network<br/>weaviate_default"]

        subgraph "Weaviate Service"
            Container["weaviate:latest<br/>Container"]
            Volumes["Volumes<br/>• /var/lib/weaviate<br/>• ./data:/data"]
            Ports["Ports<br/>• 8080:8080<br/>• 8085:50051"]
            Env["Environment<br/>• ENABLE_MODULES<br/>• PERSISTENCE_DATA_PATH"]
        end

        subgraph "Dependencies"
            T2V["text2vec-transformers<br/>Container"]
            QnAModule["qna-transformers<br/>Container"]
        end
    end

    Network --> Container
    Container --> Volumes
    Container --> Ports
    Container --> Env
    Container --> T2V
    Container --> QnAModule

    style Network fill:#e8f5e9
    style Container fill:#e3f2fd
```

## Monitoring and Metrics

```mermaid
graph LR
    subgraph "Metrics Collection"
        Weaviate --> Prometheus["Prometheus<br/>Metrics Endpoint"]
        Prometheus --> Grafana["Grafana<br/>Dashboards"]

        subgraph "Key Metrics"
            M1["Query Latency"]
            M2["Index Size"]
            M3["Memory Usage"]
            M4["Query Rate"]
            M5["Error Rate"]
        end

        Grafana --> M1
        Grafana --> M2
        Grafana --> M3
        Grafana --> M4
        Grafana --> M5
    end

    style Prometheus fill:#fff3e0
    style Grafana fill:#e8f5e9
```

## Error Handling

```mermaid
stateDiagram-v2
    [*] --> Healthy

    Healthy --> ConnectionError: Connection lost
    ConnectionError --> Retry: Automatic
    Retry --> Healthy: Success
    Retry --> Retry: Failed < 3
    Retry --> Error: Failed >= 3

    Healthy --> RateLimit: Too many requests
    RateLimit --> Backoff: Exponential
    Backoff --> Healthy: After delay

    Healthy --> SchemaError: Invalid data
    SchemaError --> Validation: Check schema
    Validation --> Transform: Fix data
    Transform --> Healthy: Retry

    Error --> Manual: Alert sent
    Manual --> Healthy: Fixed

    Error --> [*]
```

## Best Practices

1. **UUID Generation**: Use deterministic UUIDs based on document content
2. **Batch Size**: Optimal batch size is 100 objects for ingestion
3. **Connection Pooling**: Reuse connections for better performance
4. **Error Recovery**: Implement exponential backoff for transient errors
5. **Schema Evolution**: Version schemas and migrate carefully
6. **Monitoring**: Track query latency and index performance
7. **Backup**: Regular backups of Weaviate data directory

## API Examples

### Creating a Collection

```python
# Schema definition for legal_documents
schema = {
    "class": "LegalDocument",
    "properties": [
        {"name": "signature", "dataType": ["text"]},
        {"name": "court_name", "dataType": ["text"]},
        {"name": "judgment_date", "dataType": ["date"]},
        {"name": "document_text", "dataType": ["text"]},
        {"name": "judges", "dataType": ["text[]"]},
        {"name": "keywords", "dataType": ["text[]"]}
    ],
    "vectorizer": "none"  # Using pre-computed embeddings
}
```

### Querying Documents

```graphql
# GraphQL query for semantic search
{
  Get {
    LegalDocument(
      nearVector: {
        vector: [0.1, 0.2, ...]
        certainty: 0.8
      }
      limit: 10
    ) {
      signature
      court_name
      judgment_date
      _additional {
        certainty
        distance
      }
    }
  }
}
```

## Related Documentation

- [System Architecture](./SYSTEM_ARCHITECTURE.md)
- [Data Flow Pipeline](./DATA_FLOW_PIPELINE.md)
- [Embedding Generation](../../how-to/embeddings/generate_embeddings.md)
- [Weaviate Setup Guide](../../how-to/setup/weaviate_setup.md)
