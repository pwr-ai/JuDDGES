# Text Chunker

Intelligent text chunking for document segmentation with token-aware splitting and overlap support.

## Overview

The `juddges.preprocessing.text_chunker` module provides the `TextChunker` class for splitting long legal documents into manageable chunks for embedding generation and LLM processing.

## Key Features

- **Recursive Character Splitting**: Smart splitting that respects document structure
- **Token-Aware Chunking**: Uses tokenizer for accurate token counts
- **Configurable Overlap**: Maintain context between chunks
- **Minimum Length Filtering**: Remove too-short chunks
- **First N Chunks**: Limit chunks for testing
- **Batch Processing**: Efficient processing of multiple documents

## Usage Examples

### Basic Chunking

```python
from juddges.preprocessing.text_chunker import TextChunker

# Initialize chunker
chunker = TextChunker(
    id_col="judgment_id",
    text_col="full_text",
    chunk_size=512,
    chunk_overlap=50
)

# Chunk documents (HuggingFace Dataset format)
dataset = {
    "judgment_id": ["doc1", "doc2"],
    "full_text": ["Long text...", "Another long text..."]
}

chunked = chunker(dataset)
print(chunked)
# {
#     "judgment_id": ["doc1", "doc1", "doc1", "doc2", "doc2"],
#     "chunk_id": [0, 1, 2, 0, 1],
#     "chunk_len": [512, 512, 300, 512, 450],
#     "chunk_text": ["First chunk...", "Second chunk...", ...]
# }
```

### Token-Based Chunking

```python
from transformers import AutoTokenizer

# Use tokenizer for accurate token counts
tokenizer = AutoTokenizer.from_pretrained("sdadas/mmlw-roberta-large")

chunker = TextChunker(
    id_col="judgment_id",
    text_col="full_text",
    chunk_size=512,  # 512 tokens
    chunk_overlap=50,
    tokenizer=tokenizer  # Token-aware splitting
)

chunked = chunker(dataset)
# Each chunk is approximately 512 tokens
```

### Advanced Configuration

```python
chunker = TextChunker(
    id_col="judgment_id",
    text_col="full_text",
    chunk_size=1024,
    chunk_overlap=100,
    min_split_chars=100,  # Filter out chunks shorter than 100 chars
    take_n_first_chunks=5  # Only keep first 5 chunks (for testing)
)

chunked = chunker(dataset)
# Only returns first 5 chunks per document, all >= 100 chars
```

### Integration with HuggingFace Datasets

```python
from datasets import load_dataset
from juddges.preprocessing.text_chunker import TextChunker

# Load dataset
ds = load_dataset("juddges/pl-court-raw", split="train[:100]")

# Initialize chunker
chunker = TextChunker(
    id_col="judgment_id",
    text_col="full_text",
    chunk_size=512,
    chunk_overlap=50
)

# Chunk dataset
chunked_ds = ds.map(
    chunker,
    batched=True,
    batch_size=10,
    remove_columns=ds.column_names
)

print(chunked_ds[0])
# {
#     "judgment_id": "doc1",
#     "chunk_id": 0,
#     "chunk_len": 512,
#     "chunk_text": "First chunk of document..."
# }
```

## API Reference

::: juddges.preprocessing.text_chunker.TextChunker
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## Chunking Strategies

### Recursive Character Splitting

The default strategy uses LangChain's `RecursiveCharacterTextSplitter`:

1. **Try splitting on paragraphs** (`\n\n`)
2. **Fall back to sentences** (`.`, `!`, `?`)
3. **Fall back to words** (spaces)
4. **Fall back to characters** (individual chars)

This preserves document structure and creates semantically meaningful chunks.

### Token-Based vs Character-Based

**Character-Based** (default):

```python
chunker = TextChunker(
    id_col="doc_id",
    text_col="text",
    chunk_size=512  # 512 characters
)
```

**Token-Based** (recommended for embeddings):

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model-name")
chunker = TextChunker(
    id_col="doc_id",
    text_col="text",
    chunk_size=512,  # 512 tokens
    tokenizer=tokenizer
)
```

### Overlap Strategies

**No Overlap**:

```python
chunker = TextChunker(
    chunk_size=512,
    chunk_overlap=0
)
# Chunks: [0-512], [512-1024], [1024-1536]
```

**With Overlap** (recommended):

```python
chunker = TextChunker(
    chunk_size=512,
    chunk_overlap=50  # 50 token/char overlap
)
# Chunks: [0-512], [462-974], [924-1436]
# Maintains context between chunks
```

**Typical Overlap Values**:

- **10-20%**: Light overlap, more chunks
- **20-50%**: Moderate overlap (recommended)
- **50%+**: Heavy overlap, fewer missed contexts

## Performance Optimization

### Batch Processing

Process multiple documents efficiently:

```python
from datasets import load_dataset

ds = load_dataset("juddges/pl-court-raw")

# Batch processing
chunked_ds = ds.map(
    chunker,
    batched=True,
    batch_size=100,  # Process 100 docs at a time
    num_proc=8       # Use 8 CPU cores
)
```

### Memory Management

For large datasets:

```python
# Stream processing
ds = load_dataset("juddges/pl-court-raw", streaming=True)

chunked_ds = ds.map(
    chunker,
    batched=True,
    batch_size=10
)

# Process in batches
for batch in chunked_ds.iter(batch_size=100):
    process_batch(batch)
```

## Configuration Best Practices

### Embedding Generation

```python
# For embedding models (e.g., BERT, RoBERTa)
chunker = TextChunker(
    id_col="judgment_id",
    text_col="full_text",
    chunk_size=512,       # Match model's max length
    chunk_overlap=50,     # 10% overlap
    min_split_chars=100,  # Filter very short chunks
    tokenizer=tokenizer   # Use model's tokenizer
)
```

### LLM Processing

```python
# For LLM input (e.g., Llama, Mistral)
chunker = TextChunker(
    id_col="judgment_id",
    text_col="full_text",
    chunk_size=4096,      # Larger chunks for LLMs
    chunk_overlap=200,    # 5% overlap
    tokenizer=tokenizer
)
```

### Testing/Development

```python
# Quick testing with limited chunks
chunker = TextChunker(
    id_col="judgment_id",
    text_col="full_text",
    chunk_size=512,
    take_n_first_chunks=3  # Only first 3 chunks per document
)
```

## Output Schema

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `{id_col}` | str | Original document ID |
| `chunk_id` | int | Chunk index (0-based) |
| `chunk_len` | int | Length of chunk in characters |
| `chunk_text` | str | Chunk text content |

### Example Output

```python
{
    "judgment_id": ["doc1", "doc1", "doc1"],
    "chunk_id": [0, 1, 2],
    "chunk_len": [512, 512, 300],
    "chunk_text": [
        "First 512 characters...",
        "Next 512 characters...",
        "Final 300 characters..."
    ]
}
```

## Common Patterns

### Chunking Pipeline

```python
from datasets import load_dataset
from juddges.preprocessing.text_chunker import TextChunker
from transformers import AutoTokenizer

# Load model tokenizer
tokenizer = AutoTokenizer.from_pretrained("sdadas/mmlw-roberta-large")

# Initialize chunker
chunker = TextChunker(
    id_col="judgment_id",
    text_col="full_text",
    chunk_size=512,
    chunk_overlap=50,
    min_split_chars=100,
    tokenizer=tokenizer
)

# Load and chunk dataset
ds = load_dataset("juddges/pl-court-raw")
chunked_ds = ds.map(
    chunker,
    batched=True,
    batch_size=100,
    num_proc=8,
    remove_columns=ds.column_names
)

# Save chunked dataset
chunked_ds.save_to_disk("data/chunked")
```

### Chunk Statistics

```python
import numpy as np

def get_chunk_statistics(chunked_dataset):
    """Compute chunking statistics."""
    chunk_lengths = chunked_dataset["chunk_len"]

    return {
        "total_chunks": len(chunk_lengths),
        "mean_length": np.mean(chunk_lengths),
        "median_length": np.median(chunk_lengths),
        "min_length": np.min(chunk_lengths),
        "max_length": np.max(chunk_lengths),
        "std_length": np.std(chunk_lengths)
    }

stats = get_chunk_statistics(chunked_ds)
print(stats)
# {
#     "total_chunks": 150000,
#     "mean_length": 485.3,
#     "median_length": 512.0,
#     "min_length": 100,
#     "max_length": 512,
#     "std_length": 45.2
# }
```

## Related

- [Text Encoder](text_encoder.md) - Tokenization utilities
- [Context Truncator](context_truncator.md) - Context window management
- [Data Loaders](../data/loaders.md) - Dataset loading
- [How-To: Embeddings](../../../how-to/embeddings/embeddings_embed_and_ingest_weaviate.md) - Embedding workflow
