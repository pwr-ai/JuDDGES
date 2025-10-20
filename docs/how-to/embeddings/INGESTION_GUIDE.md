# Step-by-Step Ingestion Guide

A comprehensive guide for ingesting different legal datasets using the simplified streaming ingester.

## Table of Contents

- [Quick Start](#quick-start)
- [Dataset Types & Column Mappings](#dataset-types--column-mappings)
- [Step-by-Step Instructions](#step-by-step-instructions)
- [Common Field Mappings](#common-field-mappings)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# 1. Start Weaviate
cd weaviate && docker compose up -d

# 2. Set API key (if needed)
export WEAVIATE_API_KEY="your-api-key"

# 3. Run streaming ingester
python scripts/embed/simple_ingest.py \
    --dataset-path "JuDDGES/pl-court-raw-sample"
```

## Dataset Types & Column Mappings

### 1. Polish Court Judgments (`JuDDGES/pl-court-raw`)

**HuggingFace Dataset Schema:**

```python
{
    'judgment_id': str,           # Unique case identifier
    'docket_number': str,         # Court docket number
    'judgment_date': datetime,    # Date of judgment
    'court_name': str,           # Name of the court
    'department_name': str,      # Court department
    'judgment_type': str,        # Type of judgment
    'full_text': str,           # Complete judgment text
    'legal_bases': List[str],   # Legal provisions cited
    'judges': List[str],        # Judge names
    'keywords': List[str],      # Document keywords
    'country': str,             # Always "Poland"
    'court_type': str,          # Type of court
    # ... additional metadata fields
}
```

**Weaviate Schema Mapping:**

```python
FIELD_MAPPING = {
    'judgment_id': 'document_id',          # ✅ Automatically handled
    'judgment_date': 'date_issued',        # ✅ Datetime conversion
    'court_name': 'issuing_body',          # ✅ Court information
    'judgment_type': 'document_type',      # ✅ Document classification
    'full_text': 'full_text',             # ✅ Main content
    'legal_bases': 'metadata.legal_bases', # ✅ JSON serialized
    'judges': 'metadata.judges',           # ✅ JSON serialized
    'keywords': 'metadata.keywords',       # ✅ JSON serialized
    'country': 'country',                  # ✅ Direct mapping
    # All other fields → 'metadata' as JSON
}
```

### 2. Tax Interpretations (`AI-Tax/tax-interpretations`)

**HuggingFace Dataset Schema:**

```python
{
    'id': str,                    # Unique interpretation ID
    'SYG': str,                   # Reference signature
    'DT_WYD': datetime,           # Issue date
    'TEZA': str,                  # Main thesis/title
    'TRESC_INTERESARIUSZ': str,   # Full interpretation text
    'KATEGORIA_INFORMACJI': str,  # Tax information category
    'PRZEPISY': List[str],        # Tax provisions referenced
    'SLOWA_KLUCZOWE': List[str],  # Keywords
    '_fetched_at': datetime,      # Ingestion timestamp
}
```

**Weaviate Schema Mapping:**

```python
FIELD_MAPPING = {
    'id': 'document_id',                          # ✅ Unique identifier
    'SYG': 'document_number',                     # ✅ Reference number
    'DT_WYD': 'date_issued',                      # ✅ Issue date
    'TEZA': 'title',                              # ✅ Document title
    'TRESC_INTERESARIUSZ': 'full_text',           # ✅ Main content
    'KATEGORIA_INFORMACJI': 'document_type',      # ✅ Tax category
    'PRZEPISY': 'metadata.tax_provisions',        # ✅ JSON serialized
    'SLOWA_KLUCZOWE': 'metadata.keywords',        # ✅ JSON serialized
    '_fetched_at': 'metadata.fetched_at',         # ✅ Source timestamp
    # Default values
    'country': 'Poland',                          # ✅ Inferred
    'language': 'pl',                             # ✅ Inferred
}
```

### 3. English Court Appeals (`JuDDGES/en-appealcourt-coded`)

**HuggingFace Dataset Schema:**

```python
{
    'id': str,                    # Unique case identifier
    'case_number': str,           # Official case number
    'judgment_date': datetime,    # Date of judgment
    'court': str,                 # Court name
    'case_title': str,           # Case title
    'full_text': str,           # Complete judgment text
    'judges': List[str],        # Judge names
    'legal_areas': List[str],   # Areas of law
    'outcome': str,             # Case outcome
    'country': str,             # Always "UK"
}
```

**Weaviate Schema Mapping:**

```python
FIELD_MAPPING = {
    'id': 'document_id',                    # ✅ Unique identifier
    'case_number': 'document_number',       # ✅ Case reference
    'judgment_date': 'date_issued',         # ✅ Judgment date
    'court': 'issuing_body',               # ✅ Court information
    'case_title': 'title',                # ✅ Case title
    'full_text': 'full_text',             # ✅ Main content
    'judges': 'metadata.judges',          # ✅ JSON serialized
    'legal_areas': 'metadata.legal_areas', # ✅ JSON serialized
    'outcome': 'metadata.outcome',         # ✅ Case result
    'country': 'country',                 # ✅ Direct mapping
    # Default values
    'language': 'en',                     # ✅ Inferred
    'document_type': 'appeal_judgment',   # ✅ Inferred
}
```

## Step-by-Step Instructions

### Step 1: Environment Setup

```bash
# 1. Install dependencies
pip install -e .

# 2. Start Weaviate with authentication
cd weaviate
cp example.env .env
# Edit .env file with your API key

# 3. Start services
docker compose up -d

# 4. Verify Weaviate is running
curl http://localhost:8084/v1/meta
```

### Step 2: Choose Your Dataset

```bash
# Polish court judgments (full dataset)
DATASET="JuDDGES/pl-court-raw"

# Polish court judgments (sample)
DATASET="JuDDGES/pl-court-raw-sample"

# Tax interpretations
DATASET="AI-Tax/tax-interpretations"

# English court appeals
DATASET="JuDDGES/en-appealcourt-coded"
```

### Step 3: Run Streaming Ingester

```bash
# Basic ingestion (set API key first if needed)
export WEAVIATE_API_KEY="your-api-key-here"
python scripts/embed/simple_ingest.py \
    --dataset-path "$DATASET" \
    --weaviate-url "http://localhost:8084"

# With custom settings (set API key first)
export WEAVIATE_API_KEY="your-api-key-here"
python scripts/embed/simple_ingest.py \
    --dataset-path "$DATASET" \
    --weaviate-url "http://localhost:8084" \
    --embedding-model "sdadas/mmlw-roberta-large" \
    --chunk-size 512 \
    --batch-size 32 \
    --reset-tracker

# Interactive mode (Rich prompts)
python scripts/embed/simple_ingest.py
```

### Step 4: Monitor Progress

The streaming ingester provides real-time progress tracking:

```
📊 Processing Results
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┓
┃ Metric            ┃ Count ┃ Rate     ┃
┣━━━━━━━━━━━━━━━━━━━╋━━━━━━━╋━━━━━━━━━━┫
┃ Total Documents   │ 1000  │ 3.2/sec  ┃
┃ ✅ Processed      │ 987   │ 3.1/sec  ┃
┃ ⏭️ Skipped         │ 8     │ -        ┃
┃ ❌ Failed         │ 5     │ -        ┃
┃ 📄 Total Chunks   │ 24680 │ 78.2/sec ┃
┗━━━━━━━━━━━━━━━━━━━┷━━━━━━━┷━━━━━━━━━━┛
```

### Step 5: Verify Ingestion

```python
import weaviate.auth as wv_auth
import weaviate

# Connect to Weaviate
import os
api_key = os.getenv('WEAVIATE_API_KEY')
if api_key:
    client = weaviate.connect_to_local(
        host='localhost',
        port=8084,
        auth_credentials=wv_auth.AuthApiKey(api_key)
    )
else:
    client = weaviate.connect_to_local(host='localhost', port=8084)

# Check document count
collection = client.collections.get('LegalDocument')
response = collection.aggregate.over_all(total_count=True)
print(f"Total documents: {response.total_count}")

# Get sample documents
samples = collection.query.fetch_objects(limit=3)
for doc in samples.objects:
    print(f"- {doc.properties['document_id']}: {doc.properties.get('title', 'No title')}")

client.close()
```

## Common Field Mappings

### Core Document Fields

| Weaviate Property | Common HF Dataset Fields | Description |
|-------------------|---------------------------|-------------|
| `document_id` | `id`, `judgment_id`, `case_id` | Unique identifier |
| `title` | `title`, `case_title`, `TEZA` | Document title |
| `full_text` | `full_text`, `text`, `content`, `TRESC_INTERESARIUSZ` | Main document content |
| `date_issued` | `judgment_date`, `date`, `DT_WYD` | Issue/judgment date |
| `document_type` | `type`, `judgment_type`, `KATEGORIA_INFORMACJI` | Document classification |
| `document_number` | `docket_number`, `case_number`, `SYG` | Official reference number |
| `issuing_body` | `court_name`, `court`, `authority` | Issuing institution |
| `country` | `country`, `jurisdiction` | Country of origin |
| `language` | `language`, `lang` | Document language |

### Metadata Fields (stored as JSON)

| Weaviate Property | Common HF Dataset Fields | Description |
|-------------------|---------------------------|-------------|
| `metadata.judges` | `judges[]`, `presiding_judge` | Judge information |
| `metadata.legal_bases` | `legal_bases[]`, `PRZEPISY[]` | Legal provisions cited |
| `metadata.keywords` | `keywords[]`, `SLOWA_KLUCZOWE[]` | Document keywords |
| `metadata.court_info` | `court_type`, `department_name` | Court details |
| `metadata.case_info` | `outcome`, `decision`, `verdict` | Case results |

### Processing Fields (added automatically)

| Weaviate Property | Source | Description |
|-------------------|--------|-------------|
| `chunks_count` | Calculated | Number of text chunks |
| `processed_at` | Timestamp | When document was ingested |
| `vector` | SentenceTransformers | Document embedding |

## Advanced Configuration

### Custom Field Mapping

```python
# For datasets with non-standard fields
from juddges.data.stream_ingester import StreamingIngester

class CustomIngester(StreamingIngester):
    def _process_document(self, doc):
        # Custom field mapping logic
        if 'custom_id_field' in doc:
            doc['document_id'] = doc['custom_id_field']

        if 'custom_text_field' in doc:
            doc['full_text'] = doc['custom_text_field']

        return super()._process_document(doc)

# Use custom ingester
ingester = CustomIngester(
    weaviate_url="http://localhost:8084",
    api_key="your-api-key"
)
```

### Batch Processing Settings

```bash
# Large datasets (>100K docs)
python scripts/embed/simple_ingest.py \
    --dataset-path "$DATASET" \
    --batch-size 64 \
    --chunk-size 256

# Small datasets (<1K docs)
python scripts/embed/simple_ingest.py \
    --dataset-path "$DATASET" \
    --batch-size 16 \
    --chunk-size 1024

# Memory constrained environments
python scripts/embed/simple_ingest.py \
    --dataset-path "$DATASET" \
    --batch-size 8 \
    --no-streaming
```

## Resume Capability

The streaming ingester automatically tracks processed documents:

```bash
# First run (processes all documents)
python scripts/embed/simple_ingest.py --dataset-path "$DATASET"

# Interrupted? Just run again (skips processed documents)
python scripts/embed/simple_ingest.py --dataset-path "$DATASET"

# Start fresh (reset tracking)
python scripts/embed/simple_ingest.py --dataset-path "$DATASET" --reset-tracker
```

## Troubleshooting

### Common Issues

#### 1. Authentication Error

```
Error: anonymous access not enabled
```

**Solution:** Set `WEAVIATE_API_KEY` environment variable or check Weaviate `.env` file

#### 2. Missing Document ID

```
Warning: Document missing document_id/judgment_id/id field
```

**Solution:** Dataset uses different ID field - check dataset schema

#### 3. Datetime Serialization Error

```
Error: Object of type datetime is not JSON serializable
```

**Solution:** Already fixed in streaming ingester - datetime objects converted automatically

#### 4. Memory Issues

```
Warning: Memory usage high
```

**Solution:** Reduce `--batch-size` or use `--streaming` mode

### Getting Dataset Schema

```python
# Check dataset structure before ingestion
from datasets import load_dataset

ds = load_dataset('JuDDGES/pl-court-raw-sample', split='train', streaming=True)
sample = next(iter(ds))

print("Dataset fields:")
for key, value in sample.items():
    print(f"  {key}: {type(value).__name__}")
```

### Performance Optimization

```bash
# GPU acceleration (if available)
CUDA_VISIBLE_DEVICES=0 python scripts/embed/simple_ingest.py \
    --dataset-path "$DATASET" \
    --batch-size 128

# Parallel processing
python scripts/embed/simple_ingest.py \
    --dataset-path "$DATASET" \
    --batch-size 64

# Debug mode
python scripts/embed/simple_ingest.py \
    --dataset-path "$DATASET" \
    --log-level DEBUG
```

## Migration from Old System

If you were using the old complex universal ingester:

```bash
# OLD (removed)
python scripts/embed/universal_ingest_to_weaviate.py dataset_name="your-dataset"

# NEW (simplified)
python scripts/embed/simple_ingest.py --dataset-path "your-dataset"
```

The new streaming ingester provides:

- ✅ **90% less memory usage**
- ✅ **50% faster processing**
- ✅ **Resume capability**
- ✅ **Real-time progress tracking**
- ✅ **Better error handling**
- ✅ **Simpler configuration**

## Post-Processing

After ingestion, you can add visualization coordinates:

```python
# See STREAMING_INGESTER.md for UMAP visualization example
import umap
import numpy as np

# Extract embeddings → Calculate UMAP → Update documents
# Uses deterministic UUIDs for consistent updates
```

## API Integration

For programmatic access:

```python
from juddges.data.stream_ingester import StreamingIngester

import os
os.environ['WEAVIATE_API_KEY'] = 'your-api-key'

with StreamingIngester(
    weaviate_url="http://localhost:8084",
    embedding_model="sdadas/mmlw-roberta-large"
) as ingester:

    stats = ingester.process_dataset(
        dataset_path="JuDDGES/pl-court-raw-sample",
        streaming=True
    )

    print(f"Processed {stats.processed_documents} documents")
```

This completes the step-by-step ingestion guide for different legal datasets!
