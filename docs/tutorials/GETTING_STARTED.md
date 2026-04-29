# Getting Started with JuDDGES

Welcome to JuDDGES! This guide will help you get up and running with the Judicial Decision Data Gathering, Encoding, and Sharing system in under 30 minutes.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed
- **Docker** and **Docker Compose** installed
- **Git** for version control
- **40GB+ disk space** for datasets and models
- **16GB+ RAM** recommended (8GB minimum)
- **(Optional) NVIDIA GPU** with 40GB+ VRAM for fine-tuning

## Quick Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/pwr-ai/JuDDGES.git
cd JuDDGES
```

### Step 2: Run Setup Script

```bash
# Linux/macOS
./setup.sh

# Windows
setup.bat
```

This script will:

- Create a virtual environment
- Install all dependencies
- Set up pre-commit hooks
- Download sample data

### Step 3: Set Environment Variables

Create a `.env` file in the project root:

```bash
# Required for Gemini extraction
GOOGLE_API_KEY=your-google-api-key

# Optional for observability
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# Optional for GPU
CUDA_VISIBLE_DEVICES=0
NUM_PROC=10
```

Get your Google API key from: https://ai.google.dev/gemini-api/docs/api-key

## Your First Extraction

### Extract Information from a Legal Document

```python
from juddges.extraction import GeminiExtractionChain
from juddges.extraction.gemini_chain import DocumentType, ExtractionSchema

# Initialize extraction chain
chain = GeminiExtractionChain(model_name="gemini-2.5-flash")

# Define what to extract
schema = ExtractionSchema(
    fields={
        "verdict_date": "date as ISO 8601",
        "court": "string, name of the court",
        "parties": "List[string], involved parties",
        "verdict": "string, the final verdict",
    },
    language="english",
)

# Sample judgment text
judgment_text = """
JUDGMENT of the Court of Appeal
Date: 15 January 2024
Court: Court of Appeal, London
Between: John Smith (Appellant) and Jane Doe (Respondent)
Verdict: Appeal dismissed with costs to the respondent.
"""

# Extract structured information
result = chain.extract(
    document_type=DocumentType.JUDGMENT,
    text=judgment_text,
    schema=schema,
)

print(result)
# Output:
# {
#   'verdict_date': '2024-01-15',
#   'court': 'Court of Appeal, London',
#   'parties': ['John Smith', 'Jane Doe'],
#   'verdict': 'Appeal dismissed with costs to the respondent'
# }
```

## Working with Polish Legal Documents

JuDDGES specializes in multilingual legal document processing:

```python
# Polish judgment example
polish_schema = ExtractionSchema(
    fields={
        "data_wyroku": "date as ISO 8601, data wydania wyroku",
        "sad": "string, nazwa sądu",
        "sygnatura": "string, sygnatura sprawy",
        "wyrok": "string, treść wyroku",
    },
    language="polish",
)

polish_text = """
Sąd Okręgowy w Warszawie
Sygnatura: II C 123/2023
Data: 20 grudnia 2023
Wyrok: Sąd oddala powództwo w całości.
"""

result = chain.extract(
    document_type=DocumentType.JUDGMENT,
    text=polish_text,
    schema=polish_schema,
)
```

## Semantic Search with Weaviate

### Start Weaviate Vector Database

```bash
cd weaviate
docker compose up -d
```

### Search Legal Documents

```python
from juddges.data.judgments_weaviate_db import JudgmentsWeaviateDB

# Connect to Weaviate
db = JudgmentsWeaviateDB(url="http://localhost:8080")

# Semantic search
results = db.search_semantic(
    query="Swiss franc loan conversion unfair terms",
    limit=5,
)

for result in results:
    print(f"Court: {result['court']}")
    print(f"Date: {result['judgment_date']}")
    print(f"Summary: {result['summary'][:200]}...")
    print("---")
```

## Access Pre-built Datasets

### Using HuggingFace Datasets

```python
from datasets import load_dataset

# Load Polish court decisions
polish_courts = load_dataset("JuDDGES/pl-court-raw-sample")

# Load Swiss franc loans dataset
swiss_loans = load_dataset("JuDDGES/swiss_franc_loans_instruct")

# Access the data
sample = polish_courts['train'][0]
print(f"Court: {sample['court']}")
print(f"Text: {sample['text'][:500]}...")
```

## Run the Interactive Dashboard

Launch the Streamlit dashboard for visual exploration:

```bash
# Start the dashboard
streamlit run juddges/dashboards/search_judgments.py

# Open browser at http://localhost:8501
```

Features:

- Semantic and keyword search
- Document filtering by court, date, type
- Information extraction interface
- Case law trend analysis

## Next Steps

### 1. Explore Tutorials

- [Gemini Extraction Tutorial](../tutorials/GEMINI_EXTRACTION.md)
- [Langfuse Setup](../tutorials/LANGFUSE_SETUP.md)
- [Git LFS Setup](../tutorials/GIT_LFS_SETUP.md)

### 2. Deep Dive into Features

- [Polish Court Data Acquisition](../how-to/data-acquisition/data_acquisition_pl_court.md)
- [NSA Data Acquisition](../how-to/data-acquisition/data_acquisition_nsa.md)
- [Embeddings and Weaviate](../how-to/embeddings/embeddings_embed_and_ingest_weaviate.md)

### 3. Understand the Architecture

- [Project Overview](../explanation/architecture/PROJECT_OVERVIEW.md)
- [System Architecture](../explanation/architecture/SYSTEM_ARCHITECTURE.md)
- [Data Flow Pipeline](../explanation/architecture/DATA_FLOW_PIPELINE.md)

### 4. Contribute

- [GitHub Repository](https://github.com/pwr-ai/JuDDGES) - View source code and contribute
- [Documentation Style Guide](../reference/STYLE_GUIDE.md) - Documentation standards
- [Report Issues](https://github.com/pwr-ai/JuDDGES/issues) - Report bugs or request features

## Common Issues and Solutions

### Issue: "API key not found"

**Solution**: Ensure your `.env` file contains:

```bash
GOOGLE_API_KEY=your-actual-api-key
```

### Issue: "Weaviate connection failed"

**Solution**: Check if Weaviate is running:

```bash
docker ps | grep weaviate
# If not running:
cd weaviate && docker compose up -d
```

### Issue: "Out of memory during extraction"

**Solution**: Use batch processing with smaller chunks:

```python
# Process in smaller batches
texts = [text1, text2, text3, ...]
batch_size = 10

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    results = chain.batch_extract(
        document_type=DocumentType.JUDGMENT,
        texts=batch,
        schema=schema,
    )
```

## Getting Help

- **Documentation**: You're here! Browse other guides in `/docs`
- **GitHub Issues**: Report bugs and request features on the [GitHub repository](https://github.com/pwr-ai/JuDDGES/issues)
- **Discussions**: Ask questions and share experiences
- **Email**: lukasz.augustyniak@pwr.edu.pl

## What Can You Build?

With JuDDGES, you can:

1. **Legal Research Tools**: Search and analyze court decisions
2. **Compliance Systems**: Extract regulatory information
3. **Case Analytics**: Track legal trends and patterns
4. **Document Automation**: Process legal documents at scale
5. **Knowledge Graphs**: Build legal citation networks
6. **Academic Research**: Empirical legal studies

## Ready to Go Deeper?

Congratulations! You've successfully:

- ✅ Installed JuDDGES
- ✅ Extracted information from legal documents
- ✅ Performed semantic search
- ✅ Accessed pre-built datasets
- ✅ Explored the dashboard

Continue your journey:

- 📚 Read the [Project Overview](../explanation/architecture/PROJECT_OVERVIEW.md) for the big picture
- 🛠️ Check [Technical Documentation](../README.md) for detailed guides
- 🏗️ See [System Architecture](../explanation/architecture/SYSTEM_ARCHITECTURE.md) for capabilities

---

**Welcome to the JuDDGES community!** 🎉

_Last updated: 2025-10-11 | Version: 1.0_
