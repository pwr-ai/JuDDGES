# Demo Scripts

This directory contains demonstration scripts that showcase the capabilities of the JuDDGES Universal Ingestion System.

## Scripts Overview

### 📄 `show_available_datasets.py`

A rich-formatted display of all available datasets for Weaviate ingestion.

**Features:**

- 🎨 Rich console output with colors and emojis
- 📊 Organized tables showing legal datasets, HuggingFace datasets, and file formats
- 🚀 Getting started guide with command examples
- 🎯 Recommendations for different use cases

**Usage:**

```bash
python scripts/demos/show_available_datasets.py
```

### 🏛️ `demo_local_dataset_ingestion.py`

Demonstrates ingestion of local Polish court data into Weaviate.

**Features:**

- 🇵🇱 Sample Polish court judgments from different court levels
- 🤖 Automatic field mapping demonstration
- 📊 Court hierarchy representation (District → Regional → Supreme)
- 🔍 Polish legal terminology search examples
- 📂 Local file detection and statistics

**Prerequisites:**

- Weaviate running on `http://localhost:8084`
- Local Polish court data in `data/datasets/pl/raw/`

**Usage:**

```bash
# Start Weaviate first
docker run -d --name weaviate-test -p 8084:8080 cr.weaviate.io/semitechnologies/weaviate:1.26.1

# Run the demo
python scripts/demos/demo_local_dataset_ingestion.py
```

### 🗄️ `demo_weaviate_ingestion.py`

General demonstration of Weaviate ingestion with legal documents.

**Features:**

- 📋 Schema creation for legal documents and chunks
- 📥 Document and chunk ingestion workflow
- 🔍 Keyword search demonstrations
- 📊 Collection statistics
- 🏛️ Multi-court document support

**Prerequisites:**

- Weaviate running on `http://localhost:8084`

**Usage:**

```bash
# Start Weaviate first
docker run -d --name weaviate-test -p 8084:8080 cr.weaviate.io/semitechnologies/weaviate:1.26.1

# Run the demo
python scripts/demos/demo_weaviate_ingestion.py
```

## Quick Start

1. **View Available Datasets:**

   ```bash
   python scripts/demos/show_available_datasets.py
   ```

2. **Test Weaviate Connection:**

   ```bash
   # Start Weaviate
   cd weaviate/
   docker-compose up -d

   # Or use the simple docker command
   docker run -d --name weaviate-test -p 8084:8080 cr.weaviate.io/semitechnologies/weaviate:1.26.1
   ```

3. **Run Demo Scripts:**

   ```bash
   # Try the general Weaviate demo first
   python scripts/demos/demo_weaviate_ingestion.py

   # Then try the local dataset demo
   python scripts/demos/demo_local_dataset_ingestion.py
   ```

## Integration with Main System

These demos show the same patterns used by the production ingestion scripts:

- `scripts/embed/universal_ingest_to_weaviate.py` - Production ingestion
- `scripts/dataset_manager.py` - Dataset management CLI
- `scripts/embed/ingest_to_weaviate.py` - Standard ingestion

The demos use simplified data and minimal dependencies to demonstrate core concepts without requiring the full JuDDGES environment.
