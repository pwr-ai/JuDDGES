# JuDDGES Embeddings Preparation and Ingestion Guide

This document describes the process of generating text embeddings from raw documents and ingesting them into a Weaviate database for the JuDDGES system.

## Overview

The embedding workflow consists of three main steps:

1. **Embedding Generation**: Convert raw text into vector embeddings
2. **Embedding Ingestion**: Upload embeddings to a Weaviate database

## Prerequisites

1. Deploy Weaviate instance (see [docs/embeddings_deploy_weaviate.md](/docs/embeddings_deploy_weaviate.md))
2. Setup environment variables in `.env` file
  ```
  WV_HOST=localhost       # Weaviate host
  WV_PORT=8080            # Weaviate port
  WV_GRPC_PORT=50051      # Weaviate gRPC port
  WV_API_KEY=<your-key>   # Weaviate API key (if applicable)
  NUM_PROC=<n>            # Number of processors for parallel operations
  ```

## Step 1: Embedding Generation

The embedding generation script (`scripts/embed/embed_text.py`) converts raw text documents into vector embeddings using a Sentence Transformer model using dataset from huggingface hub.

### Running Embedding Generation
Full configuration of embedding generation is defined in file [`configs/embedding.yaml`](/configs/embedding.yaml). To run the embedding simply run command with proper dataset and embedding model with following command. This will output the chunk embeddings data/embeddings/<dataset_name>/<embedding_model>/all_embeddings directory. It overrides hydra config, so for embedding model use names of configs present in [`configs/embedding_model`](/configs/embedding_model), and for dataset simply use name from huggingface hub.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/embed/embed_text.py embedding_model=mmlw-roberta-large dataset_name="JuDDGES/pl-court-raw"
```

## Step 2: Embedding Ingestion

To upload the embeddings to a Weaviate database:

```bash
python scripts/embed/ingest_embeddings_to_weaviate.py --embeddings-dir data/embeddings/JuDDGES/pl-court-raw/mmlw-roberta-large/all_embeddings [--batch-size 64] [--upsert]
```

Parameters:
- `--embeddings-dir`: Path to the directory containing the embeddings
- `--batch-size`: Number of embeddings to upload in a batch (default: 64)
- `--upsert`: If set, updates existing embeddings in the database; otherwise, only uploads new ones

This will:
1. Load embeddings from the specified directory (use the one from Step 1)
2. Generate UUIDs for each embedding
3. Upload the embeddings to the Weaviate database


### Step 3: Dataset Ingestion

The final step involves ingesting the raw documents from the Hugging Face dataset into Weaviate.

```bash
python scripts/embed/ingest_documents_to_weaviate.py --dataset-name "JuDDGES/pl-court-raw" [--batch-size 64]
```

Parameters:
- `--dataset-name`: Name of the dataset on Hugging Face Hub to ingest (required)
- `--batch-size`: Number of documents to upload in a batch (default: 64)

## Notes

- The embedding model used locally should match the one configured in the Weaviate database
- For large datasets, consider adjusting the batch size and number of processors
- The chunking process can be customized through the configuration to suit your specific document characteristics
