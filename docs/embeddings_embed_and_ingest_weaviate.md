# Embed `pl-court-raw` dataset and ingest to Weaviate

This document describes the process of generating text embeddings from raw documents in [`JuDDGES/pl-court-raw`](https://huggingface.co/datasets/JuDDGES/pl-court-raw) dataset and ingesting them into a Weaviate database.

## Overview

The embedding workflow consists of three main steps:

1. **Embedding Generation**: Chunk raw text, convert chunks into vector embeddings, aggregate embeddings for each judgment
2. **Embedding Ingestion**: Upload embeddings to a Weaviate database

## Prerequisites

1. Deploy Weaviate instance (see [docs/embeddings_deploy_weaviate.md](/docs/embeddings_deploy_weaviate.md))
2. Setup environment variables in `.env` file

  ```env
  WV_HOST=localhost       # Weaviate host
  WV_PORT=8080            # Weaviate port
  WV_GRPC_PORT=50051      # Weaviate gRPC port
  WV_API_KEY=<your-key>   # Weaviate API key (if applicable)
  ```

## Step 1: Embed documents

The embedding generation script (`scripts/embed/embed_text.py`) converts raw text documents into vector embeddings using a Sentence Transformer model using dataset from huggingface hub.

### Running Embedding Generation

* Full configuration of embedding generation is defined in file [`configs/embedding.yaml`](/configs/embedding.yaml).
* To run the embedding simply run command with proper dataset and embedding model with following command.
  * It overrides hydra config, so for embedding model use names of configs present in [`configs/embedding_model`](/configs/embedding_model), and for dataset simply use name from huggingface hub.
* The output will be two dirs with chunk and aggregated embeddings:
  * `data/embeddings/<dataset_name>/<embedding_model>/chunk_embeddings`
  * `data/embeddings/<dataset_name>/<embedding_model>/agg_embeddings`
* The script can work with multiple GPUs at once (by default it uses all available GPUs, so specify them with `CUDA_VISIBLE_DEVICES`).

```bash
CUDA_VISIBLE_DEVICES=0 NUM_PROC=10 PYTHONPATH="$PWD:$PYTHONPATH" python scripts/embed/embed_text.py \
  embedding_model=mmlw-roberta-large \
  dataset_name=JuDDGES/pl-court-raw \
  output_dir=data/embeddings/pl-court-raw/mmlw-roberta-large
```

## Step 2: Ingest embeddings to Weaviate

* To upload the embeddings created in the previous step to a Weaviate database, one needs to run the following ommand with parameters similar to the previous one.
* The upload will be done in two steps:
  * Upload chunks with their embeddings
  * Upload judgments with their aggregated embeddings (full dataset with aggregated embeddings will be ingested)

```bash
PROCESSING_PROC=10 INGEST_PROC=5 PYTHONPATH="$PWD:$PYTHONPATH" python scripts/embed/ingest_to_weaviate.py \
  embedding_model=mmlw-roberta-large \
  dataset_name=JuDDGES/pl-court-raw \
  output_dir=data/embeddings/pl-court-raw/mmlw-roberta-large \
  [+ingest_batch_size=64] \
  [+upsert=true]
```

## Step 3: Test Weaviate Ingestion

Test the ingestion by running the following command, which will print all collections and their schemas, and run sample queries.

```bash
python scripts/embed/test_weaviate_ingestion.py
```

## Notes

* The embedding model used locally should match the one configured in the Weaviate database
* For large datasets, consider adjusting the batch size and number of processors
* The chunking process can be customized through the configuration to suit your specific document characteristics
* The code were adjusted to be memory efficient (uses hf datasets and polars lazyframe)
