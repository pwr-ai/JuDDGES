# Weaviate Docker Setup

This directory contains Docker configuration for running Weaviate with transformer models for vector search in the JuDDGES project.

## Prerequisites

- Docker and Docker Compose installed
- Sufficient system resources (see deployment configuration in docker-compose.yaml)

## Environment Setup

1. Create a `.env` file in this directory with the following variables:

```
# Weaviate settings
QUERY_DEFAULTS_LIMIT=20
AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
PERSISTENCE_DATA_PATH=/var/lib/weaviate
DEFAULT_VECTORIZER_MODULE=text2vec-transformers
ENABLE_MODULES=text2vec-transformers
TRANSFORMERS_INFERENCE_API=http://t2v-transformers:8080
CLUSTER_HOSTNAME=node1

# Model settings
MODEL_ENV=sdadas/mmlw-roberta-large
```

2. Prepare the external volume:

```bash
docker volume create legal_ai_weaviate
```

## Building and Running

Build and start the containers:

```bash
docker-compose up -d
```

To stop the containers:

```bash
docker-compose down
```

## Accessing Weaviate

Once running, Weaviate will be accessible at:
- http://localhost:8084/v1

## Customizing the Transformer Model

The transformer model is specified in the Dockerfile. To use a different model:

1. Edit `hf_transformers.dockerfile` to change the model
2. Update the MODEL_ENV in your `.env` file
3. Rebuild the containers:

```bash
docker-compose build --no-cache
docker-compose up -d
```

## Troubleshooting

- If you encounter memory issues, adjust the resource limits in the docker-compose.yaml file
- Check logs with `docker-compose logs weaviate` or `docker-compose logs t2v-transformers`
