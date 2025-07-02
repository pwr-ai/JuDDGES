# Weaviate Docker Setup

This directory contains the Docker configuration for running Weaviate vector database with custom transformer models for the JuDDGES legal AI project.

## Quick Start

1. **Setup environment variables**:
   ```bash
   cp example.env .env
   # Edit .env with your configuration
   ```

2. **Create Docker volumes and start services**:
   ```bash
   ./setup_docker.sh
   ```

3. **Or manually**:
   ```bash
   docker volume create legal_ai_weaviate_prod
   docker compose up -d
   ```

## Configuration

### Environment Variables (.env)

Copy `example.env` to `.env` and configure:

- `AUTHENTICATION_APIKEY_ALLOWED_KEYS`: API key for authentication
- `AUTHENTICATION_APIKEY_USERS`: Username for API access
- `MODEL_NAME`: Transformer model name (e.g., `sdadas/mmlw-roberta-large`)
- `ENABLE_CUDA`: Set to `1` to enable GPU acceleration

### Services

- **weaviate**: Main vector database (port 8084)
- **t2v-transformers**: Custom transformer service for embeddings

### Volumes

- `legal_ai_weaviate_prod`: External volume for persistent data storage

## Usage

### Start Services
```bash
docker compose up -d
```

### Stop Services
```bash
docker compose down
```

### View Logs
```bash
docker compose logs -f weaviate
docker compose logs -f t2v-transformers
```

### Access Weaviate
- **API**: http://localhost:8084
- **Console**: http://localhost:8084/v1/meta

## Resource Limits

- **Weaviate**: 25 CPUs, 60GB RAM
- **Transformers**: 8 CPUs, 16GB RAM

## Troubleshooting

1. **Volume issues**: Run `./setup_docker.sh` to create required volumes
2. **GPU not detected**: Ensure `ENABLE_CUDA=1` in `.env` and Docker supports GPU
3. **Authentication errors**: Check API key and user configuration in `.env`
4. **Memory issues**: Adjust resource limits in `docker-compose.yaml` if needed
