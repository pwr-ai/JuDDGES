# Deploy the Weaviate v2 stack (Weaviate 1.40 + BGE-M3)

The stack in `weaviate/v2/` runs a fresh Weaviate instance for court judgments
alongside the legacy one in `weaviate/docker-compose.yaml`. It uses a different
compose project name, different ports and different volumes, so both can run at once.

| Service | Image | Host port (default) |
|---|---|---|
| `weaviate` | `cr.weaviate.io/semitechnologies/weaviate:1.40.2` | `8110` (HTTP), `50061` (gRPC) |
| `embeddings` | `ghcr.io/huggingface/text-embeddings-inference:86-1.9.4` serving `BAAI/bge-m3` | `8111` |

Ports bind to `127.0.0.1` by default. Set `WEAVIATE_BIND=0.0.0.0` only behind a
firewall or reverse proxy.

## Start

```bash
cd weaviate/v2
cp .env.example .env        # set AUTHENTICATION_APIKEY_ALLOWED_KEYS / _USERS
docker compose up -d
```

On first start, TEI downloads BGE-M3 (~2.2 GB) into the `embedding_v2_model_cache`
volume. Weaviate waits until TEI reports healthy.

## Verify

```bash
source .env
curl -fs localhost:8110/v1/.well-known/ready && echo ready
curl -s -H "Authorization: Bearer $AUTHENTICATION_APIKEY_ALLOWED_KEYS" localhost:8110/v1/meta   # "version": "1.40.2"
curl -s localhost:8111/embed -H 'Content-Type: application/json' \
  -d '{"inputs":["Sąd Najwyższy oddalił kasację."]}'                                        # 1024 floats
```

## Configuration notes

- **GPU image tag.** The `86-` tag targets Ampere `sm_86` (RTX 3090, A5000). Use
  `hopper-1.9.4` for H100, `89-1.9.4` for Ada, or `1.9.4` for A100.
- **GPU selection.** `EMBEDDINGS_GPU` picks the device id.
- **Quantization.** `DEFAULT_QUANTIZATION=rq-8` gives every new collection 8-bit
  rotational quantization when its schema does not configure one. Collection schemas
  should still set the quantizer explicitly. Quantization cannot be disabled on a
  collection once it is enabled. The rationale is in
  [Embedding model and vector quantization](../../explanation/embedding-and-quantization.md).
- **Disk.** Weaviate keeps the uncompressed vectors on disk for rescoring.
  Quantization reduces RAM, not disk.

## Stop / remove

```bash
docker compose down        # keeps data volumes
docker compose down -v     # also deletes weaviate_v2_data and the model cache
```

For backups, see [Weaviate backup and restore](weaviate-backup-restore.md). The
backup directory is `WEAVIATE_BACKUP_DIR`, which defaults to `weaviate/v2/backups`.
