# Deploy the Weaviate v2 stack (Weaviate 1.40 + mmlw-retrieval-e5-large)

The stack in `weaviate/v2/` runs a fresh Weaviate instance for court judgments
alongside the legacy one in `weaviate/docker-compose.yaml`. It uses a different
compose project name, different ports and different volumes, so both can run at once.

| Service | Image | Host port (default) |
|---|---|---|
| `weaviate` | `cr.weaviate.io/semitechnologies/weaviate:1.40.2` | `8110` (HTTP), `50061` (gRPC) |
| `embeddings` | `ghcr.io/huggingface/text-embeddings-inference:86-1.9.4` serving `sdadas/mmlw-retrieval-e5-large` | `8111` |

Ports bind to `127.0.0.1` by default. Set `WEAVIATE_BIND=0.0.0.0` only behind a
firewall or reverse proxy.

## Start

```bash
cd weaviate/v2
cp .env.example .env        # set AUTHENTICATION_APIKEY_ALLOWED_KEYS / _USERS
docker compose up -d
```

On first start, TEI downloads mmlw-retrieval-e5-large (~2.2 GB) into the `embedding_v2_model_cache`
volume. Weaviate waits until TEI reports healthy.

## Verify

```bash
source .env
curl -fs localhost:8110/v1/.well-known/ready && echo ready
curl -s -H "Authorization: Bearer $AUTHENTICATION_APIKEY_ALLOWED_KEYS" localhost:8110/v1/meta   # "version": "1.40.2"
curl -s localhost:8111/embed -H 'Content-Type: application/json' \
  -d '{"inputs":["query: Sąd Najwyższy oddalił kasację."]}'                                 # 1024 floats
```

## Create the collections

The schema (`Judgments` and `JudgmentChunks`, RQ 4-bit, Polish stopwords) is defined in
`juddges/data/weaviate_v2/`. Property tables and the IE naming rules are in the
[Weaviate v2 schema reference](../../reference/weaviate-v2-schema.md).

```python
from juddges.data.weaviate_v2.ie_properties import add_ie_schema, load_ie_schema
from juddges.data.weaviate_v2.schema import (
    JUDGMENT_CHUNKS_COLLECTION,
    JUDGMENTS_COLLECTION,
    SchemaConfig,
    connect_local,
    create_collections,
)
from juddges.settings import CONFIG_PATH

client = connect_local(api_key="<key from .env>")  # ports 8110 / 50061
ie = load_ie_schema(CONFIG_PATH / "ie_schema" / "swiss_franc_loans.yaml")
create_collections(client, SchemaConfig(chunk_shards=8), ie_schemas=[ie])  # skips existing collections
```

Set the shard counts before the first ingest: they cannot be changed afterwards.

To add another IE schema to existing collections (idempotent, existing properties are skipped). Do it
before ingesting chunks if the flat enum copies should be on chunks:

```python
add_ie_schema(
    client.collections.get(JUDGMENTS_COLLECTION),
    CONFIG_PATH / "ie_schema" / "personal_rights.yaml",
    chunk_collection=client.collections.get(JUDGMENT_CHUNKS_COLLECTION),
)
```

Integration tests against a throwaway Weaviate:

```bash
docker run -d --rm --name wv-schema-test -p 127.0.0.1:8119:8080 -p 127.0.0.1:50069:50051 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e DEFAULT_VECTORIZER_MODULE=none \
  -e PERSISTENCE_DATA_PATH=/tmp/w -e CLUSTER_HOSTNAME=n1 -e RAFT_BOOTSTRAP_EXPECT=1 \
  cr.weaviate.io/semitechnologies/weaviate:1.40.2
WEAVIATE_TEST_URL=http://127.0.0.1:8119 pytest tests/weaviate_v2 -m requires_weaviate
docker stop wv-schema-test
```

## Configuration notes

- **GPU image tag.** The `86-` tag targets Ampere `sm_86` (RTX 3090, A5000). Use
  `hopper-1.9.4` for H100, `89-1.9.4` for Ada, or `1.9.4` for A100.
- **GPU selection.** `EMBEDDINGS_GPU` picks the device id.
- **Prompts.** TEI does not add prompts for this model. Clients must prefix queries
  with `query:` and chunks with `passage:`, each followed by a space. Without the prefix, cosine to the
  reference vectors drops to ~0.82. The model's context is 512 tokens, so chunk to
  480 tokens with 64 overlap.
- **Quantization.** `DEFAULT_QUANTIZATION=rq-4` gives every new collection 4-bit
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
