# Weaviate Backup and Restore

How to create and restore native Weaviate backups that preserve data, vectors, and HNSW indices.

## Prerequisites

The backup-filesystem module must be enabled in `weaviate/.env`:

```
ENABLE_MODULES='text2vec-transformers,backup-filesystem'
BACKUP_FILESYSTEM_PATH='/var/lib/weaviate-backups'
```

The NAS mount must be present in `weaviate/docker-compose.yaml`:

```yaml
volumes:
  - /mnt/readynas/datasets/legal-ai-weaviate/native-backups:/var/lib/weaviate-backups
```

After changing these, restart Weaviate:

```bash
cd weaviate/
docker compose up -d weaviate
```

## Create a Backup

```bash
cd weaviate/
python backup_native.py
```

This creates a timestamped backup (e.g., `backup-20260324-120000`) on ReadyNAS. The backup runs online — Weaviate continues serving queries during the process.

Custom backup name:

```bash
python backup_native.py --backup-id before-migration-v2
```

Expected time: 30-60 minutes for the current dataset (3.2M LegalDocuments + 37.8M DocumentChunks).

## List Existing Backups

```bash
python backup_native.py --list
```

Backups are stored at `/mnt/readynas/datasets/legal-ai-weaviate/native-backups/`.

## Restore from Backup

### Standard Restore (collections don't exist yet)

```bash
python backup_native.py --restore backup-20260324-120000
```

This restores all collections from the backup. You will be prompted to confirm.

### Restore After Data Loss (collections still exist but are corrupted/empty)

You must delete the existing collections first, then restore:

```bash
python -c "
import weaviate, os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path('weaviate/.env'))
client = weaviate.connect_to_custom(
    http_host='localhost', http_port=8084, http_secure=False,
    grpc_host='localhost', grpc_port=8085, grpc_secure=False,
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv('WEAVIATE_API_KEY', '')),
)
client.collections.delete('LegalDocuments')
client.collections.delete('DocumentChunks')
client.close()
print('Collections deleted. Now run restore.')
"

python weaviate/backup_native.py --restore backup-20260324-120000
```

### Full Disaster Recovery (fresh Weaviate instance)

If the Docker volume is lost and you need to start from scratch:

1. Ensure `/mnt/readynas` is mounted and backups are accessible
2. Recreate the Docker volume:

   ```bash
   docker volume create legal_ai_weaviate_prod
   ```

3. Start Weaviate:

   ```bash
   cd weaviate/
   docker compose up -d weaviate t2v-transformers-base
   ```

4. Wait for Weaviate to be ready:

   ```bash
   curl -s http://localhost:8084/v1/.well-known/ready
   ```

5. Restore:

   ```bash
   python backup_native.py --restore backup-20260324-120000
   ```

## Backup Strategy

| Method | What's preserved | Recovery time | Use case |
|---|---|---|---|
| **Native backup** (this guide) | Data + vectors + HNSW indices | 30-60 min | Primary disaster recovery |
| **Parquet dump** (`dump_collections.py`) | Raw data only | 8-12+ hours (re-vectorize) | Data portability, analytics |
| **Docker volume snapshot** | Everything (byte-level) | Minutes | Fast rollback before upgrades |

Recommended schedule:

- **Native backup**: Weekly or before any Weaviate version upgrade
- **Parquet dump**: Monthly or before schema changes (version-independent format)

## Troubleshooting

**"backup-filesystem module not enabled"**: Restart Weaviate after adding the module to `.env`:

```bash
docker compose restart weaviate
```

**"backup already exists"**: Each backup ID must be unique. Use `--list` to see existing backups, then choose a different name or delete the old backup directory from ReadyNAS.

**Restore fails with "collection already exists"**: Delete the existing collections first (see "Restore After Data Loss" above).

**NAS not mounted**: Verify the mount is available:

```bash
ls /mnt/readynas/datasets/legal-ai-weaviate/native-backups/
```
