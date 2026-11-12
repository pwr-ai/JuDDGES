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
create_collections(client, SchemaConfig(chunk_shards=8), ie_schemas=[ie])  # existing collections: only missing properties are added
```

Set the shard counts before the first ingest: they cannot be changed afterwards.
On collections that already exist, `create_collections` leaves vectors, sharding and the inverted-index config
alone and only adds missing properties (including those of the given IE schemas). A property that exists with
another type raises. One whose index flags, tokenization or nested keys differ only logs a warning, because
Weaviate cannot change those in place: to change them, create a new collection and re-ingest.

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

## Ingest a sample and search it

`scripts/weaviate_v2/ingest_sample.py` loads judgments from `JuDDGES/pl-court-raw`, chunks them
exactly like the embedding evaluation (480 tokens, 64 overlap, BGE-M3 tokenizer), embeds chunks
and theses through the TEI container and imports them into `Judgments` and `JudgmentChunks`.
It refuses to run if TEI serves any model other than `sdadas/mmlw-retrieval-e5-large`.
Object ids are deterministic, so a re-run overwrites instead of duplicating.

```bash
export WEAVIATE_V2_API_KEY=<key from .env>
# 1000 random judgments that have a thesis, plus optional extra ids (one judgment_id per line)
python scripts/weaviate_v2/ingest_sample.py --n 1000 --extra-ids ids.txt --stats-out stats.json

# hybrid search over chunks, aggregated to judgments by best chunk (MaxP)
python scripts/weaviate_v2/search.py query "nieważność umowy kredytu indeksowanego do CHF"

# thesis -> judgment retrieval over what is ingested, for several hybrid alphas
python scripts/weaviate_v2/search.py eval --n 300 --alphas 0,0.25,0.5,1
```

The library code is `juddges/data/weaviate_v2/ingest.py` (mapping, chunking, TEI client) and
`juddges/data/weaviate_v2/search.py` (`search_judgments`). The `eval` command searches
`JudgmentChunks` only, because the `thesis` vector on `Judgments` would return the answer.

On the first sample (1050 judgments, 18 991 chunks, 2026-11-12), with the A5000 shared with
another TEI:

| Stage | Throughput |
|---|---|
| Chunking (one process) | ~14 judgments/s |
| Embedding (TEI, batch 64) | ~217 passages/s |
| Import into Weaviate | ~2 470 chunks/s |

That sample was biased towards judgments with a thesis (~18 chunks per judgment). The full
pl-court-raw (~437k judgments) averages ~15 chunks per judgment, ~6.5M chunks, and takes ~8.6 h
to embed (see the full-dataset section below). Chunking must run in parallel processes to keep up.

## Ingest a whole dataset (streaming, resumable)

`scripts/weaviate_v2/ingest_full.py` ingests all of `JuDDGES/pl-court-raw` or `JuDDGES/pl-nsa`
from the local HF cache (`--download` fetches missing files). The pipeline lives in
`juddges/data/weaviate_v2/stream.py`:

- **Units.** The work is cut into units of `--unit-rows` rows (default 1000) of one parquet row
  group, keyed `file:row_group:start`. Only one unit is read at a time, together with the
  columns the mapping needs, so the dataset is never held in memory.
- **Checkpoint.** Finished units go to a JSON checkpoint, by default
  `.context/weaviate_v2/ingest_<dataset>.json`. Re-running the same command skips them. The
  checkpoint stores `dataset` and `unit_rows` and refuses another `--unit-rows`, because the
  unit keys would not match.
- **Failures.** A unit that raises, for example when TEI is still failing after 5 retries, is
  recorded under `failed` and is not checkpointed. So is a unit whose Weaviate batch reports
  failed objects or references. The next run retries it. After 3 failed units in a row the run
  stops, because the infrastructure is probably down. The exit code is 1 when any unit failed.
- **Pipeline.** A process pool (`--workers`, default 6) chunks the next unit while the current
  one is embedded. TEI gets `--embed-batch` passages per request (default 128), with
  `--embed-concurrency` requests in flight (default 4). Each response goes straight into a
  Weaviate batch.
- **Order of writes.** Theses are embedded and the unit's judgments are imported and flushed
  before its chunks. Weaviate 1.40 rejects a chunk whose `parent_judgment` does not exist yet.
- **Empty texts.** Rows with an empty text are skipped and counted (`empty` in the log).

```bash
export WEAVIATE_V2_API_KEY=<key from .env>
# smoke run: two units
python scripts/weaviate_v2/ingest_full.py --dataset pl-court-raw --limit-units 2
# full run (same command resumes)
python scripts/weaviate_v2/ingest_full.py --dataset pl-court-raw \
  --log-file .context/weaviate_v2/ingest_pl-court-raw.log
# NSA sample of ~200 judgments, own checkpoint so a later full run can use 1000-row units
python scripts/weaviate_v2/ingest_full.py --dataset pl-nsa --unit-rows 100 --limit-units 2 \
  --checkpoint .context/weaviate_v2/ingest_pl-nsa.smoke.json
```

After every unit the log has a line like this:

```text
unit train_00_of_9.parquet:0:3000 done: 1000 judgments, 15180 chunks, 60 theses, 0 empty | 65.7s,
232.0 passages/s, chunk wait 0.0s | run: 2 units, 29800 chunks, 207 chunks/s, 434 units left, ETA 8.7 h
```

Interpret the per-unit fields as follows:

- `chunk wait` is how long embedding waited for the chunking pool. If it stays above 0 after
  the first unit, add `--workers`.
- The ETA is the remaining units times the average time per unit in this run.

### Running it in Docker

Run the ingester in a named container, with `--restart on-failure:3` so failed units are retried
automatically. The command is the usual `uv run` runner with `-d --name v2-ingest-plcourt` added:

- Mount the repository at `/w` with `PYTHONPATH=/w`.
- Mount the HF cache at `/hf` with `HF_HUB_CACHE=/hf` and `HF_HUB_OFFLINE=1`.
- Pass `--env-file weaviate/v2/.env.runtime` and `--network host`.

```bash
docker logs -f v2-ingest-plcourt 2>&1 | grep -E "unit .* (done|failed)"   # progress
tail -f .context/weaviate_v2/ingest_pl-court-raw.log
docker stop v2-ingest-plcourt          # stop; the current unit is redone on resume
docker start v2-ingest-plcourt         # resume with the same command
```

The container runs as root, so the checkpoint and log under `.context/weaviate_v2/` are owned
by root.

### Dataset mapping

**pl-court-raw.** The rows map to properties through `judgment_properties`, and `full_text` is
the chunk text. Naive dates are stored as UTC midnight.

**pl-nsa (CBOSA).** The rows map through `nsa_judgment_properties`:

| Property | Value |
|---|---|
| `source` | `nsa` |
| `source_url` | `https://orzeczenia.nsa.gov.pl` + `judgment_id` |
| `court_type` | `administrative_court` |
| `judgment_type` | Only the kind of decision: `Wyrok`, `Postanowienie` or `Uchwała`. The court is in `court_name`. |
| `outcome` | Built from `decision`. |
| `extracted_legal_bases` | `link` becomes `address`, `article` becomes `art`, and `law` becomes `title`. `isap_id` is the `id` query parameter of `link`. `text` is `journal` + `article`. |
| `date_issued` | The Warsaw calendar day, stored as UTC midnight. Converting to UTC instead would move it to the previous day. |

`submission_date` is the filing date and is not stored as `publication_date`.

The chunk text is `full_text` without its leading `TEZY` section (`nsa_chunk_text`):

- `full_text` is TEZY + SENTENCJA + UZASADNIENIE + ZDANIE ODRĘBNE. It also covers rows whose
  sections were not split into separate columns.
- The thesis is dropped because it has its own vector. Keeping it in a chunk would let thesis
  queries match it verbatim.
- A document that consists only of a thesis keeps its full text.

Measured on 2026-11-12, with TEI on the RTX 3090:

| Run | Judgments | Chunks | Time | Throughput |
|---|---|---|---|---|
| pl-court-raw smoke, 2 × 1000 rows | 2000 | 29 885 | 148 s | ~227 passages/s, GPU at 100 % |
| pl-nsa smoke, 2 × 100 rows | 200 | 1 946 | 20 s | ~10 chunks/judgment |

Embedding is the bottleneck. On a synthetic benchmark, raising TEI concurrency beyond 2 to 4
requests did not help: the GPU was already at 100 %, and 8 × 128 got `429` responses. The full
pl-court-raw run takes about 8.6 h.

## Stop / remove

```bash
docker compose down        # keeps data volumes
docker compose down -v     # also deletes weaviate_v2_data and the model cache
```

For backups, see [Weaviate backup and restore](weaviate-backup-restore.md). The
backup directory is `WEAVIATE_BACKUP_DIR`, which defaults to `weaviate/v2/backups`.
