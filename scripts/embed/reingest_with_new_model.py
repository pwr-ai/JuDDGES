#!/usr/bin/env python3
"""
Re-ingest Weaviate collections with new embedding model (Qwen3-Embedding-0.6B).

Reads from the Parquet backup on NAS, generates new embeddings with Matryoshka
truncation (128-dim), and ingests into fresh collections with Binary Quantization.

Usage:
    # Full reingest from NAS backup
    python scripts/embed/reingest_with_new_model.py

    # Test with limited docs
    python scripts/embed/reingest_with_new_model.py --limit 1000

    # Skip collection recreation (if already done)
    python scripts/embed/reingest_with_new_model.py --skip-recreate

Requires:
    - Weaviate running with backup-filesystem module
    - Native backup already exists on NAS (for safety)
    - Qwen3-Embedding-0.6B available (downloads on first run)
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import weaviate
import weaviate.classes.config as wvc

from juddges.settings import TEXT_EMBEDDING_DIM, TEXT_EMBEDDING_MODEL, VectorName

# Load environment
load_dotenv(Path(__file__).parent.parent.parent / "weaviate" / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

console = Console()

# Paths
NAS_BACKUP_DIR = Path("/mnt/readynas/datasets/legal-ai-weaviate")
LEGAL_DOCS_PARQUET = NAS_BACKUP_DIR / "LegalDocuments_20260107_175321.parquet"
DOC_CHUNKS_PARQUET = NAS_BACKUP_DIR / "DocumentChunks_20260107_211151.parquet"

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8084"))
WEAVIATE_GRPC_PORT = 8085 if WEAVIATE_HOST in ("localhost", "127.0.0.1") else int(os.getenv("WEAVIATE_GRPC_PORT", "8085"))
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")

BATCH_SIZE = 1024
EMBED_BATCH_SIZE = 128
# Max chars to embed per text - TEI handles truncation server-side
MAX_EMBED_CHARS = 2048  # Shorter = faster embedding, BM25 handles precision
# TEI embedding server endpoint
TEI_URL = os.getenv("TEI_URL", "http://localhost:8082")
TEI_SESSION = None


def get_client() -> weaviate.WeaviateClient:
    """Connect to Weaviate."""
    auth = weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None
    return weaviate.connect_to_custom(
        http_host=WEAVIATE_HOST,
        http_port=WEAVIATE_PORT,
        http_secure=False,
        grpc_host=WEAVIATE_HOST,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=False,
        auth_credentials=auth,
        skip_init_checks=False,
    )


def recreate_collections(client: weaviate.WeaviateClient) -> None:
    """Delete old collections and create new ones with BQ."""
    for name in ["LegalDocuments", "DocumentChunks"]:
        try:
            client.collections.delete(name)
            logger.info(f"Deleted collection: {name}")
        except Exception:
            logger.info(f"Collection {name} does not exist, skipping delete")

    _create_legal_documents_collection(client)
    _create_document_chunks_collection(client)


def _create_legal_documents_collection(client: weaviate.WeaviateClient) -> None:
    """Create LegalDocuments collection with BQ and single vector."""
    client.collections.create(
        name="LegalDocuments",
        description="Legal documents with BQ-quantized embeddings",
        vectorizer_config=[
            wvc.Configure.NamedVectors.none(
                name=VectorName.DEFAULT,
                vector_index_config=wvc.Configure.VectorIndex.hnsw(
                    ef_construction=64,
                    max_connections=16,
                    distance_metric=wvc.VectorDistances.COSINE,
                    quantizer=wvc.Configure.VectorIndex.Quantizer.bq(
                        rescore_limit=200,
                    ),
                ),
            ),
        ],
        inverted_index_config=wvc.Configure.inverted_index(
            bm25_b=0.75,
            bm25_k1=1.2,
            stopwords_preset=wvc.StopwordsPreset.EN,
        ),
        properties=[
            wvc.Property(name="document_id", data_type=wvc.DataType.TEXT, index_filterable=True, index_searchable=False, tokenization=wvc.Tokenization.FIELD, skip_vectorization=True),
            wvc.Property(name="full_text", data_type=wvc.DataType.TEXT, index_searchable=True, tokenization=wvc.Tokenization.WORD, skip_vectorization=True),
            wvc.Property(name="title", data_type=wvc.DataType.TEXT, index_searchable=True, tokenization=wvc.Tokenization.WORD, skip_vectorization=True),
            wvc.Property(name="summary", data_type=wvc.DataType.TEXT, index_searchable=True, tokenization=wvc.Tokenization.WORD, skip_vectorization=True),
            wvc.Property(name="document_type", data_type=wvc.DataType.TEXT, index_filterable=True, tokenization=wvc.Tokenization.FIELD, skip_vectorization=True),
            wvc.Property(name="language", data_type=wvc.DataType.TEXT, index_filterable=True, tokenization=wvc.Tokenization.FIELD, skip_vectorization=True),
            wvc.Property(name="country", data_type=wvc.DataType.TEXT, index_filterable=True, tokenization=wvc.Tokenization.FIELD, skip_vectorization=True),
            wvc.Property(name="court_name", data_type=wvc.DataType.TEXT, index_filterable=True, skip_vectorization=True),
            wvc.Property(name="date_issued", data_type=wvc.DataType.TEXT, index_filterable=True, skip_vectorization=True),
            wvc.Property(name="document_number", data_type=wvc.DataType.TEXT, index_filterable=True, index_searchable=True, tokenization=wvc.Tokenization.WORD, skip_vectorization=True),
            wvc.Property(name="source", data_type=wvc.DataType.TEXT, index_filterable=True, tokenization=wvc.Tokenization.FIELD, skip_vectorization=True),
            wvc.Property(name="processing_status", data_type=wvc.DataType.TEXT, index_filterable=True, tokenization=wvc.Tokenization.FIELD, skip_vectorization=True),
        ],
    )
    logger.info("Created LegalDocuments collection with BQ")


def _create_document_chunks_collection(client: weaviate.WeaviateClient) -> None:
    """Create DocumentChunks collection with BQ and single vector."""
    client.collections.create(
        name="DocumentChunks",
        description="Document chunks with BQ-quantized embeddings",
        vectorizer_config=[
            wvc.Configure.NamedVectors.none(
                name=VectorName.DEFAULT,
                vector_index_config=wvc.Configure.VectorIndex.hnsw(
                    ef_construction=64,
                    max_connections=16,
                    distance_metric=wvc.VectorDistances.COSINE,
                    quantizer=wvc.Configure.VectorIndex.Quantizer.bq(
                        rescore_limit=200,
                    ),
                ),
            ),
        ],
        inverted_index_config=wvc.Configure.inverted_index(
            bm25_b=0.75,
            bm25_k1=1.2,
            stopwords_preset=wvc.StopwordsPreset.EN,
        ),
        properties=[
            wvc.Property(name="document_id", data_type=wvc.DataType.TEXT, index_filterable=True, index_searchable=False, tokenization=wvc.Tokenization.FIELD, skip_vectorization=True),
            wvc.Property(name="chunk_id", data_type=wvc.DataType.INT, index_filterable=True, skip_vectorization=True),
            wvc.Property(name="chunk_text", data_type=wvc.DataType.TEXT, index_searchable=True, tokenization=wvc.Tokenization.WORD, skip_vectorization=True),
            wvc.Property(name="document_type", data_type=wvc.DataType.TEXT, index_filterable=True, tokenization=wvc.Tokenization.FIELD, skip_vectorization=True),
            wvc.Property(name="language", data_type=wvc.DataType.TEXT, index_filterable=True, tokenization=wvc.Tokenization.FIELD, skip_vectorization=True),
            wvc.Property(name="position", data_type=wvc.DataType.INT, index_filterable=True, skip_vectorization=True),
            wvc.Property(name="segment_type", data_type=wvc.DataType.TEXT, index_filterable=True, tokenization=wvc.Tokenization.FIELD, skip_vectorization=True),
        ],
    )
    logger.info("Created DocumentChunks collection with BQ")


def ingest_collection(
    client: weaviate.WeaviateClient,
    parquet_path: Path,
    collection_name: str,
    text_field: str,
    limit: int | None = None,
) -> int:
    """Ingest a collection from Parquet backup with new embeddings."""
    if not parquet_path.exists():
        logger.error(f"Parquet file not found: {parquet_path}")
        return 0

    collection = client.collections.get(collection_name)

    # Read parquet metadata for row count
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    if limit:
        total_rows = min(total_rows, limit)

    console.print(f"\n[bold]{collection_name}[/bold]: {total_rows:,} objects from {parquet_path.name}")

    count = 0
    batch_objects = []
    batch_texts = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        task = progress.add_task(f"Ingesting {collection_name}", total=total_rows)

        for batch in pf.iter_batches(batch_size=BATCH_SIZE):
            df = batch.to_pandas()

            for _, row in df.iterrows():
                if limit and count >= limit:
                    break

                props = {}
                text_for_embedding = ""

                for col in df.columns:
                    if col == "uuid":
                        continue
                    val = row[col]
                    if val is None or (isinstance(val, str) and val == ""):
                        continue
                    # Parse JSON arrays back
                    if isinstance(val, str) and val.startswith("["):
                        try:
                            parsed = json.loads(val)
                            # Keep as JSON string for TEXT fields that get arrays
                            if isinstance(parsed, list) and col not in ("keywords", "judges", "legal_bases", "references", "tags", "cited_references"):
                                val = json.dumps(parsed, ensure_ascii=False)
                            else:
                                val = parsed
                        except (json.JSONDecodeError, ValueError):
                            pass
                    # Coerce float-encoded integers
                    if col in ("chunk_id", "position") and isinstance(val, (str, float)):
                        try:
                            val = int(float(val))
                        except (ValueError, TypeError):
                            val = 0
                    # Fix date formats to RFC3339
                    if col in ("date_issued", "publication_date", "ingestion_date", "last_updated"):
                        if isinstance(val, str) and len(val) == 10:
                            val = f"{val}T00:00:00Z"
                    props[col] = val

                text_for_embedding = str(props.get(text_field, ""))
                if not text_for_embedding:
                    count += 1
                    progress.update(task, completed=count)
                    continue

                uuid = row.get("uuid", weaviate.util.generate_uuid5(str(props.get("document_id", count))))

                batch_objects.append((uuid, props))
                batch_texts.append(text_for_embedding[:MAX_EMBED_CHARS])

                # Process batch when full
                if len(batch_texts) >= EMBED_BATCH_SIZE:
                    _flush_batch(collection, batch_objects, batch_texts)
                    count += len(batch_texts)
                    batch_objects = []
                    batch_texts = []
                    progress.update(task, completed=count)

            if limit and count >= limit:
                break

        # Flush remaining
        if batch_texts:
            _flush_batch(collection, batch_objects, batch_texts)
            count += len(batch_texts)
            progress.update(task, completed=count)

    return count


def _embed_via_tei(texts: list[str], truncate_dim: int = TEXT_EMBEDDING_DIM) -> list[list[float]]:
    """Generate embeddings via TEI HTTP API with Matryoshka truncation and retry."""
    global TEI_SESSION
    if TEI_SESSION is None:
        TEI_SESSION = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=4, max_retries=3
        )
        TEI_SESSION.mount("http://", adapter)

    for attempt in range(5):
        try:
            resp = TEI_SESSION.post(
                f"{TEI_URL}/embed",
                json={"inputs": texts, "truncate": True},
                timeout=120,
            )
            resp.raise_for_status()
            full_embeddings = resp.json()

            # Matryoshka truncation: take first N dims
            if truncate_dim and truncate_dim < len(full_embeddings[0]):
                return [emb[:truncate_dim] for emb in full_embeddings]
            return full_embeddings
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            wait = 2 ** attempt
            logger.warning(f"TEI request failed (attempt {attempt + 1}/5): {e}. Retrying in {wait}s...")
            time.sleep(wait)
            # Reset session on connection errors
            TEI_SESSION = None
            TEI_SESSION = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=4, pool_maxsize=4, max_retries=3
            )
            TEI_SESSION.mount("http://", adapter)

    raise RuntimeError(f"TEI embedding failed after 5 attempts for batch of {len(texts)} texts")


def _flush_batch(
    collection: weaviate.collections.Collection,
    batch_objects: list,
    batch_texts: list,
) -> None:
    """Generate embeddings via TEI and insert batch."""
    # Split into sub-batches for TEI (max 256 per request)
    all_embeddings = []
    for i in range(0, len(batch_texts), EMBED_BATCH_SIZE):
        chunk = batch_texts[i : i + EMBED_BATCH_SIZE]
        embeddings = _embed_via_tei(chunk)
        all_embeddings.extend(embeddings)

    with collection.batch.dynamic() as batch:
        for (uuid, props), embedding in zip(batch_objects, all_embeddings):
            batch.add_object(
                uuid=uuid,
                properties=props,
                vector={VectorName.DEFAULT: embedding},
            )


def main():
    parser = argparse.ArgumentParser(description="Re-ingest Weaviate with new embedding model")
    parser.add_argument("--limit", type=int, help="Limit number of documents per collection")
    parser.add_argument("--skip-recreate", action="store_true", help="Skip collection recreation")
    parser.add_argument("--collection", choices=["LegalDocuments", "DocumentChunks", "all"], default="all")
    parser.add_argument("--tei-url", type=str, default=None, help="TEI server URL (default: http://localhost:8082)")
    args = parser.parse_args()

    global TEI_URL
    if args.tei_url:
        TEI_URL = args.tei_url

    console.print(f"\n[bold cyan]Weaviate Re-ingestion with {TEXT_EMBEDDING_MODEL}[/bold cyan]")
    console.print(f"Embedding dim: {TEXT_EMBEDDING_DIM} (Matryoshka truncation)")
    console.print(f"Binary Quantization: enabled")
    if args.limit:
        console.print(f"Limit: {args.limit:,} per collection")

    # Verify parquet backups exist
    for p in [LEGAL_DOCS_PARQUET, DOC_CHUNKS_PARQUET]:
        if not p.exists():
            console.print(f"[red]Missing backup:[/red] {p}")
            sys.exit(1)

    # Test TEI embedding server
    console.print(f"\nTesting TEI server at {TEI_URL}...")
    try:
        test_resp = requests.get(f"{TEI_URL}/health", timeout=5)
        test_resp.raise_for_status()
        test_emb = _embed_via_tei(["test"])
        console.print(f"[green]TEI server OK[/green] (output dim: {len(test_emb[0])})")
    except Exception as e:
        console.print(f"[red]TEI server not available:[/red] {e}")
        console.print(f"Start it with: docker compose up -d embedding-server")
        sys.exit(1)

    # Connect to Weaviate
    client = None
    try:
        client = get_client()
        console.print("[green]Connected to Weaviate[/green]")

        if not args.skip_recreate:
            console.print("\n[yellow]Recreating collections...[/yellow]")
            recreate_collections(client)

        start_time = time.time()
        total_ingested = 0

        if args.collection in ("LegalDocuments", "all"):
            n = ingest_collection(
                client, LEGAL_DOCS_PARQUET, "LegalDocuments", "full_text", args.limit
            )
            total_ingested += n
            console.print(f"[green]LegalDocuments: {n:,} ingested[/green]")

        if args.collection in ("DocumentChunks", "all"):
            n = ingest_collection(
                client, DOC_CHUNKS_PARQUET, "DocumentChunks", "chunk_text", args.limit
            )
            total_ingested += n
            console.print(f"[green]DocumentChunks: {n:,} ingested[/green]")

        elapsed = time.time() - start_time
        rate = total_ingested / elapsed if elapsed > 0 else 0
        console.print(f"\n[bold green]Done![/bold green] {total_ingested:,} objects in {elapsed/60:.1f} min ({rate:.0f} obj/sec)")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.exception("Reingest failed")
        sys.exit(1)
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    main()
