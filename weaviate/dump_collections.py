#!/usr/bin/env python3
"""
Dump Weaviate collections to Parquet files using GRPC protocol.

GRPC is ~10x faster than REST API for bulk exports.

Usage:
    python dump_collections.py --output-dir /mnt/readynas/datasets/legal-ai-weaviate
    python dump_collections.py --collection LegalDocuments
    python dump_collections.py --limit 10000  # Test

Estimated time:
    - LegalDocuments (3.2M): ~20-30 min
    - DocumentChunks (37.8M): ~4-6 hours
    - TOTAL: ~5-7 hours
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import weaviate
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

console = Console()

# Load environment
load_dotenv(Path(__file__).parent / ".env", override=True)
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Configuration
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8084"))
# External GRPC port is 8085 (maps to internal 50051 in docker)
WEAVIATE_GRPC_PORT = 8085  # Hardcoded for remote access
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")

COLLECTIONS = ["LegalDocuments", "DocumentChunks"]
WRITE_BATCH_SIZE = 50_000  # Rows per parquet row group


def get_client() -> weaviate.WeaviateClient:
    """Create Weaviate client with GRPC."""
    console.print(f"  Connecting to {WEAVIATE_HOST}:{WEAVIATE_PORT} (GRPC: {WEAVIATE_GRPC_PORT})")

    client = weaviate.connect_to_custom(
        http_host=WEAVIATE_HOST,
        http_port=WEAVIATE_PORT,
        http_secure=False,
        grpc_host=WEAVIATE_HOST,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=False,
        auth_credentials=weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None,
        skip_init_checks=False,
    )
    return client


def serialize_value(value: Any) -> Any:
    """Serialize values for export."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    return value


def get_schema_from_weaviate(client: weaviate.WeaviateClient, collection_name: str) -> list[str]:
    """Get ordered list of property names from Weaviate schema."""
    collection = client.collections.get(collection_name)
    config = collection.config.get()
    return sorted([prop.name for prop in config.properties])


def dump_collection_grpc(
    client: weaviate.WeaviateClient,
    collection_name: str,
    output_path: Path,
    limit: int | None = None,
) -> int:
    """Dump collection using GRPC iterator (fast!)."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        console.print("[red]Error:[/red] pyarrow not installed. Run: pip install pyarrow")
        sys.exit(1)

    collection = client.collections.get(collection_name)

    # Get total count
    total = collection.aggregate.over_all(total_count=True).total_count or 0
    if limit:
        total = min(total, limit)

    # Get fixed schema from Weaviate (sorted for consistency)
    property_names = get_schema_from_weaviate(client, collection_name)
    console.print(f"  Total objects: {total:,}")
    console.print(f"  Properties: {len(property_names)}")

    # Build PyArrow schema with fixed column order
    schema_fields = [("uuid", pa.string())]
    for prop in property_names:
        schema_fields.append((prop, pa.string()))  # All as strings for safety
    # Add x, y as float if they exist
    schema = pa.schema([(name, pa.string()) for name, _ in schema_fields])

    count = 0
    batch = []
    writer = None

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
        task = progress.add_task(f"Dumping {collection_name}", total=total)

        # Use GRPC iterator - much faster than REST pagination!
        for obj in collection.iterator(include_vector=False):
            # Build doc with fixed column order
            doc = {"uuid": str(obj.uuid)}

            for prop in property_names:
                value = obj.properties.get(prop)
                serialized = serialize_value(value)
                # Convert lists to JSON strings for Parquet
                if isinstance(serialized, list):
                    serialized = json.dumps(serialized, ensure_ascii=False) if serialized else None
                # Convert numbers to strings for consistent schema
                if serialized is not None and not isinstance(serialized, str):
                    serialized = str(serialized)
                doc[prop] = serialized

            batch.append(doc)
            count += 1

            if count % 1000 == 0:
                progress.update(task, completed=count)

            # Write batch when full
            if len(batch) >= WRITE_BATCH_SIZE:
                table = pa.Table.from_pylist(batch, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        schema,
                        compression="zstd",
                        compression_level=3,
                    )
                writer.write_table(table)
                batch = []

            if limit and count >= limit:
                break

        # Write remaining batch
        if batch:
            table = pa.Table.from_pylist(batch, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    schema,
                    compression="zstd",
                    compression_level=3,
                )
            writer.write_table(table)

        if writer:
            writer.close()

        progress.update(task, completed=count)

    return count


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    if size_bytes > 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes > 1024**2:
        return f"{size_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_bytes / 1024:.2f} KB"


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fast Weaviate dump using GRPC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--collection",
        choices=COLLECTIONS + ["all"],
        default="all",
        help="Collection to dump (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dumps"),
        help="Output directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of documents",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    collections = COLLECTIONS if args.collection == "all" else [args.collection]

    console.print(f"\n[bold cyan]Weaviate GRPC Dump (Fast!)[/bold cyan]")
    console.print("=" * 60)
    console.print(f"Host: {WEAVIATE_HOST}")
    console.print(f"HTTP Port: {WEAVIATE_PORT}")
    console.print(f"GRPC Port: {WEAVIATE_GRPC_PORT}")
    console.print(f"Output: {args.output_dir.absolute()}")
    if args.limit:
        console.print(f"Limit: {args.limit:,}")
    console.print()

    client = None
    try:
        client = get_client()
        console.print("[green]✓[/green] Connected via GRPC\n")

        start_time = time.time()

        for collection_name in collections:
            collection = client.collections.get(collection_name)
            count = collection.aggregate.over_all(total_count=True).total_count or 0
            console.print(f"[bold]{collection_name}[/bold]: {count:,} objects")

            if count == 0:
                console.print("[yellow]  Skipping empty collection[/yellow]")
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{collection_name}_{timestamp}.parquet"
            output_path = args.output_dir / filename

            col_start = time.time()
            exported = dump_collection_grpc(
                client,
                collection_name,
                output_path,
                limit=args.limit,
            )
            col_elapsed = time.time() - col_start

            size_str = format_size(output_path.stat().st_size)
            rate = exported / col_elapsed if col_elapsed > 0 else 0

            console.print(f"[green]✓[/green] Exported {exported:,} documents")
            console.print(f"  File: {output_path}")
            console.print(f"  Size: {size_str}")
            console.print(f"  Time: {col_elapsed/60:.1f} min ({rate:.0f} docs/sec)")
            console.print()

        total_elapsed = time.time() - start_time
        hours = total_elapsed / 3600
        if hours >= 1:
            console.print(f"[bold green]✓ Dump completed in {hours:.1f} hours[/bold green]")
        else:
            console.print(f"[bold green]✓ Dump completed in {total_elapsed/60:.1f} minutes[/bold green]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.exception("Dump failed")
        sys.exit(1)
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    main()
