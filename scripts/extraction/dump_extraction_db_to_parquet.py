#!/usr/bin/env python3
"""
Dump all extraction database tables to Parquet files.

Tables dumped:
- extraction_runs: Metadata about each extraction run
- extraction_results: Individual document extraction results with full inputs/outputs
- ingestion_logs: Tracks Weaviate ingestion of extracted data
- field_coverage: Field extraction success metrics

Usage:
    python dump_extraction_db_to_parquet.py --output-dir /mnt/readynas/datasets/legal-ai-weaviate
    python dump_extraction_db_to_parquet.py --table extraction_results  # Single table
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from sqlalchemy import create_engine, text

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

console = Console()

# Tables to dump
TABLES = [
    "extraction_runs",
    "extraction_results",
    "ingestion_logs",
    "field_coverage",
]


def get_db_connection_string() -> str:
    """Build PostgreSQL connection string from environment variables."""
    host = os.getenv("EXTRACTION_POSTGRES_HOST", "localhost")
    port = os.getenv("EXTRACTION_POSTGRES_PORT", "5433")
    user = os.getenv("EXTRACTION_POSTGRES_USER", "extraction_user")
    password = os.getenv("EXTRACTION_POSTGRES_PASSWORD", "extraction_pass")
    db = os.getenv("EXTRACTION_POSTGRES_DB", "legal_extraction")

    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def get_table_count(engine, table_name: str) -> int:
    """Get row count for a table."""
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        return result.scalar()


def convert_df_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame columns to Parquet-compatible types.

    Handles:
    - UUIDs -> strings
    - JSONB (dicts/lists) -> JSON strings
    """
    import json
    from uuid import UUID

    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(1)
            if len(sample) > 0:
                sample_val = sample.iloc[0]
                # Convert UUIDs to strings
                if isinstance(sample_val, UUID):
                    df[col] = df[col].apply(lambda x: str(x) if x is not None else None)
                # Convert dicts/lists to JSON strings
                elif isinstance(sample_val, (dict, list)):
                    df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else None)

    return df


def dump_table_to_parquet(
    engine,
    table_name: str,
    output_dir: Path,
    chunk_size: int = 50000,
) -> dict:
    """Dump a table to Parquet file(s).

    Args:
        engine: SQLAlchemy engine
        table_name: Name of the table to dump
        output_dir: Output directory for Parquet files
        chunk_size: Number of rows per chunk (for large tables)

    Returns:
        Dict with dump statistics
    """
    stats = {
        "table": table_name,
        "rows": 0,
        "files": [],
        "size_bytes": 0,
    }

    # Get total count
    total_rows = get_table_count(engine, table_name)
    stats["rows"] = total_rows

    if total_rows == 0:
        logger.warning(f"Table {table_name} is empty, skipping")
        return stats

    logger.info(f"Dumping {table_name}: {total_rows:,} rows")

    # For small tables, dump in one file
    if total_rows <= chunk_size:
        output_file = output_dir / f"{table_name}.parquet"

        df = pd.read_sql_table(table_name, engine)

        # Convert complex types for Parquet compatibility
        df = convert_df_for_parquet(df)

        df.to_parquet(output_file, engine="pyarrow", compression="snappy", index=False)

        file_size = output_file.stat().st_size
        stats["files"].append(str(output_file))
        stats["size_bytes"] = file_size

        logger.info(f"  Saved to {output_file} ({file_size / 1024 / 1024:.2f} MB)")

    else:
        # For large tables, dump in chunks
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed:,}/{task.total:,})"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Dumping {table_name}...", total=total_rows)

            chunk_num = 0
            offset = 0

            while offset < total_rows:
                # Read chunk
                query = f"SELECT * FROM {table_name} ORDER BY id LIMIT {chunk_size} OFFSET {offset}"
                df = pd.read_sql_query(query, engine)

                if df.empty:
                    break

                # Convert complex types for Parquet compatibility
                df = convert_df_for_parquet(df)

                # Save chunk
                output_file = output_dir / f"{table_name}_part{chunk_num:04d}.parquet"
                df.to_parquet(output_file, engine="pyarrow", compression="snappy", index=False)

                file_size = output_file.stat().st_size
                stats["files"].append(str(output_file))
                stats["size_bytes"] += file_size

                offset += len(df)
                chunk_num += 1
                progress.update(task, completed=offset)

        logger.info(f"  Saved {chunk_num} files, total size: {stats['size_bytes'] / 1024 / 1024:.2f} MB")

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dump extraction database to Parquet files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/readynas/datasets/legal-ai-weaviate",
        help="Output directory for Parquet files",
    )
    parser.add_argument(
        "--table",
        type=str,
        choices=TABLES,
        help="Dump only this table (default: all tables)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Rows per chunk for large tables (default: 50000)",
    )

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"extraction_backup_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(
            "[bold cyan]Extraction Database Backup to Parquet[/bold cyan]",
            border_style="cyan",
        )
    )

    console.print(f"\n[cyan]Configuration:[/cyan]")
    console.print(f"  - Output directory: {output_dir}")
    console.print(f"  - Chunk size: {args.chunk_size:,}")

    # Connect to database
    conn_string = get_db_connection_string()
    logger.info(f"Connecting to PostgreSQL...")

    try:
        engine = create_engine(conn_string)

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        console.print("[green]Connected to PostgreSQL[/green]\n")

    except Exception as e:
        console.print(f"[bold red]Failed to connect to database: {e}[/bold red]")
        sys.exit(1)

    # Determine which tables to dump
    tables_to_dump = [args.table] if args.table else TABLES

    # Get table sizes
    console.print("[cyan]Table sizes:[/cyan]")
    for table in tables_to_dump:
        try:
            count = get_table_count(engine, table)
            console.print(f"  - {table}: {count:,} rows")
        except Exception as e:
            console.print(f"  - {table}: [red]Error: {e}[/red]")

    console.print()

    # Dump tables
    all_stats = []

    for table in tables_to_dump:
        try:
            stats = dump_table_to_parquet(
                engine=engine,
                table_name=table,
                output_dir=output_dir,
                chunk_size=args.chunk_size,
            )
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"Failed to dump {table}: {e}")
            all_stats.append({"table": table, "error": str(e)})

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold green]Backup Complete![/bold green]\n")

    total_rows = sum(s.get("rows", 0) for s in all_stats)
    total_size = sum(s.get("size_bytes", 0) for s in all_stats)
    total_files = sum(len(s.get("files", [])) for s in all_stats)

    console.print(f"[cyan]Summary:[/cyan]")
    console.print(f"  - Output directory: {output_dir}")
    console.print(f"  - Total rows: {total_rows:,}")
    console.print(f"  - Total files: {total_files}")
    console.print(f"  - Total size: {total_size / 1024 / 1024:.2f} MB")

    console.print(f"\n[cyan]Tables backed up:[/cyan]")
    for stats in all_stats:
        if "error" in stats:
            console.print(f"  - {stats['table']}: [red]FAILED - {stats['error']}[/red]")
        else:
            size_mb = stats.get("size_bytes", 0) / 1024 / 1024
            console.print(f"  - {stats['table']}: {stats.get('rows', 0):,} rows ({size_mb:.2f} MB)")

    # Save metadata
    metadata = {
        "backup_timestamp": timestamp,
        "output_directory": str(output_dir),
        "tables": all_stats,
        "total_rows": total_rows,
        "total_size_bytes": total_size,
        "total_files": total_files,
    }

    import json
    metadata_file = output_dir / "backup_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    console.print(f"\n[dim]Metadata saved to: {metadata_file}[/dim]")


if __name__ == "__main__":
    main()
