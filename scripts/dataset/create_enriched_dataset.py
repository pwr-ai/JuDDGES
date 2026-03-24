#!/usr/bin/env python3
"""Create enriched HuggingFace datasets by joining extraction results.

This script joins extraction results from PostgreSQL with original HuggingFace
datasets and creates new enriched datasets with all extraction fields.

Supported datasets:
- juddges/pl-court-raw → juddges/pl-court-enriched (common courts)
- AI-TAX/pl-eureka-raw → AI-TAX/pl-eureka-enriched (tax interpretations)
- JuDDGES/pl-nsa → JuDDGES/pl-nsa-enriched (admin courts)

Usage:
    # Test with small sample
    python create_enriched_dataset.py \
        --source juddges/pl-court-raw \
        --output juddges/pl-court-enriched \
        --id-field judgment_id \
        --limit 1000

    # Full run (push to HuggingFace)
    python create_enriched_dataset.py \
        --source juddges/pl-court-raw \
        --output juddges/pl-court-enriched \
        --id-field judgment_id \
        --push
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import psycopg2
from datasets import Dataset, load_dataset, load_from_disk
from dotenv import load_dotenv
from huggingface_hub import HfApi
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

console = Console()

# Extraction fields to include in enriched dataset
EXTRACTION_FIELDS = {
    # Simple string fields
    "summary": "extracted_summary",
    "thesis": "extracted_thesis",
    "factual_state": "extracted_factual_state",
    "legal_state": "extracted_legal_state",
    # List fields
    "keywords": "extracted_keywords",
    # Complex JSON fields (serialized)
    "outcome": "extracted_outcome",
    "legal_references": "extracted_legal_references",
    "legal_concepts": "extracted_legal_concepts",
    "parties": "extracted_parties",
    "legal_analysis": "extracted_legal_analysis",
    "judgment_specific": "extracted_judgment_specific",
    "tax_interpretation_specific": "extracted_tax_specific",
}


def get_db_connection():
    """Create PostgreSQL connection to extraction database."""
    return psycopg2.connect(
        host=os.getenv("EXTRACTION_POSTGRES_HOST", "localhost"),
        port=os.getenv("EXTRACTION_POSTGRES_PORT", "5434"),
        user=os.getenv("EXTRACTION_POSTGRES_USER", "extraction_user"),
        password=os.getenv("EXTRACTION_POSTGRES_PASSWORD", "extraction_pass"),
        dbname=os.getenv("EXTRACTION_POSTGRES_DB", "legal_extraction"),
    )


def load_extraction_results(conn) -> dict[str, dict[str, Any]]:
    """Load all successful extraction results from PostgreSQL.

    Returns:
        Dict mapping document_id to extracted_data
    """
    cursor = conn.cursor()

    logger.info("Loading extraction results from PostgreSQL...")

    cursor.execute("""
        SELECT document_id, extracted_data
        FROM extraction_results
        WHERE extraction_status = 'success'
          AND extracted_data IS NOT NULL
    """)

    extractions = {}
    for document_id, extracted_data in cursor:
        if document_id and extracted_data:
            extractions[document_id] = extracted_data

    logger.info(f"Loaded {len(extractions):,} successful extractions")
    return extractions


def extract_field_value(extracted_data: dict, field_name: str) -> Any:
    """Extract a field value from extraction data, handling nested structures."""
    if not extracted_data:
        return None

    value = extracted_data.get(field_name)

    if value is None:
        return None

    # Handle different field types
    if field_name == "keywords":
        # Should be a list of strings
        if isinstance(value, list):
            return value
        elif isinstance(value, str):
            return [value]
        return None

    elif field_name in ("summary", "thesis", "factual_state", "legal_state"):
        # Simple string fields
        return str(value) if value else None

    else:
        # Complex fields - serialize to JSON string
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value) if value else None


def enrich_dataset(
    source_dataset: str,
    id_field: str,
    extractions: dict[str, dict[str, Any]],
    limit: Optional[int] = None,
) -> Dataset:
    """Load HuggingFace dataset and enrich with extraction data.

    Args:
        source_dataset: HuggingFace dataset name
        id_field: Field name containing document ID
        extractions: Dict mapping document_id to extracted_data
        limit: Optional limit for testing

    Returns:
        Enriched Dataset
    """
    console.print(f"\n[cyan]Loading dataset:[/cyan] {source_dataset}")

    # Load source dataset
    ds = load_dataset(source_dataset, split="train")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))
        console.print(f"[yellow]Limited to {len(ds):,} documents for testing[/yellow]")

    console.print(f"[green]Loaded {len(ds):,} documents[/green]")

    # Prepare new columns
    new_columns = {col_name: [] for col_name in EXTRACTION_FIELDS.values()}

    # Track statistics
    matched_count = 0
    total_count = len(ds)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed:,}/{task.total:,})"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Enriching documents...", total=total_count)

        for example in ds:
            # Get document ID
            doc_id = example.get(id_field)
            if doc_id is not None:
                doc_id = str(doc_id)

            # Look up extraction data
            extracted_data = extractions.get(doc_id) if doc_id else None

            if extracted_data:
                matched_count += 1

            # Add extraction fields
            for src_field, dst_field in EXTRACTION_FIELDS.items():
                value = extract_field_value(extracted_data, src_field) if extracted_data else None
                new_columns[dst_field].append(value)

            progress.update(task, advance=1)

    # Add new columns to dataset
    for col_name, col_values in new_columns.items():
        ds = ds.add_column(col_name, col_values)

    # Print statistics
    console.print(f"\n[bold green]Enrichment complete![/bold green]")
    console.print(f"  Total documents: {total_count:,}")
    console.print(f"  With extractions: {matched_count:,}")
    console.print(f"  Coverage: {100 * matched_count / total_count:.1f}%")

    return ds


def push_to_huggingface(
    dataset: Dataset,
    repo_id: str,
    private: bool = False,
) -> None:
    """Push enriched dataset to HuggingFace Hub.

    Args:
        dataset: Enriched dataset
        repo_id: HuggingFace repository ID
        private: Whether to create private repository
    """
    console.print(f"\n[cyan]Pushing to HuggingFace:[/cyan] {repo_id}")

    # Get token from HfApi
    api = HfApi()
    token = api.token

    # Create repository first using HfApi directly
    console.print(f"[dim]Creating repository {repo_id}...[/dim]")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    # Now push the dataset
    console.print(f"[dim]Uploading dataset...[/dim]")
    dataset.push_to_hub(
        repo_id,
        private=private,
        token=token,
        commit_message="Add enriched dataset with extraction fields",
    )

    console.print(f"[bold green]Successfully pushed to {repo_id}[/bold green]")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create enriched HuggingFace datasets with extraction results"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source HuggingFace dataset (e.g., juddges/pl-court-raw)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output HuggingFace dataset (e.g., juddges/pl-court-enriched)",
    )
    parser.add_argument(
        "--id-field",
        type=str,
        default=None,
        help="Field name containing document ID (e.g., judgment_id, id)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents (for testing)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to HuggingFace Hub",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository on HuggingFace",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Save to local directory instead of pushing",
    )
    parser.add_argument(
        "--from-local",
        type=str,
        default=None,
        help="Push existing local dataset (skip enrichment)",
    )

    args = parser.parse_args()

    # Validate required args when not using --from-local
    if not args.from_local:
        if not args.source:
            parser.error("--source is required when not using --from-local")
        if not args.id_field:
            parser.error("--id-field is required when not using --from-local")

    # Print configuration
    console.print(
        Panel.fit(
            "[bold cyan]Create Enriched HuggingFace Dataset[/bold cyan]",
            border_style="cyan",
        )
    )

    table = Table(show_header=False, box=None)
    table.add_column("Parameter", style="dim")
    table.add_column("Value")
    table.add_row("Source dataset", args.source or "N/A (from local)")
    table.add_row("Output dataset", args.output)
    table.add_row("ID field", args.id_field or "N/A (from local)")
    table.add_row("Limit", str(args.limit) if args.limit else "None (full)")
    table.add_row("Push to HF", str(args.push))
    table.add_row("Private", str(args.private))
    table.add_row("From local", args.from_local or "None")
    console.print(table)

    # If pushing from local, skip enrichment
    if args.from_local:
        console.print(f"\n[cyan]Loading local dataset:[/cyan] {args.from_local}")
        enriched_ds = load_from_disk(args.from_local)
        console.print(f"[green]Loaded {len(enriched_ds):,} documents[/green]")

        if args.push:
            push_to_huggingface(
                dataset=enriched_ds,
                repo_id=args.output,
                private=args.private,
            )
        else:
            console.print("[yellow]Use --push to push to HuggingFace[/yellow]")
        return

    # Connect to database and load extractions
    try:
        conn = get_db_connection()
        console.print("[green]Connected to PostgreSQL[/green]")
    except Exception as e:
        console.print(f"[bold red]Failed to connect to database: {e}[/bold red]")
        sys.exit(1)

    extractions = load_extraction_results(conn)
    conn.close()

    # Enrich dataset
    enriched_ds = enrich_dataset(
        source_dataset=args.source,
        id_field=args.id_field,
        extractions=extractions,
        limit=args.limit,
    )

    # Show sample of enriched data
    console.print("\n[cyan]Sample enriched document:[/cyan]")
    sample_idx = None
    for i, ex in enumerate(enriched_ds):
        if ex.get("extracted_summary"):
            sample_idx = i
            break

    if sample_idx is not None:
        sample = enriched_ds[sample_idx]
        console.print(f"  Document ID: {sample.get(args.id_field)}")
        console.print(f"  extracted_summary: {(sample.get('extracted_summary') or '')[:200]}...")
        console.print(f"  extracted_keywords: {sample.get('extracted_keywords')}")
    else:
        console.print("  [yellow]No enriched documents found in sample[/yellow]")

    # Save or push
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        enriched_ds.save_to_disk(str(output_path))
        console.print(f"[green]Saved to {output_path}[/green]")

    if args.push:
        push_to_huggingface(
            dataset=enriched_ds,
            repo_id=args.output,
            private=args.private,
        )

    if not args.output_dir and not args.push:
        console.print(
            "\n[yellow]Note: Dataset not saved. Use --push or --output-dir to save.[/yellow]"
        )


if __name__ == "__main__":
    main()
