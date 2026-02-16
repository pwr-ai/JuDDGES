#!/usr/bin/env python
"""Enrich HuggingFace datasets with Gemini-extracted factual_state and legal_state fields.

This script loads extractions from PostgreSQL and joins them with multiple HF datasets,
creating NEW enriched datasets (originals are never modified).

Usage:
    python scripts/dataset/enrich_hf_with_extractions.py --dataset pl-court-raw
    python scripts/dataset/enrich_hf_with_extractions.py --all
"""

from pathlib import Path
from typing import Any, Optional

import polars as pl
import typer
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Define paths directly to avoid heavy imports from settings.py
ROOT_PATH = Path(__file__).parent.parent.parent
DATA_PATH = ROOT_PATH / "data"

# Import ExtractionStorage directly to avoid __init__.py dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extraction_storage",
    ROOT_PATH / "juddges" / "extraction" / "extraction_storage.py"
)
extraction_storage_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extraction_storage_module)
ExtractionStorage = extraction_storage_module.ExtractionStorage

load_dotenv()
console = Console()

# Dataset configurations - creates NEW enriched datasets (originals unchanged)
DATASET_CONFIGS = {
    "pl-court-raw": {
        "input_path": DATA_PATH / "datasets" / "pl" / "raw",
        "output_path": DATA_PATH / "datasets" / "pl" / "raw-enriched",
        "id_field": "_id",  # Maps to document_id in PostgreSQL
        "secondary_field": "signature",  # Maps to document_number
        "format": "parquet_shards",
        "hf_repo_new": "JuDDGES/pl-court-raw-enriched",
    },
    "pl-eureka-raw": {
        "input_path": DATA_PATH / "datasets" / "eureka",
        "output_path": DATA_PATH / "datasets" / "eureka-enriched",
        "id_field": "id",
        "secondary_field": "docker_number",
        "format": "parquet",
        "hf_repo_new": "AI-TAX/pl-eureka-raw-enriched",
    },
    "pl-nsa": {
        "input_path": DATA_PATH / "datasets" / "nsa",
        "output_path": DATA_PATH / "datasets" / "nsa-enriched",
        "id_field": "judgment_id",
        "secondary_field": "docket_number",
        "format": "parquet",
        "hf_repo_new": "JuDDGES/pl-nsa-enriched",
    },
}

app = typer.Typer(help="Enrich HuggingFace datasets with Gemini extractions")


def load_extractions_from_postgres() -> pl.DataFrame:
    """Load all successful extractions from PostgreSQL as Polars DataFrame."""
    logger.info("Loading extractions from PostgreSQL...")
    storage = ExtractionStorage()
    extractions = storage.get_extractions_for_hf_dataset()

    if not extractions:
        logger.warning("No extractions found in PostgreSQL!")
        return pl.DataFrame()

    # Create DataFrame with explicit string schema to handle mixed types
    # All extracted fields are stored as strings (JSON strings for complex types)
    df = pl.DataFrame(extractions, infer_schema_length=None)
    logger.info(f"Loaded {len(df)} extractions from PostgreSQL")

    # Log coverage by document type
    type_counts = df.group_by("document_type").len().sort("len", descending=True)
    console.print("\n[bold]Extractions by document type:[/bold]")
    console.print(type_counts)

    return df


# All extracted fields to add to the enriched dataset
EXTRACTED_FIELDS = [
    "extracted_title",
    "extracted_date_issued",
    "extracted_summary",
    "extracted_thesis",
    "extracted_keywords",
    "factual_state",
    "legal_state",
    "extracted_outcome",
    "extracted_legal_references",
    "extracted_legal_concepts",
    "extracted_parties",
    "extracted_legal_analysis",
    "extracted_judgment_specific",
    "extracted_tax_interpretation_specific",
]


def enrich_parquet_shard(
    shard_path: Path,
    extractions_df: pl.DataFrame,
    output_path: Path,
    id_field: str,
    secondary_field: str,
) -> dict[str, Any]:
    """Enrich a single parquet shard with extraction data.

    Args:
        shard_path: Path to input parquet shard
        extractions_df: DataFrame with all extraction fields
        output_path: Path to write enriched parquet
        id_field: Primary ID field name in the shard (e.g., "_id", "id", "judgment_id")
        secondary_field: Secondary ID field name (e.g., "signature", "docker_number", "docket_number")

    Returns:
        Dict with statistics about the enrichment
    """
    # Read shard
    shard_df = pl.read_parquet(shard_path)
    original_count = len(shard_df)
    original_columns = shard_df.columns

    # Get available extraction columns (some might not exist in older data)
    available_fields = [f for f in EXTRACTED_FIELDS if f in extractions_df.columns]

    # Prepare extractions for join
    join_columns = ["document_id", "document_number"] + available_fields
    extractions_for_join = extractions_df.select([
        pl.col(c) for c in join_columns if c in extractions_df.columns
    ])

    # Primary join: by document_id = id_field (unique on document_id)
    primary_cols = ["document_id"] + [pl.col(f).alias(f"{f}_primary") for f in available_fields]
    primary_extractions = extractions_for_join.select(primary_cols).unique(subset=["document_id"])

    enriched_df = shard_df.join(
        primary_extractions,
        left_on=id_field,
        right_on="document_id",
        how="left",
    )

    # Secondary join: by document_number = secondary_field (for rows that didn't match)
    # Deduplicate on document_number to avoid creating extra rows
    secondary_cols = ["document_number"] + [pl.col(f).alias(f"{f}_secondary") for f in available_fields]
    secondary_extractions = extractions_for_join.select(secondary_cols).unique(subset=["document_number"])

    enriched_df = enriched_df.join(
        secondary_extractions,
        left_on=secondary_field,
        right_on="document_number",
        how="left",
    )

    # Coalesce: prefer primary match, fallback to secondary for each field
    coalesce_exprs = [
        pl.coalesce([f"{f}_primary", f"{f}_secondary"]).alias(f)
        for f in available_fields
    ]
    drop_cols = [f"{f}_primary" for f in available_fields] + [f"{f}_secondary" for f in available_fields]

    enriched_df = enriched_df.with_columns(coalesce_exprs).drop(drop_cols)

    # Validate
    assert len(enriched_df) == original_count, f"Row count mismatch! {len(enriched_df)} != {original_count}"
    assert all(
        col in enriched_df.columns for col in original_columns
    ), f"Missing original columns!"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    enriched_df.write_parquet(output_path)

    # Calculate statistics - count non-null values for key fields
    stats = {
        "shard": shard_path.name,
        "total": original_count,
    }

    # Count matches for key fields
    for field in ["factual_state", "legal_state", "extracted_summary", "extracted_thesis"]:
        if field in enriched_df.columns:
            count = enriched_df.filter(pl.col(field).is_not_null()).height
            stats[f"{field}_matched"] = count

    # Calculate overall coverage based on factual_state
    if "factual_state" in enriched_df.columns:
        factual_coverage = stats.get("factual_state_matched", 0) / original_count * 100 if original_count > 0 else 0
        stats["coverage_pct"] = round(factual_coverage, 2)

    return stats


def enrich_dataset(
    dataset_name: str,
    extractions_df: pl.DataFrame,
    config: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Enrich a single dataset with extractions.

    Args:
        dataset_name: Name of the dataset
        extractions_df: DataFrame with extractions
        config: Dataset configuration dict
        dry_run: If True, only show what would be done

    Returns:
        Dict with statistics about the enrichment
    """
    input_path = Path(config["input_path"])
    output_path = Path(config["output_path"])
    id_field = config["id_field"]
    secondary_field = config["secondary_field"]
    format_type = config["format"]

    console.print(f"\n[bold blue]Processing dataset: {dataset_name}[/bold blue]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_path}")
    console.print(f"  ID field: {id_field} (secondary: {secondary_field})")

    if not input_path.exists():
        logger.warning(f"Input path does not exist: {input_path}")
        return {"dataset": dataset_name, "error": "Input path not found"}

    if dry_run:
        console.print("[yellow]  DRY RUN - no files will be written[/yellow]")

    results = []

    if format_type == "parquet_shards":
        # Multiple parquet shards
        shard_files = sorted(input_path.glob("*.parquet"))
        if not shard_files:
            logger.warning(f"No parquet files found in {input_path}")
            return {"dataset": dataset_name, "error": "No parquet files found"}

        console.print(f"  Found {len(shard_files)} parquet shards")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Enriching {dataset_name}...", total=len(shard_files))

            for shard_path in shard_files:
                output_file = output_path / shard_path.name

                if not dry_run:
                    result = enrich_parquet_shard(
                        shard_path=shard_path,
                        extractions_df=extractions_df,
                        output_path=output_file,
                        id_field=id_field,
                        secondary_field=secondary_field,
                    )
                    results.append(result)
                else:
                    results.append({"shard": shard_path.name, "dry_run": True})

                progress.advance(task)

    else:
        # Single parquet file or directory
        parquet_files = list(input_path.glob("*.parquet")) if input_path.is_dir() else [input_path]

        for parquet_file in parquet_files:
            output_file = output_path / parquet_file.name if input_path.is_dir() else output_path

            if not dry_run:
                result = enrich_parquet_shard(
                    shard_path=parquet_file,
                    extractions_df=extractions_df,
                    output_path=output_file,
                    id_field=id_field,
                    secondary_field=secondary_field,
                )
                results.append(result)
            else:
                results.append({"shard": parquet_file.name, "dry_run": True})

    # Aggregate statistics
    if results and not dry_run and "error" not in results[0]:
        total_docs = sum(r.get("total", 0) for r in results)
        total_factual = sum(r.get("factual_state_matched", 0) for r in results)
        total_legal = sum(r.get("legal_state_matched", 0) for r in results)
        overall_coverage = round(total_factual / total_docs * 100, 2) if total_docs > 0 else 0

        return {
            "dataset": dataset_name,
            "total_documents": total_docs,
            "factual_state_matched": total_factual,
            "legal_state_matched": total_legal,
            "coverage_pct": overall_coverage,
            "shards_processed": len(results),
            "output_path": str(output_path),
        }

    return {"dataset": dataset_name, "results": results}


@app.command()
def enrich(
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Dataset to enrich (pl-court-raw, pl-eureka-raw, pl-nsa)",
    ),
    all_datasets: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Enrich all configured datasets",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without making changes",
    ),
):
    """Enrich HuggingFace datasets with Gemini extractions."""
    if not dataset and not all_datasets:
        console.print("[red]Error: Specify --dataset or --all[/red]")
        console.print("\nAvailable datasets:")
        for name in DATASET_CONFIGS:
            console.print(f"  - {name}")
        raise typer.Exit(1)

    # Load extractions
    extractions_df = load_extractions_from_postgres()
    if extractions_df.is_empty():
        console.print("[red]No extractions found! Run Gemini extraction first.[/red]")
        raise typer.Exit(1)

    # Process datasets
    datasets_to_process = list(DATASET_CONFIGS.keys()) if all_datasets else [dataset]
    all_results = []

    for ds_name in datasets_to_process:
        if ds_name not in DATASET_CONFIGS:
            console.print(f"[yellow]Warning: Unknown dataset '{ds_name}', skipping[/yellow]")
            continue

        result = enrich_dataset(
            dataset_name=ds_name,
            extractions_df=extractions_df,
            config=DATASET_CONFIGS[ds_name],
            dry_run=dry_run,
        )
        all_results.append(result)

    # Print summary
    console.print("\n[bold green]Enrichment Summary[/bold green]")
    table = Table(title="Results")
    table.add_column("Dataset", style="cyan")
    table.add_column("Total Docs", justify="right")
    table.add_column("Factual State", justify="right")
    table.add_column("Legal State", justify="right")
    table.add_column("Coverage %", justify="right")
    table.add_column("Output Path")

    for result in all_results:
        if "error" in result:
            table.add_row(
                result["dataset"],
                "-",
                "-",
                "-",
                "-",
                f"[red]{result['error']}[/red]",
            )
        elif "total_documents" in result:
            table.add_row(
                result["dataset"],
                str(result["total_documents"]),
                str(result["factual_state_matched"]),
                str(result["legal_state_matched"]),
                f"{result['coverage_pct']}%",
                result["output_path"],
            )
        else:
            table.add_row(
                result["dataset"],
                "-",
                "-",
                "-",
                "-",
                "[yellow]dry run[/yellow]",
            )

    console.print(table)


@app.command()
def list_datasets():
    """List all configured datasets."""
    console.print("[bold]Configured datasets:[/bold]\n")
    for name, config in DATASET_CONFIGS.items():
        console.print(f"[cyan]{name}[/cyan]")
        console.print(f"  Input: {config['input_path']}")
        console.print(f"  Output: {config['output_path']}")
        console.print(f"  ID field: {config['id_field']}")
        console.print(f"  Secondary: {config['secondary_field']}")
        console.print(f"  HF repo: {config['hf_repo_new']}")
        console.print()


if __name__ == "__main__":
    app()
