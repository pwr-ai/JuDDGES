#!/usr/bin/env python
"""Import extraction results from JSONL files to PostgreSQL.

Usage:
    python scripts/dataset/import_extractions_to_postgres.py
"""

import json
from pathlib import Path
from uuid import uuid4

import typer
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Define paths directly to avoid heavy imports
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
app = typer.Typer(help="Import extraction results from JSONL files to PostgreSQL")


def import_jsonl_file(file_path: Path, storage: ExtractionStorage, run_id) -> dict:
    """Import a single JSONL file to PostgreSQL.

    Returns:
        Dict with statistics about the import
    """
    imported = 0
    skipped = 0
    errors = 0

    with open(file_path, "r") as f:
        for line in f:
            try:
                record = json.loads(line.strip())

                # Skip failed extractions
                if record.get("extraction_status") != "success":
                    skipped += 1
                    continue

                # Extract the data we need
                document_id = record.get("document_id", "")
                document_number = record.get("document_number", "")
                document_type = record.get("document_type", "")
                extracted_data = record.get("extracted_data", {})

                # Skip if no extracted data
                if not extracted_data:
                    skipped += 1
                    continue

                # Insert into PostgreSQL using the existing method
                storage.save_extraction_result(
                    run_id=run_id,
                    document_id=document_id,
                    document_number=document_number,
                    document_type=document_type,
                    full_text="",  # Not needed for enrichment
                    extraction_status="success",
                    extracted_data=extracted_data,
                )
                imported += 1

            except json.JSONDecodeError as e:
                errors += 1
                logger.warning(f"JSON decode error in {file_path}: {e}")
            except Exception as e:
                errors += 1
                logger.warning(f"Error importing record from {file_path}: {e}")

    return {
        "file": file_path.name,
        "imported": imported,
        "skipped": skipped,
        "errors": errors,
    }


@app.command()
def import_all(
    data_dir: Path = typer.Option(
        DATA_PATH / "extraction_results",
        "--data-dir",
        "-d",
        help="Directory containing extraction results",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be imported without making changes",
    ),
):
    """Import all JSONL extraction files to PostgreSQL."""

    # Find all JSONL files
    jsonl_files = list(data_dir.rglob("*_extracted.jsonl"))

    if not jsonl_files:
        console.print(f"[red]No *_extracted.jsonl files found in {data_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Found {len(jsonl_files)} JSONL files to import[/bold]\n")

    for f in jsonl_files:
        console.print(f"  - {f.relative_to(data_dir)}")

    if dry_run:
        console.print("\n[yellow]DRY RUN - no changes made[/yellow]")
        return

    # Initialize storage
    storage = ExtractionStorage()

    # Create an extraction run record for this import
    run_id = storage.create_extraction_run(
        model_name="import-from-jsonl",
        sample_size=0,
        batch_size=1,
        max_workers=1,
        weaviate_host="localhost",
        weaviate_port=8080,
        notes=f"Imported from JSONL files in {data_dir}",
    )
    console.print(f"\n[bold]Import run ID: {run_id}[/bold]")

    # Import each file
    results = []
    total_imported = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Importing...", total=len(jsonl_files))

        for jsonl_file in jsonl_files:
            progress.update(task, description=f"Importing {jsonl_file.name}...")
            result = import_jsonl_file(jsonl_file, storage, run_id)
            results.append(result)
            total_imported += result["imported"]
            progress.advance(task)

    # Print summary
    console.print("\n[bold green]Import Summary[/bold green]")
    console.print(f"Total imported: {total_imported}")

    for result in results:
        console.print(
            f"  {result['file']}: {result['imported']} imported, "
            f"{result['skipped']} skipped, {result['errors']} errors"
        )


if __name__ == "__main__":
    app()
