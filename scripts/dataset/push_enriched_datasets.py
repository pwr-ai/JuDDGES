#!/usr/bin/env python
"""Push enriched datasets to Hugging Face Hub.

Creates NEW repositories for enriched datasets with factual_state and legal_state fields.

Usage:
    python scripts/dataset/push_enriched_datasets.py --dataset pl-court-raw-enriched
    python scripts/dataset/push_enriched_datasets.py --all
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    DatasetCardData,
    HfApi,
    create_repo,
)
from huggingface_hub.errors import RepositoryNotFoundError
from loguru import logger
from rich.console import Console
from rich.table import Table

load_dotenv()
console = Console()

# Define paths directly to avoid heavy imports
ROOT_PATH = Path(__file__).parent.parent.parent
DATA_PATH = ROOT_PATH / "data"

# Dataset configurations for enriched datasets
ENRICHED_DATASETS = {
    "pl-court-raw-enriched": {
        "data_dir": DATA_PATH / "datasets" / "pl" / "raw-enriched",
        "hf_repo": "JuDDGES/pl-court-raw-enriched",
        "original_repo": "JuDDGES/pl-court-raw",
        "pretty_name": "Polish Court Judgments Raw (Enriched)",
        "description": "Polish court judgments enriched with Gemini-extracted factual_state and legal_state fields.",
        "language": "pl",
        "size_category": "100K<n<1M",
    },
    "pl-eureka-raw-enriched": {
        "data_dir": DATA_PATH / "datasets" / "eureka-enriched",
        "hf_repo": "AI-TAX/pl-eureka-raw-enriched",
        "original_repo": "AI-TAX/pl-eureka-raw",
        "pretty_name": "Polish Tax Interpretations (Enriched)",
        "description": "Polish tax interpretations enriched with Gemini-extracted factual_state and legal_state fields.",
        "language": "pl",
        "size_category": "10K<n<100K",
    },
    "pl-nsa-enriched": {
        "data_dir": DATA_PATH / "datasets" / "nsa-enriched",
        "hf_repo": "JuDDGES/pl-nsa-enriched",
        "original_repo": "JuDDGES/pl-nsa",
        "pretty_name": "Polish NSA Judgments (Enriched)",
        "description": "Polish Supreme Administrative Court judgments enriched with Gemini-extracted factual_state and legal_state fields.",
        "language": "pl",
        "size_category": "1M<n<10M",
    },
}

DEFAULT_DATA_DIR_IN_REPO = "data"
DATA_SHARD_FILE_PATTERN = "*.parquet"

app = typer.Typer(help="Push enriched datasets to Hugging Face Hub")


def generate_readme_content(config: dict) -> str:
    """Generate README content for enriched dataset."""
    return f"""# {config['pretty_name']}

{config['description']}

## Dataset Description

This dataset is an enriched version of [{config['original_repo']}](https://huggingface.co/datasets/{config['original_repo']}) with additional fields extracted using Google Gemini 2.5 Pro.

### New Fields

#### Core Extracted Fields

| Field | Type | Description |
|-------|------|-------------|
| `factual_state` | string | Objective narrative of facts (stan faktyczny) - the factual circumstances forming the basis for the case |
| `legal_state` | string | Legal framework and provisions (stan prawny) - applicable laws and legal provisions used in reasoning |
| `extracted_title` | string | Extracted document title |
| `extracted_date_issued` | string | Extracted issue date (YYYY-MM-DD format) |
| `extracted_summary` | string | Brief summary of the document |
| `extracted_thesis` | string | Legal thesis or principle established by the document |
| `extracted_keywords` | JSON string | List of keywords extracted from the document |

#### Structured Legal Data

| Field | Type | Description |
|-------|------|-------------|
| `extracted_outcome` | JSON string | Decision outcome with decision_type and decision_summary |
| `extracted_legal_references` | JSON string | List of cited laws, regulations, and legal acts |
| `extracted_legal_concepts` | JSON string | Legal concepts mentioned with definitions and context |
| `extracted_parties` | JSON string | Parties involved in the case with roles and representation |
| `extracted_legal_analysis` | JSON string | Detailed legal reasoning analysis |

#### Document-Type Specific Fields

| Field | Type | Description |
|-------|------|-------------|
| `extracted_judgment_specific` | JSON string | Fields specific to court judgments |
| `extracted_tax_interpretation_specific` | JSON string | Fields specific to tax interpretations |

### Data Processing

- **Extraction Model**: Google Gemini 2.5 Pro
- **Extraction Method**: Structured output extraction with Polish legal schema
- **Join Strategy**: Primary join on `document_id`, fallback on `document_number`

## Usage

```python
from datasets import load_dataset
import json

dataset = load_dataset("{config['hf_repo']}")

# Access text fields directly
print(dataset['train'][0]['factual_state'])

# Parse JSON fields
legal_refs = json.loads(dataset['train'][0]['extracted_legal_references'])
```

## Citation

If you use this dataset, please cite the original dataset and the JuDDGES project.

## License

Same as the original dataset: [{config['original_repo']}](https://huggingface.co/datasets/{config['original_repo']})

---

*Generated on {datetime.now().strftime('%Y-%m-%d')} by JuDDGES enrichment pipeline*
"""


def create_dataset_card(config: dict, data_dir: Path) -> tuple[str, DatasetCardData]:
    """Create dataset card content and metadata."""
    readme_content = generate_readme_content(config)

    card_data = DatasetCardData(
        language=config["language"],
        multilinguality="monolingual",
        size_categories=config["size_category"],
        source_datasets=[config["original_repo"]],
        pretty_name=config["pretty_name"],
        tags=["legal", "polish", "enriched", "gemini", "factual-state", "legal-state"],
        configs=[
            {
                "config_name": "default",
                "data_files": [
                    {
                        "split": "train",
                        "path": f"{DEFAULT_DATA_DIR_IN_REPO}/{DATA_SHARD_FILE_PATTERN}",
                    }
                ],
            }
        ],
    )

    full_content = f"---\n{card_data}\n---\n\n{readme_content}"
    return full_content, card_data


def prepare_commit_operations(
    repo_id: str,
    data_dir: Path,
    readme_content: str,
    api: HfApi,
) -> list:
    """Prepare commit operations for uploading dataset."""
    operations = []

    # Check if repo exists and get existing files
    try:
        existing_files = list(api.list_repo_files(repo_id, repo_type="dataset"))

        # Delete old data files
        for f_name in existing_files:
            if f_name.startswith(f"{DEFAULT_DATA_DIR_IN_REPO}/"):
                operations.append(CommitOperationDelete(path_in_repo=f_name))

        # Delete old README
        if "README.md" in existing_files:
            operations.append(CommitOperationDelete(path_in_repo="README.md"))

    except RepositoryNotFoundError:
        logger.info(f"Repository {repo_id} is new, no files to delete")

    # Add new parquet files
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    for file_path in sorted(parquet_files):
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"{DEFAULT_DATA_DIR_IN_REPO}/{file_path.name}",
                path_or_fileobj=file_path,
            )
        )

    # Add README
    operations.append(
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=readme_content.encode("utf-8"),
        )
    )

    return operations


def push_dataset(
    dataset_name: str,
    config: dict,
    commit_message: str,
    create_if_missing: bool = True,
    dry_run: bool = False,
) -> bool:
    """Push a single enriched dataset to HuggingFace Hub."""
    data_dir = config["data_dir"]
    repo_id = config["hf_repo"]

    console.print(f"\n[bold blue]Processing: {dataset_name}[/bold blue]")
    console.print(f"  Data dir: {data_dir}")
    console.print(f"  HF repo: {repo_id}")

    # Check data directory exists
    if not data_dir.exists():
        console.print(f"[red]  Error: Data directory does not exist: {data_dir}[/red]")
        return False

    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        console.print(f"[red]  Error: No parquet files found in {data_dir}[/red]")
        return False

    console.print(f"  Found {len(parquet_files)} parquet files")

    api = HfApi()

    # Check/create repo
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        console.print(f"  [green]Repository exists[/green]")
    except RepositoryNotFoundError:
        if create_if_missing:
            console.print(f"  [yellow]Creating new repository: {repo_id}[/yellow]")
            if not dry_run:
                create_repo(repo_id=repo_id, repo_type="dataset", private=False)
        else:
            console.print(f"[red]  Error: Repository does not exist and create_if_missing=False[/red]")
            return False

    # Generate README
    readme_content, _ = create_dataset_card(config, data_dir)

    # Prepare operations
    operations = prepare_commit_operations(repo_id, data_dir, readme_content, api)

    # Show operations
    operations_table = [(op.path_in_repo, type(op).__name__) for op in operations]
    console.print("\n  Operations:")
    for path, op_type in operations_table[:10]:
        console.print(f"    {op_type}: {path}")
    if len(operations_table) > 10:
        console.print(f"    ... and {len(operations_table) - 10} more")

    if dry_run:
        console.print("  [yellow]DRY RUN - no changes made[/yellow]")
        return True

    # Execute
    console.print(f"  [bold]Pushing to {repo_id}...[/bold]")
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=commit_message,
    )
    console.print(f"  [green]Successfully pushed to {repo_id}[/green]")

    return True


@app.command()
def push(
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Dataset to push (pl-court-raw-enriched, pl-eureka-raw-enriched, pl-nsa-enriched)",
    ),
    all_datasets: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Push all enriched datasets",
    ),
    commit_message: str = typer.Option(
        "Update enriched dataset with factual_state and legal_state fields",
        "--message",
        "-m",
        help="Commit message",
    ),
    create_if_missing: bool = typer.Option(
        True,
        "--create/--no-create",
        help="Create repository if it doesn't exist",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without making changes",
    ),
    skip_confirmation: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
):
    """Push enriched datasets to HuggingFace Hub."""
    if not dataset and not all_datasets:
        console.print("[red]Error: Specify --dataset or --all[/red]")
        console.print("\nAvailable datasets:")
        for name in ENRICHED_DATASETS:
            console.print(f"  - {name}")
        raise typer.Exit(1)

    datasets_to_push = list(ENRICHED_DATASETS.keys()) if all_datasets else [dataset]

    # Validate datasets
    for ds_name in datasets_to_push:
        if ds_name not in ENRICHED_DATASETS:
            console.print(f"[red]Error: Unknown dataset '{ds_name}'[/red]")
            raise typer.Exit(1)

    # Show summary
    console.print("[bold]Datasets to push:[/bold]")
    table = Table()
    table.add_column("Dataset")
    table.add_column("HF Repo")
    table.add_column("Data Dir")
    table.add_column("Status")

    for ds_name in datasets_to_push:
        config = ENRICHED_DATASETS[ds_name]
        exists = config["data_dir"].exists()
        has_files = len(list(config["data_dir"].glob("*.parquet"))) > 0 if exists else False
        status = "[green]Ready[/green]" if has_files else "[red]No data[/red]"
        table.add_row(ds_name, config["hf_repo"], str(config["data_dir"]), status)

    console.print(table)

    if not skip_confirmation and not dry_run:
        if not typer.confirm("\nProceed with push?"):
            console.print("Operation cancelled.")
            raise typer.Abort()

    # Push datasets
    results = []
    for ds_name in datasets_to_push:
        success = push_dataset(
            dataset_name=ds_name,
            config=ENRICHED_DATASETS[ds_name],
            commit_message=commit_message,
            create_if_missing=create_if_missing,
            dry_run=dry_run,
        )
        results.append((ds_name, success))

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    for ds_name, success in results:
        status = "[green]Success[/green]" if success else "[red]Failed[/red]"
        console.print(f"  {ds_name}: {status}")


@app.command()
def list_datasets():
    """List all enriched datasets and their status."""
    console.print("[bold]Enriched Datasets:[/bold]\n")

    table = Table()
    table.add_column("Name")
    table.add_column("HF Repo")
    table.add_column("Original")
    table.add_column("Data Dir Exists")
    table.add_column("Parquet Files")

    for name, config in ENRICHED_DATASETS.items():
        exists = config["data_dir"].exists()
        file_count = len(list(config["data_dir"].glob("*.parquet"))) if exists else 0

        table.add_row(
            name,
            config["hf_repo"],
            config["original_repo"],
            "[green]Yes[/green]" if exists else "[red]No[/red]",
            str(file_count) if file_count > 0 else "[red]0[/red]",
        )

    console.print(table)


if __name__ == "__main__":
    app()
