"""
Command-line interface for dataset management and ingestion.
"""

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

from juddges.data.config import IngestConfig
from juddges.data.dataset_registry import DatasetConfig, get_registry
from juddges.data.universal_processor import UniversalDatasetProcessor
from juddges.data.validators import DatasetValidator, ValidationLevel

app = typer.Typer(help="Dataset management and ingestion CLI")
console = Console()

# Initialize components
registry = get_registry()
processor = UniversalDatasetProcessor(registry=registry)
validator = DatasetValidator()


@app.command()
def list_datasets():
    """List all registered datasets."""
    datasets = registry.list_datasets()

    if not datasets:
        console.print("[yellow]No datasets registered yet.[/yellow]")
        console.print("Use 'add' command to register a new dataset.")
        return

    table = Table(title="Registered Datasets")
    table.add_column("Dataset Name", style="cyan")
    table.add_column("Document Type", style="green")
    table.add_column("Text Fields", style="blue")
    table.add_column("Required Fields", style="red")

    for dataset_name in datasets:
        config = registry.get_config(dataset_name)
        if config:
            table.add_row(
                dataset_name,
                config.document_type,
                ", ".join(config.text_fields[:3]) + ("..." if len(config.text_fields) > 3 else ""),
                ", ".join(config.required_fields),
            )

    console.print(table)


@app.command()
def preview(
    dataset_name: str = typer.Argument(..., help="HuggingFace dataset name"),
    sample_size: int = typer.Option(10, "--samples", "-s", help="Number of samples to show"),
    force_reload: bool = typer.Option(False, "--force", "-f", help="Force reload configuration"),
):
    """Preview a dataset and show suggested configuration."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Loading preview for {dataset_name}...", total=None)

        try:
            preview_result = processor.preview_dataset(
                dataset_name, sample_size=sample_size, force_reload=force_reload
            )
        except Exception as e:
            console.print(f"[red]Error loading dataset: {e}[/red]")
            raise typer.Exit(1)

        progress.remove_task(task)

    # Display dataset info
    info_panel = Panel(
        f"Dataset: [cyan]{preview_result.dataset_name}[/cyan]\n"
        f"Total rows: [green]{preview_result.total_rows:,}[/green]\n"
        f"Columns: [blue]{len(preview_result.columns)}[/blue]\n"
        f"Est. processing time: [yellow]{preview_result.estimated_processing_time / 60:.1f} minutes[/yellow]",
        title="Dataset Information",
    )
    console.print(info_panel)

    # Show sample data
    if preview_result.sample_rows:
        table = Table(title="Sample Data")

        # Add columns (limit to first 6 for readability)
        display_columns = preview_result.columns[:6]
        for col in display_columns:
            table.add_column(col, max_width=20)

        # Add rows (limit to first 5)
        for row in preview_result.sample_rows[:5]:
            table.add_row(
                *[
                    str(row.get(col, ""))[:50] + ("..." if len(str(row.get(col, ""))) > 50 else "")
                    for col in display_columns
                ]
            )

        console.print(table)

    # Show suggested mapping
    if preview_result.suggested_mapping:
        mapping_table = Table(title="Suggested Column Mapping")
        mapping_table.add_column("Source Column", style="cyan")
        mapping_table.add_column("Target Field", style="green")

        for source, target in preview_result.suggested_mapping.items():
            mapping_table.add_row(source, target)

        console.print(mapping_table)

    # Show schema compatibility
    if preview_result.schema_compatibility.get("compatible", True):
        console.print("[green]✓ Schema is compatible with Weaviate[/green]")
    else:
        console.print("[red]⚠ Schema compatibility issues detected[/red]")
        if "error" in preview_result.schema_compatibility:
            console.print(f"Error: {preview_result.schema_compatibility['error']}")


@app.command()
def add(
    dataset_name: str = typer.Argument(..., help="HuggingFace dataset name"),
    mapping: Optional[str] = typer.Option(
        None, "--mapping", "-m", help="Column mapping as 'source=target,source2=target2'"
    ),
    text_fields: Optional[str] = typer.Option(
        None, "--text-fields", "-t", help="Comma-separated text fields"
    ),
    required_fields: Optional[str] = typer.Option(
        None, "--required", "-r", help="Comma-separated required fields"
    ),
    document_type: str = typer.Option("judgment", "--doc-type", "-d", help="Document type"),
    auto: bool = typer.Option(
        False, "--auto", "-a", help="Auto-generate configuration from preview"
    ),
):
    """Add a new dataset configuration."""

    # Check if dataset already exists
    if registry.get_config(dataset_name) and not auto:
        if not Confirm.ask(f"Dataset {dataset_name} already registered. Overwrite?"):
            raise typer.Exit(0)

    if auto:
        # Auto-generate configuration
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating automatic configuration...", total=None)

            try:
                config = processor.register_new_dataset(dataset_name)
                progress.remove_task(task)

                console.print(f"[green]✓ Auto-registered dataset: {dataset_name}[/green]")
                _display_config(config)

            except Exception as e:
                console.print(f"[red]Error auto-registering dataset: {e}[/red]")
                raise typer.Exit(1)

    else:
        # Manual configuration
        config_overrides = {}

        if mapping:
            try:
                mapping_dict = {}
                for pair in mapping.split(","):
                    source, target = pair.split("=")
                    mapping_dict[source.strip()] = target.strip()
                config_overrides["column_mapping"] = mapping_dict
            except ValueError:
                console.print(
                    "[red]Invalid mapping format. Use 'source=target,source2=target2'[/red]"
                )
                raise typer.Exit(1)

        if text_fields:
            config_overrides["text_fields"] = [f.strip() for f in text_fields.split(",")]

        if required_fields:
            config_overrides["required_fields"] = [f.strip() for f in required_fields.split(",")]

        config_overrides["document_type"] = document_type

        if config_overrides:
            try:
                config = processor.register_new_dataset(dataset_name, config_overrides)
                console.print(f"[green]✓ Registered dataset: {dataset_name}[/green]")
                _display_config(config)
            except Exception as e:
                console.print(f"[red]Error registering dataset: {e}[/red]")
                raise typer.Exit(1)
        else:
            console.print(
                "[yellow]No configuration provided. Use --auto for automatic configuration.[/yellow]"
            )


@app.command()
def validate(
    dataset_name: str = typer.Argument(..., help="Dataset name to validate"),
    sample_size: int = typer.Option(1000, "--sample-size", "-s", help="Sample size for validation"),
):
    """Validate a dataset before ingestion."""

    config = registry.get_config(dataset_name)
    if not config:
        console.print(f"[red]Dataset {dataset_name} not registered. Use 'add' command first.[/red]")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Validating {dataset_name}...", total=None)

        try:
            validation_result = validator.validate_dataset(dataset_name, config)
            progress.remove_task(task)
        except Exception as e:
            console.print(f"[red]Validation failed: {e}[/red]")
            raise typer.Exit(1)

    # Display validation results
    _display_validation_results(validation_result)


@app.command()
def ingest(
    dataset_name: str = typer.Argument(..., help="Dataset name to ingest"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size for ingestion"),
    skip_validation: bool = typer.Option(False, "--skip-validation", help="Skip validation step"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview only, don't actually ingest"),
):
    """Ingest a dataset to Weaviate."""

    config = registry.get_config(dataset_name)
    if not config:
        console.print(f"[red]Dataset {dataset_name} not registered. Use 'add' command first.[/red]")
        raise typer.Exit(1)

    # Validation step
    if not skip_validation:
        console.print("🔍 Running validation...")
        try:
            validation_result = validator.validate_dataset(dataset_name, config)

            if validation_result.has_critical_issues():
                console.print("[red]❌ Critical validation issues found. Cannot proceed.[/red]")
                _display_validation_results(validation_result)
                raise typer.Exit(1)

            if validation_result.get_issues_by_level(ValidationLevel.ERROR):
                console.print("[yellow]⚠ Validation errors found.[/yellow]")
                _display_validation_results(validation_result)
                if not Confirm.ask("Continue with ingestion despite errors?"):
                    raise typer.Exit(0)

            console.print("[green]✓ Validation passed[/green]")

        except Exception as e:
            console.print(f"[red]Validation failed: {e}[/red]")
            if not Confirm.ask("Continue without validation?"):
                raise typer.Exit(1)

    # Prepare ingestion config
    ingest_config = IngestConfig(batch_size=batch_size)

    # Start ingestion
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        if dry_run:
            task = progress.add_task("Running dry run...", total=None)
            try:
                result = processor.process_dataset(
                    dataset_name, config=config, ingest_config=ingest_config, preview_only=True
                )
                progress.remove_task(task)
                console.print("[green]✓ Dry run completed successfully[/green]")
            except Exception as e:
                console.print(f"[red]Dry run failed: {e}[/red]")
                raise typer.Exit(1)
        else:
            task = progress.add_task(f"Ingesting {dataset_name}...", total=None)
            try:
                result = processor.process_dataset(
                    dataset_name, config=config, ingest_config=ingest_config, create_embeddings=True
                )
                progress.remove_task(task)
            except Exception as e:
                console.print(f"[red]Ingestion failed: {e}[/red]")
                raise typer.Exit(1)

    # Display results
    _display_ingestion_results(result)


@app.command()
def show(dataset_name: str = typer.Argument(..., help="Dataset name to show")):
    """Show detailed configuration for a dataset."""

    config = registry.get_config(dataset_name)
    if not config:
        console.print(f"[red]Dataset {dataset_name} not registered.[/red]")
        raise typer.Exit(1)

    _display_config(config)


@app.command()
def remove(dataset_name: str = typer.Argument(..., help="Dataset name to remove")):
    """Remove a dataset configuration."""

    if not registry.get_config(dataset_name):
        console.print(f"[red]Dataset {dataset_name} not registered.[/red]")
        raise typer.Exit(1)

    if Confirm.ask(f"Remove configuration for {dataset_name}?"):
        registry.remove_dataset(dataset_name)
        console.print(f"[green]✓ Removed dataset: {dataset_name}[/green]")


@app.command()
def init():
    """Initialize registry with default configurations."""
    registry.create_default_configs()
    console.print("[green]✓ Initialized registry with default configurations[/green]")
    list_datasets()


def _display_config(config: DatasetConfig):
    """Display dataset configuration in a nice format."""
    config_panel = Panel(
        f"Name: [cyan]{config.name}[/cyan]\n"
        f"Document Type: [green]{config.document_type}[/green]\n"
        f"Required Fields: [red]{', '.join(config.required_fields)}[/red]\n"
        f"Text Fields: [blue]{', '.join(config.text_fields)}[/blue]\n"
        f"Date Fields: [yellow]{', '.join(config.date_fields)}[/yellow]\n"
        f"Array Fields: [magenta]{', '.join(config.array_fields)}[/magenta]",
        title="Dataset Configuration",
    )
    console.print(config_panel)

    if config.column_mapping:
        mapping_table = Table(title="Column Mapping")
        mapping_table.add_column("Source", style="cyan")
        mapping_table.add_column("Target", style="green")

        for source, target in config.column_mapping.items():
            mapping_table.add_row(source, target)

        console.print(mapping_table)


def _display_validation_results(result):
    """Display validation results in a nice format."""
    summary = result.get_summary()

    # Summary panel
    status_color = "green" if result.validation_passed else "red"
    status_text = "PASSED" if result.validation_passed else "FAILED"

    summary_panel = Panel(
        f"Status: [{status_color}]{status_text}[/{status_color}]\n"
        f"Total Rows: [cyan]{result.total_rows:,}[/cyan]\n"
        f"Critical: [red]{summary['critical']}[/red] | "
        f"Errors: [red]{summary['error']}[/red] | "
        f"Warnings: [yellow]{summary['warning']}[/yellow] | "
        f"Info: [blue]{summary['info']}[/blue]",
        title="Validation Summary",
    )
    console.print(summary_panel)

    # Show issues if any
    critical_issues = result.get_issues_by_level(ValidationLevel.CRITICAL)
    error_issues = result.get_issues_by_level(ValidationLevel.ERROR)

    if critical_issues or error_issues:
        issues_table = Table(title="Issues")
        issues_table.add_column("Level", style="red")
        issues_table.add_column("Category")
        issues_table.add_column("Message")
        issues_table.add_column("Suggestion", style="blue")

        for issue in critical_issues + error_issues:
            issues_table.add_row(
                issue.level.value.upper(), issue.category, issue.message, issue.suggestion or ""
            )

        console.print(issues_table)

    # Resource estimates
    if result.resource_estimates:
        estimates = result.resource_estimates
        resource_panel = Panel(
            f"Processing Time: [yellow]{estimates.get('estimated_processing_time_minutes', 0):.1f} minutes[/yellow]\n"
            f"Memory Required: [blue]{estimates.get('estimated_memory_mb', 0):.1f} MB[/blue]\n"
            f"Storage Required: [green]{estimates.get('estimated_storage_mb', 0):.1f} MB[/green]\n"
            f"Recommended Batch Size: [cyan]{estimates.get('recommended_batch_size', 32)}[/cyan]",
            title="Resource Estimates",
        )
        console.print(resource_panel)


def _display_ingestion_results(result):
    """Display ingestion results."""
    status_color = "green" if result.success else "red"
    status_text = "SUCCESS" if result.success else "FAILED"

    results_panel = Panel(
        f"Status: [{status_color}]{status_text}[/{status_color}]\n"
        f"Dataset: [cyan]{result.dataset_name}[/cyan]\n"
        f"Total Rows: [blue]{result.total_rows:,}[/blue]\n"
        f"Processed: [green]{result.processed_rows:,}[/green]\n"
        f"Documents Ingested: [yellow]{result.ingested_documents:,}[/yellow]\n"
        f"Chunks Ingested: [magenta]{result.ingested_chunks:,}[/magenta]\n"
        f"Processing Time: [cyan]{result.processing_time_seconds:.2f} seconds[/cyan]",
        title="Ingestion Results",
    )
    console.print(results_panel)

    # Show errors if any
    if result.errors:
        console.print("[red]Errors:[/red]")
        for error in result.errors:
            console.print(f"  • {error}")

    # Show warnings if any
    if result.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"  • {warning}")


if __name__ == "__main__":
    app()
