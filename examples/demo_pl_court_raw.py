#!/usr/bin/env python3
"""
Live demonstration of Universal Dataset Ingestion with JuDDGES/pl-court-raw.

This script shows the complete workflow from preview to ingestion
for the Polish court judgments dataset.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from juddges.data.config import IngestConfig
from juddges.data.dataset_registry import get_registry
from juddges.data.universal_processor import UniversalDatasetProcessor
from juddges.data.validators import DatasetValidator, ValidationLevel


def main():
    """Demonstrate ingestion with JuDDGES/pl-court-raw dataset."""

    console = Console()
    dataset_name = "JuDDGES/pl-court-raw"

    # Main header
    console.print(
        Panel.fit(
            "🏛️ JuDDGES Polish Court Dataset Ingestion Demo",
            style="bold blue",
            border_style="bright_blue",
        )
    )

    console.print(f"\n🎯 [bold]Target Dataset:[/bold] [cyan]{dataset_name}[/cyan]")
    console.print("📋 [bold]Demo Steps:[/bold]")

    steps_tree = Tree("🚀 Universal Ingestion Workflow")
    steps_tree.add("📋 Preview dataset structure")
    steps_tree.add("⚙️ Auto-generate configuration")
    steps_tree.add("📝 Register dataset")
    steps_tree.add("🔍 Comprehensive validation")
    steps_tree.add("🔧 Schema compatibility check")
    steps_tree.add("🏃 Dry run processing")
    steps_tree.add("💻 CLI commands demo")
    steps_tree.add("✨ System improvements showcase")

    console.print(steps_tree)

    # Initialize components
    console.print("\n⚡ [bold yellow]Initializing Universal Ingestion System...[/bold yellow]")
    registry = get_registry()
    processor = UniversalDatasetProcessor(registry=registry)
    validator = DatasetValidator()

    try:
        # Step 1: Preview the dataset structure
        console.print(
            "\n📋 [bold green]Step 1: Previewing JuDDGES/pl-court-raw structure...[/bold green]"
        )

        with console.status("[bold green]Loading dataset preview...", spinner="dots"):
            preview = processor.preview_dataset(dataset_name, sample_size=3)

        # Dataset Information Table
        info_table = Table(title="📊 Dataset Information", show_header=False, box=None)
        info_table.add_column("Property", style="bold cyan")
        info_table.add_column("Value", style="white")

        info_table.add_row("📂 Dataset", f"[bold]{preview.dataset_name}[/bold]")
        info_table.add_row("📊 Total rows", f"[yellow]{preview.total_rows:,}[/yellow]")
        info_table.add_row("📋 Columns", f"[yellow]{len(preview.columns)}[/yellow]")
        info_table.add_row(
            "⏱️ Est. processing time",
            f"[yellow]{preview.estimated_processing_time / 60:.1f} minutes[/yellow]",
        )

        console.print(Panel(info_table, border_style="bright_blue"))

        # Dataset Columns
        columns_table = Table(
            title="📋 Dataset Columns", show_header=True, header_style="bold magenta"
        )
        columns_table.add_column("#", style="cyan", width=4)
        columns_table.add_column("Column Name", style="white")

        for i, col in enumerate(preview.columns, 1):
            columns_table.add_row(str(i), col)

        console.print(columns_table)

        # Sample Data
        if preview.sample_rows:
            sample = preview.sample_rows[0]
            sample_table = Table(
                title="📄 Sample Data (First Row)", show_header=True, header_style="bold magenta"
            )
            sample_table.add_column("Field", style="cyan", width=20)
            sample_table.add_column("Value", style="white", width=60)

            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    display_value = value[:100] + "..."
                else:
                    display_value = str(value)
                sample_table.add_row(key, display_value)

            console.print(sample_table)

        # Smart Column Mapping
        mapping_table = Table(
            title="🤖 Smart Column Mapping", show_header=True, header_style="bold magenta"
        )
        mapping_table.add_column("Source Field", style="cyan", width=25)
        mapping_table.add_column("→", style="bold", width=3)
        mapping_table.add_column("Target Field", style="green", width=25)

        for source, target in preview.suggested_mapping.items():
            mapping_table.add_row(source, "→", target)

        console.print(
            Panel(mapping_table, title="🧠 Automatic Field Detection", border_style="green")
        )

        # Step 2: Show the auto-generated configuration
        console.print("\n⚙️ [bold green]Step 2: Auto-generated configuration...[/bold green]")
        config = preview.suggested_config

        config_table = Table(title="⚙️ Auto-Generated Configuration", show_header=False, box=None)
        config_table.add_column("Setting", style="bold cyan", width=20)
        config_table.add_column("Value", style="white")

        config_table.add_row(
            "📄 Document type", f"[bold yellow]{config.document_type}[/bold yellow]"
        )
        config_table.add_row(
            "⭐ Required fields", f"[green]{', '.join(config.required_fields)}[/green]"
        )
        config_table.add_row("📝 Text fields", f"[blue]{', '.join(config.text_fields)}[/blue]")
        config_table.add_row(
            "📅 Date fields", f"[magenta]{', '.join(config.date_fields)}[/magenta]"
        )
        config_table.add_row(
            "📋 Array fields", f"[yellow]{', '.join(config.array_fields)}[/yellow]"
        )
        config_table.add_row("🔗 JSON fields", f"[cyan]{', '.join(config.json_fields)}[/cyan]")
        config_table.add_row("🎯 Default values", f"[white]{config.default_values}[/white]")

        console.print(Panel(config_table, border_style="bright_green"))

        # Step 3: Register the dataset
        console.print("\n📝 [bold green]Step 3: Registering dataset configuration...[/bold green]")

        existing_config = registry.get_config(dataset_name)
        if existing_config:
            console.print(
                f"✅ [green]Dataset already registered:[/green] [bold]{dataset_name}[/bold]"
            )
            config = existing_config
        else:
            config = processor.register_new_dataset(dataset_name)
            console.print(f"✅ [green]Successfully registered:[/green] [bold]{dataset_name}[/bold]")

        # Step 4: Comprehensive validation
        console.print("\n🔍 [bold green]Step 4: Running comprehensive validation...[/bold green]")

        with console.status("[bold green]Validating dataset...", spinner="dots"):
            validation_result = validator.validate_dataset(dataset_name, config)

        # Validation Results
        summary = validation_result.get_summary()
        status_color = "green" if validation_result.validation_passed else "red"
        status_icon = "✅" if validation_result.validation_passed else "❌"
        status_text = "PASSED" if validation_result.validation_passed else "FAILED"

        validation_table = Table(
            title=f"{status_icon} Validation Results", show_header=False, box=None
        )
        validation_table.add_column("Metric", style="bold cyan", width=20)
        validation_table.add_column("Value", style="white")

        validation_table.add_row("🏆 Status", f"[{status_color}]{status_text}[/{status_color}]")
        validation_table.add_row(
            "📊 Rows validated", f"[yellow]{validation_result.total_rows:,}[/yellow]"
        )
        validation_table.add_row("🚨 Critical issues", f"[red]{summary['critical']}[/red]")
        validation_table.add_row("❌ Errors", f"[red]{summary['error']}[/red]")
        validation_table.add_row("⚠️ Warnings", f"[yellow]{summary['warning']}[/yellow]")
        validation_table.add_row("ℹ️ Info messages", f"[blue]{summary['info']}[/blue]")

        console.print(Panel(validation_table, border_style=status_color))

        # Show any critical issues
        for level in [ValidationLevel.CRITICAL, ValidationLevel.ERROR]:
            issues = validation_result.get_issues_by_level(level)
            if issues:
                issue_text = Text()
                issue_text.append(f"\n{level.value.upper()} ISSUES:\n", style="bold red")
                for issue in issues[:3]:  # Show first 3
                    issue_text.append(f"• {issue.message}\n", style="red")
                    if issue.suggestion:
                        issue_text.append(f"  💡 Suggestion: {issue.suggestion}\n", style="yellow")
                console.print(Panel(issue_text, border_style="red"))

        # Resource estimates
        if validation_result.resource_estimates:
            estimates = validation_result.resource_estimates
            resource_table = Table(title="📊 Resource Estimates", show_header=False, box=None)
            resource_table.add_column("Resource", style="bold cyan", width=20)
            resource_table.add_column("Estimate", style="white")

            resource_table.add_row(
                "⏱️ Processing time",
                f"[yellow]{estimates.get('estimated_processing_time_minutes', 0):.1f} minutes[/yellow]",
            )
            resource_table.add_row(
                "💾 Memory required",
                f"[blue]{estimates.get('estimated_memory_mb', 0):.1f} MB[/blue]",
            )
            resource_table.add_row(
                "💿 Storage required",
                f"[green]{estimates.get('estimated_storage_mb', 0):.1f} MB[/green]",
            )
            resource_table.add_row(
                "📦 Batch size", f"[magenta]{estimates.get('recommended_batch_size', 32)}[/magenta]"
            )

            console.print(Panel(resource_table, border_style="bright_blue"))

        # Step 5: Schema compatibility check
        console.print(
            "\n🔧 [bold green]Step 5: Checking Weaviate schema compatibility...[/bold green]"
        )
        schema_compat = preview.schema_compatibility

        if schema_compat.get("compatible", True):
            compat_text = Text("✅ Schema is fully compatible with Weaviate", style="bold green")
            console.print(Panel(compat_text, title="🔧 Schema Compatibility", border_style="green"))
        else:
            issue_text = Text()
            issue_text.append("⚠️ Schema compatibility issues detected:\n", style="bold yellow")
            for issue in schema_compat.get("issues", []):
                issue_text.append(f"• {issue}\n", style="yellow")
            console.print(Panel(issue_text, title="⚠️ Schema Issues", border_style="yellow"))

        # Step 6: Dry run demonstration
        console.print(
            "\n🏃 [bold green]Step 6: Running dry run (preview processing)...[/bold green]"
        )

        ingest_config = IngestConfig(
            max_documents=10,  # Small limit for demo
            batch_size=8,
            upsert=True,
        )

        with console.status("[bold green]Running dry run...", spinner="dots"):
            dry_run_result = processor.process_dataset(
                dataset_name=dataset_name,
                config=config,
                ingest_config=ingest_config,
                preview_only=True,
            )

        # Dry run results
        dry_run_status = "✅" if dry_run_result.success else "❌"
        dry_run_color = "green" if dry_run_result.success else "red"

        dry_run_table = Table(
            title=f"{dry_run_status} Dry Run Results", show_header=False, box=None
        )
        dry_run_table.add_column("Metric", style="bold cyan", width=20)
        dry_run_table.add_column("Value", style="white")

        if dry_run_result.success:
            dry_run_table.add_row("🏆 Status", "[green]SUCCESS[/green]")
            dry_run_table.add_row(
                "📊 Would process", f"[yellow]{dry_run_result.total_rows:,} rows[/yellow]"
            )
            dry_run_table.add_row(
                "⏱️ Processing time",
                f"[blue]{dry_run_result.processing_time_seconds:.2f} seconds[/blue]",
            )

            if dry_run_result.warnings:
                warning_text = Text()
                warning_text.append("⚠️ Warnings:\n", style="bold yellow")
                for warning in dry_run_result.warnings:
                    warning_text.append(f"• {warning}\n", style="yellow")
                console.print(Panel(warning_text, border_style="yellow"))
        else:
            dry_run_table.add_row("🏆 Status", "[red]FAILED[/red]")
            error_text = Text()
            error_text.append("❌ Errors:\n", style="bold red")
            for error in dry_run_result.errors:
                error_text.append(f"• {error}\n", style="red")
            console.print(Panel(error_text, border_style="red"))

        console.print(Panel(dry_run_table, border_style=dry_run_color))

        # Step 7: CLI commands demonstration
        console.print("\n💻 [bold green]Step 7: CLI Commands for Manual Operation[/bold green]")

        cli_commands = [
            ("1️⃣ Preview dataset", f"python scripts/dataset_manager.py preview '{dataset_name}'"),
            ("2️⃣ Validate dataset", f"python scripts/dataset_manager.py validate '{dataset_name}'"),
            (
                "3️⃣ Dry run ingestion",
                f"python scripts/dataset_manager.py ingest '{dataset_name}' --max-docs 100 --dry-run",
            ),
            (
                "4️⃣ Small batch ingestion",
                f"python scripts/dataset_manager.py ingest '{dataset_name}' --max-docs 1000 --batch-size 32",
            ),
            ("5️⃣ Full ingestion", f"python scripts/dataset_manager.py ingest '{dataset_name}'"),
            (
                "6️⃣ Enhanced ingestion",
                f"python scripts/embed/universal_ingest_to_weaviate.py \\\n    dataset_name='{dataset_name}' \\\n    max_documents=1000 \\\n    preview_only=false",
            ),
        ]

        cli_table = Table(title="💻 CLI Commands", show_header=True, header_style="bold magenta")
        cli_table.add_column("Command", style="cyan", width=25)
        cli_table.add_column("Usage", style="white", width=60)

        for cmd_name, cmd_usage in cli_commands:
            cli_table.add_row(cmd_name, f"[green]{cmd_usage}[/green]")

        console.print(Panel(cli_table, border_style="bright_cyan"))

        # Step 8: Show improvements
        console.print("\n✨ [bold green]Step 8: System Improvements Showcase[/bold green]")

        improvements_text = Text()
        improvements_text.append("🚀 IMPROVEMENTS OVER OLD SYSTEM\n\n", style="bold green")

        improvements = [
            "✅ Automatic field mapping (no manual configuration needed)",
            "✅ Comprehensive validation before processing",
            "✅ Intelligent data type conversion",
            "✅ Error recovery and detailed reporting",
            "✅ Resource estimation and optimization",
            "✅ User-friendly CLI interface",
            "✅ Support for any HuggingFace dataset structure",
            "✅ Dynamic schema adaptation",
        ]

        for improvement in improvements:
            improvements_text.append(f"{improvement}\n", style="green")

        console.print(
            Panel(
                improvements_text,
                title="🎯 Universal Ingestion Benefits",
                border_style="bright_green",
            )
        )

        # Success message
        console.print("\n")
        success_panel = Panel.fit(
            "🎉 Demo completed successfully!\n\n"
            "The system is ready to ingest JuDDGES/pl-court-raw or any other legal dataset!",
            style="bold green",
            border_style="bright_green",
        )
        console.print(success_panel)

    except Exception as e:
        console.print(f"\n❌ [bold red]Demo failed:[/bold red] {e}")
        console.print(
            "⚠️ [yellow]This might be due to network connectivity or dataset access issues[/yellow]"
        )
        console.print(
            "💡 [blue]You can still try the CLI commands with any accessible dataset[/blue]"
        )

        error_panel = Panel(
            f"🔧 [bold]Troubleshooting:[/bold]\n\n"
            f"1. Check internet connection for HuggingFace dataset access\n"
            f"2. Verify dataset name: {dataset_name}\n"
            f"3. Try with a different dataset name\n"
            f"4. Use local datasets in data/datasets/ directory",
            title="❌ Error Recovery",
            border_style="red",
        )
        console.print(error_panel)


if __name__ == "__main__":
    main()
