#!/usr/bin/env python3
"""
Simple streaming ingestion script for legal documents.
Usage: python simple_ingest.py [config_file.yaml]
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from juddges.data.stream_ingester import StreamingIngester
from juddges.settings import ROOT_PATH

load_dotenv(ROOT_PATH / ".env", override=True)


class DatasetConfig(BaseModel):
    """Configuration for dataset-specific settings loaded from YAML."""

    name: str = Field(..., description="Dataset name/path")
    document_type: str = Field(default="judgment", description="Type of legal document")
    chunk_overlap: int = Field(default=200, ge=0, description="Chunk overlap in characters")
    max_chunk_size: int = Field(default=1000, gt=0, description="Maximum chunk size in characters")
    chunk_strategy: str = Field(default="recursive", description="Chunking strategy")
    column_mapping: Dict[str, str] = Field(default_factory=dict, description="Column name mapping")
    required_fields: List[str] = Field(default_factory=list, description="Required dataset fields")
    text_fields: List[str] = Field(default_factory=list, description="Text fields to process")
    date_fields: List[str] = Field(default_factory=list, description="Date fields")
    array_fields: List[str] = Field(default_factory=list, description="Array fields")
    json_fields: List[str] = Field(default_factory=list, description="JSON fields")
    default_values: Dict[str, Any] = Field(default_factory=dict, description="Default field values")
    embedding_path: Optional[str] = Field(None, description="Path for aggregate embeddings")
    chunks_path: Optional[str] = Field(None, description="Path for chunk embeddings")
    num_proc: int = Field(default=1, gt=0, description="Number of processes for dataset operations")
    batch_size: int = Field(default=100, gt=0, description="Batch size for dataset operations")
    embedding_models: Dict[str, str] = Field(
        default_factory=lambda: {
            "base": "sdadas/mmlw-roberta-large",
            "dev": "sentence-transformers/all-MiniLM-L6-v2", 
            "fast": "sentence-transformers/all-mpnet-base-v2"
        },
        description="Mapping of named vectors to embedding models"
    )
    
    @field_validator("embedding_models")
    @classmethod
    def validate_embedding_models(cls, v):
        """Validate that embedding models are properly configured."""
        if not v:
            raise ValueError("embedding_models cannot be empty - at least one model is required")
        
        # Check for required vector names
        required_vectors = {"base", "dev", "fast"}
        missing_vectors = required_vectors - set(v.keys())
        if missing_vectors:
            raise ValueError(f"Missing required vector names: {missing_vectors}")
        
        # Validate that we have actual model names
        for vector_name, model_name in v.items():
            if not model_name or not isinstance(model_name, str):
                raise ValueError(f"Invalid model name for vector '{vector_name}': {model_name}")
        
        # Check for duplicates and warn if found
        unique_models = set(v.values())
        if len(unique_models) < len(v):
            # Note: Using print instead of logger since this is in a validator
            print("Warning: Some embedding models are duplicated - consider using different models for better performance")
        
        return v


class IngestionConfig(BaseModel):
    """Configuration for the streaming ingestion process."""

    # Dataset configuration
    dataset_config: DatasetConfig = Field(..., description="Dataset-specific configuration")

    # Weaviate settings
    weaviate_url: str = Field(default="http://localhost:8084", description="Weaviate instance URL")

    # Embedding settings
    embedding_model: str = Field(
        default="sdadas/mmlw-roberta-large", description="Sentence transformer model name"
    )
    embedding_batch_size: int = Field(
        default=32, gt=0, description="Embedding generation batch size"
    )

    # Processing settings
    streaming: bool = Field(default=True, description="Use streaming mode for dataset loading")
    tracker_db: str = Field(
        default="processed_documents.db", description="SQLite tracker database path"
    )
    reset_tracker: bool = Field(
        default=False, description="Reset tracker database before processing"
    )

    # System settings
    log_level: str = Field(default="INFO", description="Logging verbosity level")

    @property
    def dataset_path(self) -> str:
        """Get dataset path from dataset config."""
        return self.dataset_config.name

    @property
    def chunk_size(self) -> int:
        """Get chunk size from dataset config."""
        return self.dataset_config.max_chunk_size

    @property
    def overlap(self) -> int:
        """Get overlap from dataset config."""
        return self.dataset_config.chunk_overlap

    @property
    def batch_size(self) -> int:
        """Get batch size from embedding settings."""
        return self.embedding_batch_size

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level is one of the accepted values."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @field_validator("dataset_config")
    @classmethod
    def validate_dataset_config(cls, v):
        """Validate dataset configuration."""
        if v.chunk_overlap >= v.max_chunk_size:
            raise ValueError("Chunk overlap must be less than max chunk size")
        return v

    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(cls, v):
        """Validate embedding model is one of the supported models."""
        supported_models = {
            "sdadas/mmlw-roberta-large",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        }
        if v not in supported_models:
            # Allow custom models but warn
            pass
        return v

    def to_display_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display purposes."""
        display_dict = {
            "dataset_name": self.dataset_config.name,
            "document_type": self.dataset_config.document_type,
            "weaviate_url": self.weaviate_url,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "embedding_batch_size": self.embedding_batch_size,
            "tracker_db": self.tracker_db,
            "streaming": self.streaming,
            "reset_tracker": self.reset_tracker,
            "log_level": self.log_level,
        }
        
        # Add embedding models if available
        if hasattr(self.dataset_config, 'embedding_models') and self.dataset_config.embedding_models:
            for vector_name, model in self.dataset_config.embedding_models.items():
                display_dict[f"embedding_model_{vector_name}"] = model
        
        return display_dict

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


def create_help_panel() -> Panel:
    """Create a help panel with examples."""
    help_text = """[bold blue]Examples:[/bold blue]

[dim]# Interactive mode (no arguments)[/dim]
[green]python simple_ingest.py[/green]

[dim]# With specific config file[/dim]
[green]python simple_ingest.py configs/datasets/JuDDGES_pl-court-raw-sample.yaml[/green]

[dim]# Using relative path[/dim]
[green]python simple_ingest.py JuDDGES_pl-court-raw-sample.yaml[/green]

[bold yellow]Environment Variables:[/bold yellow]
[dim]# Set Weaviate API key (if authentication is required)[/dim]
[green]export WEAVIATE_API_KEY=your-api-key-here[/green]
[dim]# Alternative variable name[/dim]
[green]export WV_API_KEY=your-api-key-here[/green]

[bold cyan]Quick Setup:[/bold cyan]
[dim]# 1. Start Weaviate[/dim]
[green]cd weaviate && docker compose up -d[/green]
[dim]# 2. Set API key (check your Weaviate .env file)[/dim]
[green]export WEAVIATE_API_KEY=$(grep AUTHENTICATION_APIKEY_ALLOWED_KEYS weaviate/.env | cut -d= -f2)[/green]
[dim]# 3. Run ingestion[/dim]
[green]python simple_ingest.py[/green]

[bold magenta]Config File Structure:[/bold magenta]
[dim]Dataset configs are YAML files in configs/datasets/ with dataset-specific settings like:[/dim]
[green]- Dataset name and type[/green]
[green]- Column mappings[/green] 
[green]- Chunk size and overlap[/green]
[green]- Required and text fields[/green]"""

    return Panel(help_text, title="Usage Examples", border_style="blue")


def load_dataset_config(config_path: Path) -> DatasetConfig:
    """Load dataset configuration from YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return DatasetConfig(**config_data)
    except Exception as e:
        raise ValueError(f"Failed to load dataset config from {config_path}: {e}")


def find_available_configs() -> List[Path]:
    """Find all available dataset config files."""
    configs_dir = ROOT_PATH / "configs" / "datasets"
    if not configs_dir.exists():
        return []

    return sorted(configs_dir.glob("*.yaml"))


def select_dataset_config(console: Console) -> Optional[DatasetConfig]:
    """Let user select a dataset configuration."""
    available_configs = find_available_configs()

    if not available_configs:
        console.print("[bold red]No dataset configs found in configs/datasets/[/bold red]")
        return None

    console.print("\n[bold cyan]Available Dataset Configurations:[/bold cyan]")
    for i, config_path in enumerate(available_configs, 1):
        # Extract readable name from filename
        name = config_path.stem.replace("_", "/").replace("-", "-")
        console.print(f"  {i}. {name} ({config_path.name})")

    while True:
        try:
            choice = Prompt.ask(
                "[bold cyan]Select dataset config[/bold cyan]",
                choices=[str(i) for i in range(1, len(available_configs) + 1)],
                default="1",
            )
            selected_config = available_configs[int(choice) - 1]
            return load_dataset_config(selected_config)
        except (ValueError, IndexError) as e:
            console.print(f"[red]Invalid selection: {e}[/red]")


def get_configuration(config_file: Optional[Path] = None) -> IngestionConfig:
    """Get configuration through Rich prompts or config file."""
    console = Console()

    console.print(
        Panel.fit(
            "[bold blue]🚀 Streaming Legal Document Ingester[/bold blue]\n\n"
            "This tool processes legal documents with embeddings and ingests them into Weaviate.\n"
            "Configure your settings below or use defaults.",
            title="Welcome",
            border_style="blue",
        )
    )

    # Load dataset configuration
    if config_file:
        console.print(f"[bold green]Loading config from: {config_file}[/bold green]")
        try:
            dataset_config = load_dataset_config(config_file)
            console.print(f"[green]✓ Loaded dataset config for: {dataset_config.name}[/green]")
        except Exception as e:
            console.print(f"[bold red]Failed to load config: {e}[/bold red]")
            raise
    else:
        # Interactive dataset selection
        dataset_config = select_dataset_config(console)
        if not dataset_config:
            raise ValueError("No dataset configuration selected")

    config_dict: Dict[str, Any] = {"dataset_config": dataset_config}

    # Weaviate settings (only prompt if interactive mode)
    if not config_file:
        config_dict["weaviate_url"] = Prompt.ask(
            "[bold cyan]Weaviate URL[/bold cyan]", default="http://localhost:8084"
        )

        # Test Weaviate connection
        test_connection = Confirm.ask(
            "[bold cyan]Test Weaviate connection now?[/bold cyan]", default=False
        )
    else:
        # Use defaults when loading from file
        config_dict["weaviate_url"] = "http://localhost:8084"
        test_connection = False

    if test_connection:
        console.print("[dim]Testing connection...[/dim]")
        try:
            import os
            import weaviate

            url_parts = config_dict["weaviate_url"].split("://")[-1].split(":")
            host = url_parts[0]
            port = int(url_parts[1]) if len(url_parts) > 1 else 8080

            api_key = os.getenv("WEAVIATE_API_KEY") or os.getenv("WV_API_KEY")

            if api_key:
                import weaviate.auth as wv_auth

                test_client = weaviate.connect_to_local(
                    host=host, port=port, auth_credentials=wv_auth.AuthApiKey(api_key)
                )
            else:
                test_client = weaviate.connect_to_local(host=host, port=port)

            # Test the connection by getting meta info
            _ = test_client.get_meta()
            test_client.close()
            console.print("[green]✓ Connection successful![/green]")

        except Exception as e:
            if "401" in str(e) or "anonymous access not enabled" in str(e):
                console.print(
                    "[red]✗ Authentication required - make sure to set WEAVIATE_API_KEY[/red]"
                )
            else:
                console.print(f"[red]✗ Connection failed: {e}[/red]")

            if not Confirm.ask("[yellow]Continue anyway?[/yellow]", default=False):
                console.print("[bold red]Cancelled by user[/bold red]")
                raise KeyboardInterrupt("User cancelled configuration")

    # Embedding model selection
    if not config_file:
        models = [
            "sdadas/mmlw-roberta-large",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ]

        console.print("\n[bold cyan]Available embedding models:[/bold cyan]")
        for i, model in enumerate(models, 1):
            console.print(f"  {i}. {model}")

        model_choice = Prompt.ask(
            "[bold cyan]Select embedding model[/bold cyan]", choices=["1", "2", "3"], default="1"
        )
        config_dict["embedding_model"] = models[int(model_choice) - 1]
    else:
        # Use default embedding model when loading from file
        config_dict["embedding_model"] = "sdadas/mmlw-roberta-large"

    # Advanced settings (only prompt if interactive mode)
    if not config_file:
        show_advanced = Confirm.ask("[bold cyan]Show advanced settings?[/bold cyan]", default=False)

        if show_advanced:
            config_dict["embedding_batch_size"] = int(
                Prompt.ask("[bold cyan]Embedding batch size[/bold cyan]", default="32")
            )

            config_dict["tracker_db"] = Prompt.ask(
                "[bold cyan]Tracker database path[/bold cyan]", default="processed_documents.db"
            )

            config_dict["streaming"] = Confirm.ask(
                "[bold cyan]Use streaming mode?[/bold cyan]", default=True
            )
        else:
            # Use defaults in interactive mode
            config_dict.update(
                {
                    "embedding_batch_size": 32,
                    "tracker_db": "processed_documents.db",
                    "streaming": True,
                }
            )
    else:
        # Use defaults when loading from file
        config_dict.update(
            {
                "embedding_batch_size": 32,
                "tracker_db": "processed_documents.db",
                "streaming": True,
            }
        )

    # Reset tracker and log level (only prompt if interactive mode)
    if not config_file:
        config_dict["reset_tracker"] = Confirm.ask(
            "[bold cyan]Reset tracker database?[/bold cyan]", default=False
        )

        config_dict["log_level"] = Prompt.ask(
            "[bold cyan]Log level[/bold cyan]",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
        )
    else:
        # Use defaults when loading from file
        config_dict["reset_tracker"] = False
        config_dict["log_level"] = "INFO"

    try:
        return IngestionConfig(**config_dict)
    except Exception as e:
        console.print(f"[bold red]Configuration error: {e}[/bold red]")
        raise


def display_configuration(config: IngestionConfig) -> None:
    """Display configuration in a nice table."""
    console = Console()

    table = Table(title="Configuration Settings", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan", width=20)
    table.add_column("Value", style="green")
    table.add_column("Description", style="dim")

    descriptions = {
        "dataset_name": "HuggingFace dataset name",
        "document_type": "Type of legal documents",
        "weaviate_url": "Weaviate instance URL",
        "embedding_model": "Primary sentence transformer model",
        "embedding_model_base": "Base vector embedding model",
        "embedding_model_dev": "Dev vector embedding model", 
        "embedding_model_fast": "Fast vector embedding model",
        "chunk_size": "Text chunk size (from dataset config)",
        "overlap": "Chunk overlap (from dataset config)",
        "embedding_batch_size": "Embedding generation batch size",
        "tracker_db": "SQLite tracker database",
        "streaming": "Use streaming mode",
        "reset_tracker": "Reset tracker on start",
        "log_level": "Logging verbosity",
    }

    for key, value in config.to_display_dict().items():
        table.add_row(key.replace("_", " ").title(), str(value), descriptions.get(key, ""))

    console.print(table)


def display_final_stats(stats, tracker_stats: Dict[str, int]) -> None:
    """Display final statistics with Rich formatting."""
    console = Console()

    # Main statistics table
    main_table = Table(title="📊 Processing Results", show_header=True, header_style="bold blue")
    main_table.add_column("Metric", style="cyan")
    main_table.add_column("Count", style="green", justify="right")
    main_table.add_column("Rate", style="yellow", justify="right")

    processing_time = max(stats.processing_time, 1)

    main_table.add_row(
        "Total Documents",
        str(stats.total_documents),
        f"{stats.total_documents / processing_time:.1f}/sec",
    )
    main_table.add_row(
        "✅ Processed",
        str(stats.processed_documents),
        f"{stats.processed_documents / processing_time:.1f}/sec",
    )
    main_table.add_row("⏭️ Skipped", str(stats.skipped_documents), "-")
    main_table.add_row("❌ Failed", str(stats.failed_documents), "-")
    main_table.add_row(
        "📄 Total Chunks",
        str(stats.total_chunks),
        f"{stats.total_chunks / processing_time:.1f}/sec",
    )
    main_table.add_row("⏱️ Processing Time", f"{stats.processing_time:.1f}s", "-")

    # Tracker statistics table
    tracker_table = Table(
        title="💾 Tracker Database", show_header=True, header_style="bold magenta"
    )
    tracker_table.add_column("Status", style="cyan")
    tracker_table.add_column("Count", style="green", justify="right")
    tracker_table.add_column("Percentage", style="yellow", justify="right")

    total_tracked = max(tracker_stats["total"], 1)

    tracker_table.add_row("📝 Total Tracked", str(tracker_stats["total"]), "100%")
    tracker_table.add_row(
        "✅ Successful",
        str(tracker_stats["successful"]),
        f"{tracker_stats['successful'] / total_tracked * 100:.1f}%",
    )
    tracker_table.add_row(
        "❌ Failed",
        str(tracker_stats["failed"]),
        f"{tracker_stats['failed'] / total_tracked * 100:.1f}%",
    )

    # Display tables side by side
    console.print(Columns([main_table, tracker_table], equal=True))

    # Success message
    if stats.failed_documents == 0:
        console.print(
            Panel.fit(
                f"[bold green]🎉 Processing completed successfully![/bold green]\n\n"
                f"All {stats.processed_documents} documents processed without errors.\n"
                f"Generated {stats.total_chunks} searchable chunks.\n"
                f"Average processing rate: {stats.processed_documents / processing_time:.1f} docs/sec",
                title="Success",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel.fit(
                f"[bold yellow]⚠️ Processing completed with issues[/bold yellow]\n\n"
                f"Successfully processed: {stats.processed_documents} documents\n"
                f"Failed: {stats.failed_documents} documents\n"
                f"Check logs for error details.",
                title="Warning",
                border_style="yellow",
            )
        )


def check_environment() -> None:
    """Check environment variables and give helpful warnings."""
    import os

    console = Console()

    # Check for API key
    api_key = os.getenv("WEAVIATE_API_KEY") or os.getenv("WV_API_KEY")
    if not api_key:
        console.print(
            Panel.fit(
                "[bold yellow]⚠️ No Weaviate API key found[/bold yellow]\n\n"
                "If your Weaviate instance requires authentication, please set:\n"
                "[green]export WEAVIATE_API_KEY='your-api-key'[/green]\n\n"
                "If you're using a local Weaviate without authentication, you can ignore this warning.",
                title="Authentication Warning",
                border_style="yellow",
            )
        )
    else:
        console.print(f"[green]✓[/green] Found API key: {api_key[:8]}{'*' * (len(api_key) - 8)}")


def resolve_config_path(config_arg: str) -> Path:
    """Resolve config file path from command line argument."""
    config_path = Path(config_arg)

    # If absolute path or relative path that exists, use as-is
    if config_path.is_absolute() or config_path.exists():
        return config_path

    # Try configs/datasets/ directory
    configs_dir = ROOT_PATH / "configs" / "datasets"
    full_path = configs_dir / config_arg
    if full_path.exists():
        return full_path

    # Try with .yaml extension
    if not config_arg.endswith(".yaml"):
        yaml_path = configs_dir / f"{config_arg}.yaml"
        if yaml_path.exists():
            return yaml_path

    raise FileNotFoundError(f"Config file not found: {config_arg}")


def main():
    """Main CLI interface using Rich."""
    console = Console()

    # Check environment first
    check_environment()

    # Handle command line arguments
    config_file = None

    if len(sys.argv) > 1:
        # Check for help first
        if sys.argv[1] in ["--help", "-h", "--help-examples"]:
            console.print(create_help_panel())
            return

        # Assume first argument is config file
        try:
            config_file = resolve_config_path(sys.argv[1])
            console.print(f"[bold green]Using config file: {config_file}[/bold green]")
        except FileNotFoundError as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            console.print("[yellow]Available configs:[/yellow]")
            for config in find_available_configs():
                console.print(f"  - {config.name}")
            sys.exit(1)

    # Get configuration (interactive or from file)
    try:
        config = get_configuration(config_file)
    except KeyboardInterrupt:
        console.print("[bold red]Cancelled by user[/bold red]")
        return
    except Exception as e:
        console.print(f"[bold red]Configuration error: {e}[/bold red]")
        sys.exit(1)

    # Display configuration
    console.print("\n")
    display_configuration(config)

    # Confirm before starting (only in interactive mode)
    if not config_file:
        if not Confirm.ask("\n[bold cyan]Start processing with these settings?[/bold cyan]"):
            console.print("[bold red]Cancelled by user[/bold red]")
            return

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=config.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )

    try:
        # Initialize ingester with context manager
        console.print("\n[bold green]🚀 Initializing streaming ingester...[/bold green]")

        # Get embedding models from dataset config - ensure we always have multiple models
        embedding_models = config.dataset_config.embedding_models
        console.print(f"[bold cyan]Using embedding models:[/bold cyan] {list(embedding_models.keys())}")
        for vector_name, model_name in embedding_models.items():
            console.print(f"  {vector_name}: {model_name}")
        
        console.print("\n[bold yellow]📋 Initializing with dataset configuration validation...[/bold yellow]")
        
        with StreamingIngester(
            weaviate_url=config.weaviate_url,
            embedding_model=config.embedding_model,
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            batch_size=config.embedding_batch_size,
            tracker_db=config.tracker_db,
            dataset_config=config.dataset_config,
            embedding_models=embedding_models,
        ) as ingester:
            # Reset tracker if requested
            if config.reset_tracker:
                console.print("[bold yellow]🔄 Resetting tracker database...[/bold yellow]")
                ingester.reset_tracker()

            # Process dataset
            console.print("[bold green]📊 Starting dataset processing...[/bold green]")
            stats = ingester.process_dataset(
                dataset_path=config.dataset_path, streaming=config.streaming
            )

            # Get tracker stats
            tracker_stats = ingester.tracker.get_stats()

            # Display final statistics
            console.print("\n")
            display_final_stats(stats, tracker_stats)

    except KeyboardInterrupt:
        console.print("\n[bold red]⚠️ Process interrupted by user[/bold red]")
        sys.exit(1)
    except Exception as e:
        # Check for authentication error
        if "401" in str(e) or "anonymous access not enabled" in str(e):
            console.print(
                Panel.fit(
                    "[bold red]❌ Weaviate Authentication Failed[/bold red]\n\n"
                    "Your Weaviate instance requires authentication but no valid API key was found.\n\n"
                    "[bold cyan]Solutions:[/bold cyan]\n"
                    "1. Set the API key: [green]export WEAVIATE_API_KEY='your-api-key'[/green]\n"
                    "2. Check your Weaviate .env file for the correct API key\n"
                    "3. Disable authentication in Weaviate if running locally\n\n"
                    f"Weaviate URL: {config.weaviate_url}\n"
                    f"Error: {e}",
                    title="Authentication Error",
                    border_style="red",
                )
            )
        else:
            console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
            logger.exception("Ingestion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
