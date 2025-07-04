#!/usr/bin/env python3
"""
Simple streaming ingestion script for legal documents.
Usage: python simple_ingest.py --dataset-path <path> [options]
"""

from pathlib import Path
import sys
from typing import Optional, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.columns import Columns
from rich.align import Align
from rich import print as rprint
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from juddges.data.stream_ingester import StreamingIngester


def create_help_panel() -> Panel:
    """Create a help panel with examples."""
    help_text = """[bold blue]Examples:[/bold blue]

[dim]# Basic usage[/dim]
[green]python simple_ingest.py --dataset-path JuDDGES/pl-court-raw[/green]

[dim]# With custom settings[/dim]
[green]python simple_ingest.py --dataset-path JuDDGES/pl-court-raw \\
    --weaviate-url http://localhost:8080 \\
    --embedding-model sdadas/mmlw-roberta-large \\
    --chunk-size 512 \\
    --batch-size 32[/green]

[dim]# Reset tracker and start fresh[/dim]
[green]python simple_ingest.py --dataset-path JuDDGES/pl-court-raw --reset-tracker[/green]

[dim]# Non-streaming mode (load full dataset)[/dim]
[green]python simple_ingest.py --dataset-path JuDDGES/pl-court-raw --no-streaming[/green]"""
    
    return Panel(help_text, title="Usage Examples", border_style="blue")


def get_configuration() -> Dict[str, Any]:
    """Get configuration through Rich prompts."""
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]🚀 Streaming Legal Document Ingester[/bold blue]\n\n"
        "This tool processes legal documents with embeddings and ingests them into Weaviate.\n"
        "Configure your settings below or use defaults.",
        title="Welcome",
        border_style="blue"
    ))
    
    config = {}
    
    # Dataset path (required)
    config['dataset_path'] = Prompt.ask(
        "[bold cyan]Dataset path[/bold cyan]",
        default="JuDDGES/pl-court-raw"
    )
    
    # Weaviate URL
    config['weaviate_url'] = Prompt.ask(
        "[bold cyan]Weaviate URL[/bold cyan]",
        default="http://localhost:8080"
    )
    
    # Embedding model
    models = [
        "sdadas/mmlw-roberta-large",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ]
    
    console.print("\n[bold cyan]Available embedding models:[/bold cyan]")
    for i, model in enumerate(models, 1):
        console.print(f"  {i}. {model}")
    
    model_choice = Prompt.ask(
        "[bold cyan]Select embedding model[/bold cyan]",
        choices=["1", "2", "3"],
        default="1"
    )
    config['embedding_model'] = models[int(model_choice) - 1]
    
    # Advanced settings
    show_advanced = Confirm.ask(
        "[bold cyan]Show advanced settings?[/bold cyan]",
        default=False
    )
    
    if show_advanced:
        config['chunk_size'] = int(Prompt.ask(
            "[bold cyan]Chunk size (characters)[/bold cyan]",
            default="512"
        ))
        
        config['overlap'] = int(Prompt.ask(
            "[bold cyan]Chunk overlap (characters)[/bold cyan]",
            default="128"
        ))
        
        config['batch_size'] = int(Prompt.ask(
            "[bold cyan]Batch size (embeddings)[/bold cyan]",
            default="32"
        ))
        
        config['tracker_db'] = Prompt.ask(
            "[bold cyan]Tracker database path[/bold cyan]",
            default="processed_documents.db"
        )
        
        config['streaming'] = Confirm.ask(
            "[bold cyan]Use streaming mode?[/bold cyan]",
            default=True
        )
    else:
        # Use defaults
        config.update({
            'chunk_size': 512,
            'overlap': 128,
            'batch_size': 32,
            'tracker_db': 'processed_documents.db',
            'streaming': True
        })
    
    # Reset tracker
    config['reset_tracker'] = Confirm.ask(
        "[bold cyan]Reset tracker database?[/bold cyan]",
        default=False
    )
    
    # Log level
    config['log_level'] = Prompt.ask(
        "[bold cyan]Log level[/bold cyan]",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO"
    )
    
    return config


def display_configuration(config: Dict[str, Any]) -> None:
    """Display configuration in a nice table."""
    console = Console()
    
    table = Table(title="Configuration Settings", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan", width=20)
    table.add_column("Value", style="green")
    table.add_column("Description", style="dim")
    
    descriptions = {
        'dataset_path': 'HuggingFace dataset path',
        'weaviate_url': 'Weaviate instance URL',
        'embedding_model': 'Sentence transformer model',
        'chunk_size': 'Text chunk size in characters',
        'overlap': 'Chunk overlap in characters',
        'batch_size': 'Embedding batch size',
        'tracker_db': 'SQLite tracker database',
        'streaming': 'Use streaming mode',
        'reset_tracker': 'Reset tracker on start',
        'log_level': 'Logging verbosity'
    }
    
    for key, value in config.items():
        table.add_row(
            key.replace('_', ' ').title(),
            str(value),
            descriptions.get(key, '')
        )
    
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
    
    main_table.add_row("Total Documents", str(stats.total_documents), f"{stats.total_documents / processing_time:.1f}/sec")
    main_table.add_row("✅ Processed", str(stats.processed_documents), f"{stats.processed_documents / processing_time:.1f}/sec")
    main_table.add_row("⏭️ Skipped", str(stats.skipped_documents), "-")
    main_table.add_row("❌ Failed", str(stats.failed_documents), "-")
    main_table.add_row("📄 Total Chunks", str(stats.total_chunks), f"{stats.total_chunks / processing_time:.1f}/sec")
    main_table.add_row("⏱️ Processing Time", f"{stats.processing_time:.1f}s", "-")
    
    # Tracker statistics table
    tracker_table = Table(title="💾 Tracker Database", show_header=True, header_style="bold magenta")
    tracker_table.add_column("Status", style="cyan")
    tracker_table.add_column("Count", style="green", justify="right")
    tracker_table.add_column("Percentage", style="yellow", justify="right")
    
    total_tracked = max(tracker_stats['total'], 1)
    
    tracker_table.add_row("📝 Total Tracked", str(tracker_stats['total']), "100%")
    tracker_table.add_row("✅ Successful", str(tracker_stats['successful']), f"{tracker_stats['successful'] / total_tracked * 100:.1f}%")
    tracker_table.add_row("❌ Failed", str(tracker_stats['failed']), f"{tracker_stats['failed'] / total_tracked * 100:.1f}%")
    
    # Display tables side by side
    console.print(Columns([main_table, tracker_table], equal=True))
    
    # Success message
    if stats.failed_documents == 0:
        console.print(Panel.fit(
            f"[bold green]🎉 Processing completed successfully![/bold green]\n\n"
            f"All {stats.processed_documents} documents processed without errors.\n"
            f"Generated {stats.total_chunks} searchable chunks.\n"
            f"Average processing rate: {stats.processed_documents / processing_time:.1f} docs/sec",
            title="Success",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            f"[bold yellow]⚠️ Processing completed with issues[/bold yellow]\n\n"
            f"Successfully processed: {stats.processed_documents} documents\n"
            f"Failed: {stats.failed_documents} documents\n"
            f"Check logs for error details.",
            title="Warning",
            border_style="yellow"
        ))


def main():
    """Main CLI interface using Rich."""
    console = Console()
    
    # Check if running in interactive mode (no arguments)
    if len(sys.argv) == 1:
        # Interactive mode with Rich prompts
        config = get_configuration()
        
        # Display configuration
        console.print("\n")
        display_configuration(config)
        
        # Confirm before starting
        if not Confirm.ask("\n[bold cyan]Start processing with these settings?[/bold cyan]"):
            console.print("[bold red]Cancelled by user[/bold red]")
            return
    
    else:
        # Command line mode - parse arguments
        import argparse
        
        parser = argparse.ArgumentParser(
            description="Simple streaming ingestion for legal documents",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Required arguments
        parser.add_argument(
            "--dataset-path",
            required=True,
            help="Path to HuggingFace dataset (e.g., 'JuDDGES/pl-court-raw')"
        )
        
        # Optional arguments
        parser.add_argument(
            "--weaviate-url",
            default="http://localhost:8080",
            help="Weaviate instance URL (default: http://localhost:8080)"
        )
        
        parser.add_argument(
            "--embedding-model",
            default="sdadas/mmlw-roberta-large",
            help="Embedding model name (default: sdadas/mmlw-roberta-large)"
        )
        
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=512,
            help="Text chunk size in characters (default: 512)"
        )
        
        parser.add_argument(
            "--overlap",
            type=int,
            default=128,
            help="Chunk overlap in characters (default: 128)"
        )
        
        parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Batch size for embedding generation (default: 32)"
        )
        
        parser.add_argument(
            "--tracker-db",
            default="processed_documents.db",
            help="SQLite database for tracking processed documents"
        )
        
        parser.add_argument(
            "--reset-tracker",
            action="store_true",
            help="Reset the processed documents tracker before starting"
        )
        
        parser.add_argument(
            "--no-streaming",
            action="store_true",
            help="Disable streaming mode (load full dataset into memory)"
        )
        
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
            help="Log level (default: INFO)"
        )
        
        parser.add_argument(
            "--help-examples",
            action="store_true",
            help="Show usage examples"
        )
        
        parser.add_argument(
            "--api-key",
            help="Weaviate API key for authentication"
        )
        
        args = parser.parse_args()
        
        if args.help_examples:
            console.print(create_help_panel())
            return
        
        # Convert args to config dict
        config = {
            'dataset_path': args.dataset_path,
            'weaviate_url': args.weaviate_url,
            'embedding_model': args.embedding_model,
            'chunk_size': args.chunk_size,
            'overlap': args.overlap,
            'batch_size': args.batch_size,
            'tracker_db': args.tracker_db,
            'reset_tracker': args.reset_tracker,
            'streaming': not args.no_streaming,
            'log_level': args.log_level,
            'api_key': args.api_key
        }
        
        # Display configuration
        display_configuration(config)
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=config['log_level'],
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}"
    )
    
    try:
        # Initialize ingester with context manager
        console.print("\n[bold green]🚀 Initializing streaming ingester...[/bold green]")
        with StreamingIngester(
            weaviate_url=config['weaviate_url'],
            embedding_model=config['embedding_model'],
            chunk_size=config['chunk_size'],
            overlap=config['overlap'],
            batch_size=config['batch_size'],
            tracker_db=config['tracker_db'],
            api_key=config['api_key']
        ) as ingester:
            
            # Reset tracker if requested
            if config['reset_tracker']:
                console.print("[bold yellow]🔄 Resetting tracker database...[/bold yellow]")
                ingester.reset_tracker()
            
            # Process dataset
            console.print("[bold green]📊 Starting dataset processing...[/bold green]")
            stats = ingester.process_dataset(
                dataset_path=config['dataset_path'],
                streaming=config['streaming']
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
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        logger.exception("Ingestion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()