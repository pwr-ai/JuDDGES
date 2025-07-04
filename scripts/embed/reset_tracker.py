#!/usr/bin/env python3
"""
Reset the processed documents tracker database.
This removes all tracking information so documents can be reprocessed.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from juddges.data.stream_ingester import StreamingIngester
from juddges.settings import ROOT_PATH

load_dotenv(ROOT_PATH / ".env", override=True)


def reset_tracker():
    """Reset the processed documents tracker."""
    console = Console()
    
    # Header
    console.print(
        Panel.fit("🔄 Reset Tracker Database", style="bold yellow", border_style="bright_yellow")
    )
    
    # Confirm reset
    if "--force" not in sys.argv:
        if not Confirm.ask(
            "[bold yellow]⚠️  This will reset all document processing history. Continue?[/bold yellow]", 
            default=False
        ):
            console.print("[yellow]Operation cancelled.[/yellow]")
            return
    else:
        console.print("[yellow]Force flag detected - proceeding with reset[/yellow]")
    
    try:
        # Get Weaviate URL from environment variables
        weaviate_host = os.getenv("WEAVIATE_HOST", "127.0.0.1")
        weaviate_port = os.getenv("WEAVIATE_PORT", "8084")
        weaviate_url = f"http://{weaviate_host}:{weaviate_port}"
        
        with StreamingIngester(weaviate_url=weaviate_url) as ingester:
            console.print("✅ [green]Connected to StreamingIngester[/green]")
            
            # Show current tracker stats
            tracker_stats = ingester.tracker.get_stats()
            console.print(f"\n[bold]Current tracker state:[/bold]")
            console.print(f"  Total tracked: {tracker_stats['total']}")
            console.print(f"  Successful: {tracker_stats['successful']}")
            console.print(f"  Failed: {tracker_stats['failed']}")
            
            if tracker_stats['total'] > 0:
                # Reset tracker
                console.print(f"\n[bold yellow]Resetting tracker database...[/bold yellow]")
                ingester.reset_tracker()
                
                # Verify reset
                new_stats = ingester.tracker.get_stats()
                console.print(f"\n[bold green]Tracker reset complete![/bold green]")
                console.print(f"  Total tracked: {new_stats['total']}")
                console.print(f"  Successful: {new_stats['successful']}")
                console.print(f"  Failed: {new_stats['failed']}")
            else:
                console.print("\n[dim]Tracker database is already empty[/dim]")
                
    except Exception as e:
        console.print(f"❌ [red]Error during tracker reset:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    reset_tracker()