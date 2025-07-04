#!/usr/bin/env python3
"""
Check errors from the processed documents tracker database.
This shows details about failed document processing.
"""

import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from juddges.data.stream_ingester import StreamingIngester
from juddges.settings import ROOT_PATH

load_dotenv(ROOT_PATH / ".env", override=True)


def check_errors():
    """Check and display processing errors."""
    console = Console()
    
    # Header
    console.print(
        Panel.fit("🔍 Processing Error Analysis", style="bold red", border_style="bright_red")
    )
    
    try:
        # Get Weaviate URL from environment variables
        weaviate_host = os.getenv("WEAVIATE_HOST", "127.0.0.1")
        weaviate_port = os.getenv("WEAVIATE_PORT", "8084")
        weaviate_url = f"http://{weaviate_host}:{weaviate_port}"
        
        with StreamingIngester(weaviate_url=weaviate_url) as ingester:
            console.print("✅ [green]Connected to StreamingIngester[/green]")
            
            # Get error documents
            error_docs = ingester.tracker.get_error_documents()
            
            if error_docs:
                console.print(f"\n[bold red]Found {len(error_docs)} failed documents:[/bold red]")
                
                # Create table for errors
                table = Table(show_header=True, header_style="bold red")
                table.add_column("Document ID", style="yellow", width=30)
                table.add_column("Processed At", style="cyan", width=20)
                table.add_column("Error Message", style="white", width=50)
                
                for doc in error_docs:
                    error_msg = doc["error_message"]
                    # Truncate long error messages
                    if len(error_msg) > 50:
                        error_msg = error_msg[:47] + "..."
                    
                    table.add_row(
                        doc["document_id"][:30],
                        doc["processed_at"][:20] if doc["processed_at"] else "Unknown",
                        error_msg
                    )
                
                console.print(table)
                
                # Show error summary
                error_types = {}
                for doc in error_docs:
                    error_msg = doc["error_message"]
                    # Extract error type (first few words)
                    error_type = " ".join(error_msg.split()[:3]) if error_msg else "Unknown"
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                
                console.print(f"\n[bold]Error Types Summary:[/bold]")
                for error_type, count in error_types.items():
                    console.print(f"  • {error_type}: {count} occurrences")
                
            else:
                console.print("\n[green]No error documents found in tracker[/green]")
                
            # Show general tracker stats
            tracker_stats = ingester.tracker.get_stats()
            console.print(f"\n[bold]Current tracker state:[/bold]")
            console.print(f"  Total tracked: {tracker_stats['total']}")
            console.print(f"  Successful: {tracker_stats['successful']}")
            console.print(f"  Failed: {tracker_stats['failed']}")
                
    except Exception as e:
        console.print(f"❌ [red]Error checking processing errors:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    check_errors()