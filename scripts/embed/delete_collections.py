#!/usr/bin/env python3
"""
Delete all collections from Weaviate database.
This script uses StreamingIngester to remove all legal document collections.
"""

import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from juddges.data.stream_ingester import StreamingIngester
from juddges.settings import ROOT_PATH

load_dotenv(ROOT_PATH / ".env", override=True)


def delete_all_collections():
    """Delete all collections from Weaviate."""
    console = Console()

    # Header
    console.print(
        Panel.fit("🗑️ Delete Weaviate Collections", style="bold red", border_style="bright_red")
    )

    # Confirm deletion
    if "--force" not in sys.argv:
        if not Confirm.ask(
            "[bold red]⚠️  This will DELETE ALL collections and data. Are you sure?[/bold red]",
            default=False,
        ):
            console.print("[yellow]Operation cancelled.[/yellow]")
            return
    else:
        console.print("[yellow]Force flag detected - proceeding with deletion[/yellow]")

    try:
        # Get Weaviate URL from environment variables
        weaviate_host = os.getenv("WEAVIATE_HOST", "127.0.0.1")
        weaviate_port = os.getenv("WEAVIATE_PORT", "8084")
        weaviate_url = f"http://{weaviate_host}:{weaviate_port}"

        with StreamingIngester(weaviate_url=weaviate_url) as ingester:
            console.print("✅ [green]Connected to Weaviate database[/green]")

            # List collections before deletion
            collections = ingester.weaviate_client.collections.list_all()

            # Handle different return types from list_all()
            if isinstance(collections, dict):
                collection_names = list(collections.keys())
            elif hasattr(collections, "__iter__"):
                collection_names = [collection.name for collection in collections]
            else:
                collection_names = []

            if collection_names:
                console.print(
                    f"\n[bold]Found {len(collection_names)} collection(s) to delete:[/bold]"
                )
                for name in collection_names:
                    console.print(f"  - {name}")

                console.print("\n[bold red]Deleting collections...[/bold red]")

                # Delete each collection
                deleted_count = 0
                for collection_name in collection_names:
                    try:
                        ingester.weaviate_client.collections.delete(collection_name)
                        console.print(f"[green]✓ Deleted: {collection_name}[/green]")
                        deleted_count += 1
                    except Exception as e:
                        console.print(f"[red]✗ Failed to delete {collection_name}: {e}[/red]")

                console.print(
                    f"\n[bold green]Successfully deleted {deleted_count}/{len(collection_names)} collections[/bold green]"
                )

            else:
                console.print("[dim]No collections found to delete[/dim]")

    except Exception as e:
        console.print(f"❌ [red]Error during deletion:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    delete_all_collections()
