#!/usr/bin/env python3
"""
Recreate Weaviate collections with proper multi-vector named vectors configuration.
This script deletes existing collections and creates new ones with the correct schema.
"""

import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase
from juddges.settings import ROOT_PATH

load_dotenv(ROOT_PATH / ".env", override=True)


def recreate_collections():
    """Delete and recreate collections with proper multi-vector schema."""
    console = Console()

    # Header
    console.print(
        Panel.fit(
            "🔄 Recreate Weaviate Collections with Multi-Vector Support",
            style="bold blue",
            border_style="bright_blue",
        )
    )

    console.print(
        "[yellow]This will:[/yellow]\n"
        "1. Delete existing LegalDocuments and DocumentChunks collections\n"
        "2. Recreate them with proper named vectors (base, dev, fast)\n"
        "3. Configure multi-vector support for embeddings"
    )

    # Confirm deletion
    if "--force" not in sys.argv:
        if not Confirm.ask(
            "[bold red]⚠️  This will DELETE ALL existing collections and data. Continue?[/bold red]",
            default=False,
        ):
            console.print("[yellow]Operation cancelled.[/yellow]")
            return
    else:
        console.print("[yellow]Force flag detected - proceeding with recreation[/yellow]")

    try:
        # Get Weaviate connection details
        weaviate_host = os.getenv("WEAVIATE_HOST", "127.0.0.1")
        weaviate_port = os.getenv("WEAVIATE_PORT", "8084")
        weaviate_url = f"http://{weaviate_host}:{weaviate_port}"

        console.print(f"[dim]Connecting to Weaviate at {weaviate_url}[/dim]")

        # Initialize database with context manager
        with WeaviateLegalDocumentsDatabase() as db:
            console.print("✅ [green]Connected to Weaviate database[/green]")

            # List existing collections
            try:
                collections = db.client.collections.list_all()
                if isinstance(collections, dict):
                    collection_names = list(collections.keys())
                elif hasattr(collections, "__iter__"):
                    collection_names = [collection.name for collection in collections]
                else:
                    collection_names = []

                if collection_names:
                    console.print(
                        f"\n[bold]Found {len(collection_names)} existing collection(s):[/bold]"
                    )
                    for name in collection_names:
                        console.print(f"  - {name}")

                    console.print("\n[bold red]Deleting existing collections...[/bold red]")

                    # Delete each collection
                    deleted_count = 0
                    for collection_name in collection_names:
                        try:
                            db.delete_collection(collection_name)
                            console.print(f"[green]✓ Deleted: {collection_name}[/green]")
                            deleted_count += 1
                        except Exception as e:
                            console.print(f"[red]✗ Failed to delete {collection_name}: {e}[/red]")

                    console.print(
                        f"[bold green]Successfully deleted {deleted_count}/{len(collection_names)} collections[/bold green]"
                    )
                else:
                    console.print("[dim]No existing collections found[/dim]")

            except Exception as e:
                console.print(f"[yellow]Warning: Could not list collections: {e}[/yellow]")
                console.print("[dim]Proceeding with collection creation...[/dim]")

            # Create new collections with proper schema
            console.print(
                "\n[bold green]Creating collections with multi-vector support...[/bold green]"
            )

            db.create_collections()

            console.print("[bold green]✅ Collections created successfully![/bold green]")

            # Verify collections
            try:
                collections = db.client.collections.list_all()
                if isinstance(collections, dict):
                    new_collection_names = list(collections.keys())
                elif hasattr(collections, "__iter__"):
                    new_collection_names = [collection.name for collection in collections]
                else:
                    new_collection_names = []

                console.print(f"\n[bold cyan]Created collections:[/bold cyan]")
                for name in new_collection_names:
                    console.print(f"  ✓ {name}")

                # Show vector configuration
                console.print(f"\n[bold cyan]Vector Configuration:[/bold cyan]")
                console.print("Each collection now supports named vectors:")
                console.print("  • [bold]base[/bold]: sdadas/mmlw-roberta-large (high accuracy)")
                console.print(
                    "  • [bold]dev[/bold]: sentence-transformers/all-mpnet-base-v2 (balanced)"
                )
                console.print(
                    "  • [bold]fast[/bold]: sentence-transformers/all-MiniLM-L6-v2 (fast)"
                )

            except Exception as e:
                console.print(f"[yellow]Warning: Could not verify collections: {e}[/yellow]")

            # Success message
            console.print("\n")
            success_panel = Panel.fit(
                "🎉 [bold green]Collections Recreated Successfully![/bold green]\n\n"
                "Your Weaviate instance now supports multi-vector embeddings.\n"
                "You can now run ingestion with multiple embedding models.",
                style="bold green",
                border_style="bright_green",
            )
            console.print(success_panel)

    except Exception as e:
        console.print(f"❌ [red]Error during collection recreation:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    recreate_collections()
