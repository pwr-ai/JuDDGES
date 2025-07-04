#!/usr/bin/env python3
"""
Check document counts in Weaviate database collections.
This script uses the existing WeaviateLegalDocumentsDatabase class to query collection sizes.
"""

import os
import sys
from typing import Dict

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from juddges.data.stream_ingester import StreamingIngester
from juddges.settings import ROOT_PATH

load_dotenv(ROOT_PATH / ".env", override=True)


def list_all_collections() -> Dict[str, int]:
    """
    List all collections in Weaviate with their document counts.

    Returns:
        Dict[str, int]: Dictionary with collection names as keys and document counts as values.
    """
    console = Console()

    # Header
    console.print(
        Panel.fit("📋 All Weaviate Collections", style="bold cyan", border_style="bright_cyan")
    )

    collection_counts = {}

    try:
        # Get Weaviate URL from environment variables
        weaviate_host = os.getenv("WEAVIATE_HOST", "127.0.0.1")
        weaviate_port = os.getenv("WEAVIATE_PORT", "8084")
        weaviate_url = f"http://{weaviate_host}:{weaviate_port}"

        with StreamingIngester(weaviate_url=weaviate_url) as ingester:
            console.print("✅ [green]Connected to Weaviate database[/green]")

            # Get all collections
            collections = ingester.weaviate_client.collections.list_all()

            # Handle different return types from list_all()
            if isinstance(collections, dict):
                collection_names = list(collections.keys())
            elif hasattr(collections, "__iter__"):
                collection_names = [collection.name for collection in collections]
            else:
                collection_names = []

            if collection_names:
                console.print(f"\n[bold]Found {len(collection_names)} collection(s):[/bold]")
                for name in sorted(collection_names):
                    try:
                        collection = ingester.weaviate_client.collections.get(name)
                        response = collection.aggregate.over_all(total_count=True)
                        count = response.total_count
                        collection_counts[name] = count
                        console.print(
                            f"  📊 [cyan]{name}:[/cyan] [bold yellow]{count:,}[/bold yellow] objects"
                        )
                    except Exception as e:
                        console.print(f"  ❌ [red]{name}:[/red] Error getting count ({e})")
                        collection_counts[name] = 0
            else:
                console.print("[dim]No collections found[/dim]")

    except Exception as e:
        console.print(f"❌ [red]Database connection error:[/red] {e}")
        console.print("\n💡 [yellow]To start Weaviate:[/yellow]")
        console.print("cd weaviate/ && docker-compose up -d")

    return collection_counts


def check_document_counts() -> Dict[str, int]:
    """
    Check document counts in both legal_documents and document_chunks collections.

    Returns:
        Dict[str, int]: Dictionary with collection names as keys and document counts as values.
    """
    console = Console()

    # Header
    console.print(
        Panel.fit("🗄️ Weaviate Document Count Check", style="bold blue", border_style="bright_blue")
    )

    counts = {}

    try:
        # Get Weaviate URL from environment variables
        weaviate_host = os.getenv("WEAVIATE_HOST", "127.0.0.1")
        weaviate_port = os.getenv("WEAVIATE_PORT", "8084")
        weaviate_url = f"http://{weaviate_host}:{weaviate_port}"

        # Initialize StreamingIngester connection
        with StreamingIngester(weaviate_url=weaviate_url) as ingester:
            console.print("✅ [green]Connected to Weaviate database[/green]")

            # Check legal_documents collection
            try:
                legal_collection = ingester.weaviate_client.collections.get(
                    ingester.LEGAL_DOCUMENTS_COLLECTION
                )
                legal_response = legal_collection.aggregate.over_all(total_count=True)
                legal_docs_count = legal_response.total_count
                counts["legal_documents"] = legal_docs_count
                console.print(
                    f"📊 [cyan]Legal documents collection:[/cyan] [bold yellow]{legal_docs_count:,}[/bold yellow] documents"
                )
            except Exception as e:
                console.print(f"❌ [red]Error getting legal documents count:[/red] {e}")
                counts["legal_documents"] = 0

            # Check document_chunks collection
            try:
                chunks_collection = ingester.weaviate_client.collections.get(
                    ingester.DOCUMENT_CHUNKS_COLLECTION
                )
                chunks_response = chunks_collection.aggregate.over_all(total_count=True)
                chunks_count = chunks_response.total_count
                counts["document_chunks"] = chunks_count
                console.print(
                    f"📄 [cyan]Document chunks collection:[/cyan] [bold yellow]{chunks_count:,}[/bold yellow] chunks"
                )
            except Exception as e:
                console.print(f"❌ [red]Error getting document chunks count:[/red] {e}")
                counts["document_chunks"] = 0

            # Summary table
            table = Table(
                title="📊 Collection Summary",
                show_header=True,
                header_style="bold magenta",
                border_style="bright_blue",
            )
            table.add_column("Collection", style="cyan", width=20)
            table.add_column("Count", style="yellow", width=15, justify="right")
            table.add_column("Type", style="white", width=25)

            table.add_row(
                "⚖️ legal_documents", f"{counts['legal_documents']:,}", "Full legal documents"
            )
            table.add_row(
                "📄 document_chunks", f"{counts['document_chunks']:,}", "Text chunks for search"
            )

            console.print(table)

            # Total summary
            legal_count = counts.get("legal_documents", 0) or 0
            chunks_count = counts.get("document_chunks", 0) or 0
            total_count = legal_count + chunks_count
            summary_text = Text()
            summary_text.append("📊 Total Objects: ", style="bold cyan")
            summary_text.append(f"{total_count:,}", style="bold yellow")
            summary_text.append("\n🏗️ Collections: ", style="bold cyan")
            summary_text.append("2", style="bold yellow")

            console.print(Panel(summary_text, title="📈 Database Summary", border_style="green"))

            # Show ratio if both collections have data
            if legal_count > 0 and chunks_count > 0:
                ratio = chunks_count / legal_count
                console.print(
                    f"\n📊 [bold green]Chunks per document ratio:[/bold green] [yellow]{ratio:.1f}[/yellow] chunks per document"
                )

    except Exception as e:
        console.print(f"❌ [red]Database connection error:[/red] {e}")
        console.print("\n💡 [yellow]To start Weaviate:[/yellow]")
        console.print("cd weaviate/ && docker-compose up -d")
        return {"legal_documents": 0, "document_chunks": 0}

    return counts


def inspect_documents() -> None:
    """Inspect the actual documents in the collections to understand what's stored."""
    console = Console()

    # Header
    console.print(
        Panel.fit("🔍 Document Inspection", style="bold magenta", border_style="bright_magenta")
    )

    try:
        # Get Weaviate URL from environment variables
        weaviate_host = os.getenv("WEAVIATE_HOST", "127.0.0.1")
        weaviate_port = os.getenv("WEAVIATE_PORT", "8084")
        weaviate_url = f"http://{weaviate_host}:{weaviate_port}"

        with StreamingIngester(weaviate_url=weaviate_url) as ingester:
            console.print("✅ [green]Connected to Weaviate database[/green]")

            # Inspect legal documents
            console.print("\n[bold cyan]Legal Documents Sample:[/bold cyan]")
            try:
                legal_collection = ingester.weaviate_client.collections.get(
                    ingester.LEGAL_DOCUMENTS_COLLECTION
                )

                # Get first 10 documents with their properties
                response = legal_collection.query.fetch_objects(
                    limit=10,
                    return_properties=[
                        "document_id",
                        "title",
                        "document_type",
                        "language",
                        "date_issued",
                    ],
                )

                if response.objects:
                    table = Table(show_header=True, header_style="bold cyan")
                    table.add_column("ID", style="yellow", width=15)
                    table.add_column("Title", style="white", width=30)
                    table.add_column("Type", style="green", width=15)
                    table.add_column("Language", style="blue", width=8)
                    table.add_column("Date", style="magenta", width=12)

                    for obj in response.objects:
                        props = obj.properties
                        table.add_row(
                            str(props.get("document_id", ""))[:15],
                            str(props.get("title", ""))[:30],
                            str(props.get("document_type", "")),
                            str(props.get("language", "")),
                            str(props.get("date_issued", ""))[:12],
                        )

                    console.print(table)
                else:
                    console.print("[red]No legal documents found[/red]")

            except Exception as e:
                console.print(f"❌ [red]Error inspecting legal documents:[/red] {e}")

            # Inspect document chunks - show sample and unique document IDs
            console.print("\n[bold cyan]Document Chunks Analysis:[/bold cyan]")
            try:
                chunks_collection = ingester.weaviate_client.collections.get(
                    ingester.DOCUMENT_CHUNKS_COLLECTION
                )

                # Get unique document IDs from chunks
                response = chunks_collection.query.fetch_objects(
                    limit=2349,  # Get all chunks
                    return_properties=["document_id", "position", "chunk_text"],
                )

                if response.objects:
                    # Count unique document IDs
                    document_ids = set()
                    chunk_lengths = []
                    for obj in response.objects:
                        doc_id = obj.properties.get("document_id")
                        if doc_id:
                            document_ids.add(doc_id)
                        chunk_text = obj.properties.get("chunk_text", "")
                        if isinstance(chunk_text, str):
                            chunk_lengths.append(len(chunk_text))

                    console.print(
                        f"📊 [yellow]Unique document IDs in chunks:[/yellow] {len(document_ids)}"
                    )
                    console.print(
                        f"📊 [yellow]Average chunk length:[/yellow] {sum(chunk_lengths) / len(chunk_lengths):.1f} characters"
                    )

                    # Show sample document IDs
                    sample_ids = list(document_ids)[:10]
                    console.print(f"📄 [cyan]Sample document IDs:[/cyan] {', '.join(sample_ids)}")

                    # Show chunk distribution
                    doc_chunk_counts = {}
                    for obj in response.objects:
                        doc_id = obj.properties.get("document_id")
                        if doc_id:
                            doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0) + 1

                    if doc_chunk_counts:
                        max_chunks = max(doc_chunk_counts.values())
                        min_chunks = min(doc_chunk_counts.values())
                        avg_chunks = sum(doc_chunk_counts.values()) / len(doc_chunk_counts)
                        console.print(
                            f"📊 [yellow]Chunks per document - Min:[/yellow] {min_chunks}, [yellow]Max:[/yellow] {max_chunks}, [yellow]Avg:[/yellow] {avg_chunks:.1f}"
                        )

                else:
                    console.print("[red]No document chunks found[/red]")

            except Exception as e:
                console.print(f"❌ [red]Error inspecting document chunks:[/red] {e}")

    except Exception as e:
        console.print(f"❌ [red]Database connection error:[/red] {e}")


def main():
    """Main function to run the document count check."""
    from rich.console import Console
    from rich.prompt import Confirm

    console = Console()

    # Check for inspect flag
    if "--inspect" in sys.argv:
        inspect_documents()
        return

    # Ask user what they want to see
    show_all = "--all" in sys.argv or Confirm.ask(
        "Show all collections (not just legal document collections)?", default=False
    )

    if show_all:
        # Show all collections
        counts = list_all_collections()
        total_objects = sum(counts.values())

        if total_objects == 0:
            console.print("\n[red]No objects found in any collection.[/red]")
            sys.exit(1)
        else:
            console.print(
                f"\n[green]Found {total_objects} total objects across {len(counts)} collections.[/green]"
            )
            sys.exit(0)
    else:
        # Show only legal document collections
        counts = check_document_counts()

        # Return appropriate exit code
        if counts["legal_documents"] == 0 and counts["document_chunks"] == 0:
            console.print("\n[red]No documents found in either collection.[/red]")
            sys.exit(1)
        else:
            console.print(
                f"\n[green]Found {counts['legal_documents']} legal documents and {counts['document_chunks']} document chunks.[/green]"
            )
            sys.exit(0)


if __name__ == "__main__":
    main()
