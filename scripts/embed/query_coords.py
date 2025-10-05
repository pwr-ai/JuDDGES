#!/usr/bin/env python3
"""
Query and count documents with non-null x and y coordinates in Weaviate.

This script provides utilities to:
1. Count documents that have both x and y coordinates set
2. Query and display sample documents with coordinates

Usage:
    # Run interactively to see samples from both collections
    docker compose run --rm web python scripts/embed/query_coords.py

    # Use as a module
    from scripts.embed.query_coords import count_documents_with_coordinates

    with WeaviateLegalDocumentsDatabase() as db:
        count = count_documents_with_coordinates(db, "LegalDocuments")
        print(f"Documents with coords: {count}")
"""

from typing import List, Dict
from weaviate.classes.query import Filter
from rich.console import Console
from rich.table import Table

from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase


def count_documents_with_coordinates(
    db: WeaviateLegalDocumentsDatabase,
    collection_name: str,
    console: Console = None,
) -> int:
    """Count documents that have non-null x and y coordinates.

    Args:
        db: Weaviate database instance
        collection_name: Name of collection (LegalDocuments or DocumentChunks)
        console: Rich console for progress output

    Returns:
        Count of documents with both x and y coordinates set
    """
    collection = db.get_collection(collection_name)
    count = 0
    scanned = 0

    # Iterate and count documents with non-null x and y
    for obj in collection.iterator(return_properties=["x", "y"]):
        x = obj.properties.get("x")
        y = obj.properties.get("y")

        # Count if both x and y are not None
        if x is not None and y is not None:
            count += 1

        scanned += 1

        # Show progress every 1000 documents
        if console and scanned % 1000 == 0:
            console.print(
                f"[dim]Scanned {scanned:,} docs, found {count:,} with coords...[/dim]", end="\r"
            )

    if console:
        console.print("")  # New line after progress

    return count


def get_documents_with_coordinates(
    db: WeaviateLegalDocumentsDatabase,
    collection_name: str,
    limit: int = None,
) -> List[Dict]:
    """Get documents that have non-null x and y coordinates.

    Args:
        db: Weaviate database instance
        collection_name: Name of collection (LegalDocuments or DocumentChunks)
        limit: Maximum number of documents to return

    Returns:
        List of documents with x, y coordinates
    """
    collection = db.get_collection(collection_name)

    documents = []

    # Query all documents and filter for non-null x and y
    for obj in collection.iterator():
        x = obj.properties.get("x")
        y = obj.properties.get("y")

        # Skip if x or y is None
        if x is None or y is None:
            continue

        doc = {
            "uuid": str(obj.uuid),
            "x": x,
            "y": y,
        }

        # Add some identifying properties if they exist
        if "country" in obj.properties:
            doc["country"] = obj.properties.get("country")
        if "source_url" in obj.properties:
            doc["source_url"] = obj.properties.get("source_url")
        if "language" in obj.properties:
            doc["language"] = obj.properties.get("language")
        if "document_type" in obj.properties:
            doc["document_type"] = obj.properties.get("document_type")

        documents.append(doc)

        if limit and len(documents) >= limit:
            break

    return documents


def main():
    console = Console()

    with WeaviateLegalDocumentsDatabase() as db:
        for collection_name in ["LegalDocuments", "DocumentChunks"]:
            console.print(f"\n[bold cyan]Collection: {collection_name}[/bold cyan]")

            # Get top 5 documents with coordinates
            docs = get_documents_with_coordinates(db, collection_name, limit=5)

            if not docs:
                console.print(f"[yellow]No documents with coordinates found[/yellow]")
                continue

            # Create table
            table = Table(title=f"Top 5 Documents with Coordinates ({collection_name})")
            table.add_column("UUID", style="cyan", overflow="fold", max_width=20)
            table.add_column("X", style="green", justify="right")
            table.add_column("Y", style="green", justify="right")

            # Add dynamic columns based on first document
            extra_cols = [k for k in docs[0].keys() if k not in ["uuid", "x", "y"]]
            for col in extra_cols:
                table.add_column(col.title(), style="yellow", overflow="fold", max_width=30)

            # Add rows
            for doc in docs:
                row = [
                    doc["uuid"][:16] + "...",
                    f"{doc['x']:.4f}",
                    f"{doc['y']:.4f}",
                ]
                for col in extra_cols:
                    val = doc.get(col, "")
                    row.append(str(val)[:50] if val else "")

                table.add_row(*row)

            console.print(table)

            # Get total count (efficient - only fetches x,y properties)
            console.print("[dim]Counting total documents with coordinates...[/dim]")
            total_count = count_documents_with_coordinates(db, collection_name, console)
            console.print(f"[green]✓ Total documents with coordinates: {total_count:,}[/green]")


if __name__ == "__main__":
    main()
