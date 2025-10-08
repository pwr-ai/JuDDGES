#!/usr/bin/env python3
"""
Analyze raw_content field coverage in Weaviate.

This script provides statistics and insights about:
- Overall raw_content coverage
- Coverage by document type
- Text field comparison (full_text vs raw_content)
- Sample documents missing raw_content
"""

import argparse
from typing import Dict, List

from loguru import logger
from rich.console import Console
from rich.table import Table

from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase


console = Console()


def print_statistics(stats: Dict) -> None:
    """Print raw_content statistics in a formatted table."""
    table = Table(title="Raw Text Coverage Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="green", justify="right")

    table.add_row("Total Documents", str(stats["total_documents"]))
    table.add_row("With raw_content", str(stats["with_raw_content"]))
    table.add_row("Without raw_content", str(stats["without_raw_content"]))
    table.add_row("Coverage %", f"{stats['coverage_percentage']}%")

    console.print(table)


def print_missing_samples(docs: List[Dict], limit: int = 5) -> None:
    """Print sample documents missing raw_content."""
    table = Table(
        title=f"Sample Documents Missing raw_content (showing {min(limit, len(docs))} of {len(docs)})",
        show_header=True,
        header_style="bold yellow",
    )
    table.add_column("Document ID", style="cyan", width=50)
    table.add_column("Type", style="magenta", width=20)
    table.add_column("Title", style="white", width=40)

    for doc in docs[:limit]:
        table.add_row(
            doc.get("document_id", "N/A")[:50],
            doc.get("document_type", "N/A"),
            (doc.get("title", "No title") or "No title")[:40],
        )

    console.print(table)


def print_comparison(comparison: Dict) -> None:
    """Print full_text vs raw_content comparison."""
    table = Table(title="Text Fields Comparison", show_header=True, header_style="bold blue")
    table.add_column("Field", style="cyan", width=30)
    table.add_column("Value", style="green", justify="right")

    table.add_row("Document ID", comparison["document_id"][:50])
    table.add_row("Has full_text", "✓" if comparison["has_full_text"] else "✗")
    table.add_row("Has raw_content", "✓" if comparison["has_raw_content"] else "✗")
    table.add_row("full_text length", f"{comparison['full_text_length']:,} chars")
    table.add_row("raw_content length", f"{comparison['raw_content_length']:,} chars")
    table.add_row("Length difference", f"{comparison['length_difference']:,} chars")

    if comparison["length_ratio"]:
        table.add_row("Length ratio (full/raw)", str(comparison["length_ratio"]))

    console.print(table)


def print_type_coverage(db: WeaviateLegalDocumentsDatabase, document_types: List[str]) -> None:
    """Print coverage statistics by document type."""
    table = Table(
        title="Coverage by Document Type", show_header=True, header_style="bold green"
    )
    table.add_column("Document Type", style="cyan", width=25)
    table.add_column("With raw_content", style="green", justify="right")
    table.add_column("Without raw_content", style="red", justify="right")

    for doc_type in document_types:
        with_raw = db.filter_by_document_type_and_raw_content(
            document_type=doc_type, has_raw_content=True, limit=10000
        )
        without_raw = db.filter_by_document_type_and_raw_content(
            document_type=doc_type, has_raw_content=False, limit=10000
        )

        table.add_row(doc_type, str(len(with_raw)), str(len(without_raw)))

    console.print(table)


def main():
    """Analyze raw_content coverage in Weaviate."""
    parser = argparse.ArgumentParser(description="Analyze raw_content field coverage in Weaviate")
    parser.add_argument(
        "--weaviate-host",
        type=str,
        default="localhost",
        help="Weaviate host (default: localhost)",
    )
    parser.add_argument(
        "--weaviate-port",
        type=int,
        default=8222,
        help="Weaviate HTTP port (default: 8222)",
    )
    parser.add_argument(
        "--weaviate-grpc-port",
        type=int,
        default=50051,
        help="Weaviate gRPC port (default: 50051)",
    )
    parser.add_argument(
        "--show-missing",
        type=int,
        default=10,
        help="Number of missing documents to display (default: 10)",
    )
    parser.add_argument(
        "--compare-document",
        type=str,
        default=None,
        help="Document ID to compare full_text vs raw_content",
    )
    parser.add_argument(
        "--by-type",
        action="store_true",
        help="Show coverage breakdown by document type",
    )

    args = parser.parse_args()

    # Connect to Weaviate
    console.print(
        f"[bold blue]Connecting to Weaviate at {args.weaviate_host}:{args.weaviate_port}..."
    )
    db = WeaviateLegalDocumentsDatabase(
        host=args.weaviate_host,
        port=args.weaviate_port,
        grpc_port=args.weaviate_grpc_port,
    )

    # Overall statistics
    console.print("\n[bold blue]Fetching statistics...")
    stats = db.get_raw_content_statistics()
    print_statistics(stats)

    # Coverage by type
    if args.by_type:
        console.print("\n[bold blue]Analyzing coverage by document type...")
        document_types = ["judgment", "tax_interpretation", "legal_act"]
        print_type_coverage(db, document_types)

    # Show missing documents
    if stats["without_raw_content"] > 0:
        console.print(f"\n[bold blue]Finding documents missing raw_content...")
        missing = db.filter_by_raw_content_presence(has_raw_content=False, limit=args.show_missing)
        print_missing_samples(missing, limit=args.show_missing)

        console.print(
            f"\n[bold yellow]→ Run update script to populate raw_content for {stats['without_raw_content']} documents"
        )
        console.print(
            "[dim]  python scripts/embed/update_raw_content.py --dataset-name juddges/pl-court-raw"
        )

    # Compare specific document
    if args.compare_document:
        console.print(
            f"\n[bold blue]Comparing text fields for document {args.compare_document}..."
        )
        comparison = db.compare_text_fields(args.compare_document)

        if comparison:
            print_comparison(comparison)
        else:
            console.print(f"[red]Document {args.compare_document} not found")

    console.print("\n[bold green]✓ Analysis complete")


if __name__ == "__main__":
    main()
