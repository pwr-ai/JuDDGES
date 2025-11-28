#!/usr/bin/env python3
"""
Check for duplicated information in sample documents.
"""

import os
import json
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8084")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "<REDACTED-WEAVIATE-API-KEY>")
HEADERS = {"Authorization": f"Bearer {WEAVIATE_API_KEY}"}


def fetch_sample_documents(document_type: str, limit: int = 5):
    """Fetch sample documents by type."""
    query = f"""
    {{
      Get {{
        LegalDocuments(
          where: {{
            path: ["document_type"]
            operator: Equal
            valueText: "{document_type}"
          }}
          limit: {limit}
        ) {{
          document_id
          document_type
          title
          country
          language
          date_issued
          document_number
          full_text
          summary
          thesis
          keywords
          issuing_body
          source
          legal_references
          parties
          outcome
        }}
      }}
    }}
    """

    try:
        response = requests.post(
            f"{WEAVIATE_URL}/v1/graphql",
            json={"query": query},
            headers=HEADERS,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            if "errors" in data:
                console.print(f"[red]GraphQL Errors:[/red]")
                for error in data["errors"]:
                    console.print(f"  • {error.get('message')}")
                return []

            return data.get("data", {}).get("Get", {}).get("LegalDocuments", [])
        else:
            console.print(f"[red]Error:[/red] {response.status_code}")
            return []
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return []


def analyze_field_overlap(doc: dict) -> dict:
    """Analyze which fields contain overlapping information."""
    overlaps = []

    full_text = doc.get("full_text", "")
    summary = doc.get("summary", "")
    thesis = doc.get("thesis", "")
    title = doc.get("title", "")

    # Check if summary is contained in full_text
    if summary and full_text and len(summary) > 20:
        if summary in full_text:
            overlaps.append({
                "type": "summary_in_full_text",
                "description": "Summary is substring of full_text",
                "summary_length": len(summary),
                "full_text_length": len(full_text)
            })

    # Check if thesis is contained in full_text
    if thesis and full_text and len(thesis) > 20:
        if thesis in full_text:
            overlaps.append({
                "type": "thesis_in_full_text",
                "description": "Thesis is substring of full_text",
                "thesis_length": len(thesis),
                "full_text_length": len(full_text)
            })

    # Check if thesis and summary are the same
    if thesis and summary and len(thesis) > 20 and len(summary) > 20:
        if thesis == summary:
            overlaps.append({
                "type": "thesis_equals_summary",
                "description": "Thesis and summary are identical",
                "length": len(thesis)
            })
        elif thesis in summary or summary in thesis:
            overlaps.append({
                "type": "thesis_summary_overlap",
                "description": "Thesis and summary overlap",
                "thesis_length": len(thesis),
                "summary_length": len(summary)
            })

    # Check if title is in full_text
    if title and full_text and len(title) > 10:
        if title.lower() in full_text.lower():
            overlaps.append({
                "type": "title_in_full_text",
                "description": "Title appears in full_text",
                "title_length": len(title)
            })

    return {
        "document_id": doc.get("document_id"),
        "document_type": doc.get("document_type"),
        "has_overlaps": len(overlaps) > 0,
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "field_sizes": {
            "full_text": len(full_text) if full_text else 0,
            "summary": len(summary) if summary else 0,
            "thesis": len(thesis) if thesis else 0,
            "title": len(title) if title else 0
        }
    }


def display_document_analysis(doc: dict, analysis: dict):
    """Display detailed analysis of a document."""
    console.print(f"\n[bold cyan]Document: {analysis['document_id']}[/bold cyan]")
    console.print(f"[dim]Type: {analysis['document_type']}[/dim]\n")

    # Field sizes
    table = Table(title="Field Sizes")
    table.add_column("Field", style="cyan")
    table.add_column("Size (chars)", style="green", justify="right")
    table.add_column("Empty", style="yellow")

    for field, size in analysis["field_sizes"].items():
        is_empty = "Yes" if size == 0 else "No"
        table.add_row(field, f"{size:,}", is_empty)

    console.print(table)

    # Overlaps
    if analysis["has_overlaps"]:
        console.print(f"\n[yellow]⚠ Found {analysis['overlap_count']} overlap(s):[/yellow]\n")

        for overlap in analysis["overlaps"]:
            console.print(f"  • [bold]{overlap['type']}[/bold]")
            console.print(f"    {overlap['description']}")

            # Show additional details
            for key, value in overlap.items():
                if key not in ["type", "description"]:
                    console.print(f"    {key}: {value:,}")
            console.print()
    else:
        console.print("\n[green]✓ No overlaps detected[/green]\n")

    # Show sample content
    console.print("[bold]Sample Content:[/bold]\n")

    if doc.get("title"):
        console.print(f"[cyan]Title:[/cyan] {doc['title'][:100]}...")

    if doc.get("summary"):
        summary_preview = doc["summary"][:200] + "..." if len(doc.get("summary", "")) > 200 else doc.get("summary", "")
        console.print(f"\n[cyan]Summary:[/cyan] {summary_preview}")

    if doc.get("thesis"):
        thesis_preview = doc["thesis"][:200] + "..." if len(doc.get("thesis", "")) > 200 else doc.get("thesis", "")
        console.print(f"\n[cyan]Thesis:[/cyan] {thesis_preview}")

    if doc.get("full_text"):
        text_preview = doc["full_text"][:300] + "..." if len(doc.get("full_text", "")) > 300 else doc.get("full_text", "")
        console.print(f"\n[cyan]Full Text Preview:[/cyan] {text_preview}")


def main():
    console.print("[bold magenta]═══════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]  Document Duplication Analysis[/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════[/bold magenta]")

    document_types = ["judgment", "tax_interpretation"]

    all_results = {}

    for doc_type in document_types:
        console.print(f"\n[bold yellow]{'=' * 60}[/bold yellow]")
        console.print(f"[bold yellow]Analyzing: {doc_type.upper()}[/bold yellow]")
        console.print(f"[bold yellow]{'=' * 60}[/bold yellow]")

        console.print(f"\n[dim]Fetching sample documents...[/dim]")
        documents = fetch_sample_documents(doc_type, limit=3)

        if not documents:
            console.print(f"[red]No documents found for type: {doc_type}[/red]")
            continue

        console.print(f"[green]✓[/green] Found {len(documents)} sample documents\n")

        doc_analyses = []

        for doc in documents:
            analysis = analyze_field_overlap(doc)
            doc_analyses.append(analysis)
            display_document_analysis(doc, analysis)

        all_results[doc_type] = doc_analyses

    # Summary statistics
    console.print(f"\n[bold yellow]{'=' * 60}[/bold yellow]")
    console.print("[bold yellow]SUMMARY STATISTICS[/bold yellow]")
    console.print(f"[bold yellow]{'=' * 60}[/bold yellow]\n")

    summary_table = Table(title="Duplication Summary")
    summary_table.add_column("Document Type", style="cyan")
    summary_table.add_column("Samples", style="green", justify="right")
    summary_table.add_column("With Overlaps", style="yellow", justify="right")
    summary_table.add_column("Overlap Rate", style="red", justify="right")

    for doc_type, analyses in all_results.items():
        total = len(analyses)
        with_overlaps = sum(1 for a in analyses if a["has_overlaps"])
        rate = (with_overlaps / total * 100) if total > 0 else 0

        summary_table.add_row(
            doc_type,
            str(total),
            str(with_overlaps),
            f"{rate:.1f}%"
        )

    console.print(summary_table)

    # Common patterns
    console.print("\n[bold]Common Duplication Patterns:[/bold]\n")

    all_overlap_types = {}
    for analyses in all_results.values():
        for analysis in analyses:
            for overlap in analysis["overlaps"]:
                overlap_type = overlap["type"]
                all_overlap_types[overlap_type] = all_overlap_types.get(overlap_type, 0) + 1

    if all_overlap_types:
        for overlap_type, count in sorted(all_overlap_types.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  • {overlap_type}: {count} occurrences")
    else:
        console.print("  [green]No common duplication patterns found[/green]")

    # Recommendations
    console.print("\n[bold cyan]Recommendations:[/bold cyan]\n")

    if any(a["has_overlaps"] for analyses in all_results.values() for a in analyses):
        console.print("  1. [yellow]Consider removing redundant fields[/yellow] to save storage")
        console.print("  2. [yellow]Keep only full_text + metadata[/yellow] if summary/thesis are substrings")
        console.print("  3. [yellow]Use references[/yellow] instead of duplicating text")
        console.print("  4. [yellow]Implement deduplication[/yellow] in ingestion pipeline")
    else:
        console.print("  [green]✓ No significant duplication detected[/green]")
        console.print("  [green]✓ Current schema appears optimized[/green]")


if __name__ == "__main__":
    main()
