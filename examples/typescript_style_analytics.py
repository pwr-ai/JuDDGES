#!/usr/bin/env python3
"""
Example script showing TypeScript-style aggregation queries using Python.

This demonstrates how to achieve similar results to your TypeScript queries:
```typescript
const [documentTypeStats, countryStats] = await Promise.all([
  documentsCollection.aggregate.groupBy.overAll({
    groupBy: { property: 'document_type' }
  }),
  documentsCollection.aggregate.groupBy.overAll({
    groupBy: { property: 'country' }
  })
]);
```
"""

import asyncio
from typing import Any, Dict, Tuple

from rich.console import Console
from rich.table import Table

from juddges.data.stream_ingester import StreamingIngester


async def get_parallel_aggregations(
    ingester: StreamingIngester,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    TypeScript-style parallel aggregation queries.

    Equivalent to:
    const [documentTypeStats, countryStats] = await Promise.all([...])
    """
    # In Python, we can simulate parallel execution
    # For Weaviate queries, they're fast enough to run sequentially
    # but we could use asyncio for true parallelism if needed

    document_type_stats = ingester.get_document_type_stats()
    country_stats = ingester.get_country_stats()

    return document_type_stats, country_stats


def format_aggregation_results(stats: Dict[str, Any], title: str) -> Table:
    """Format aggregation results in a nice table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Value", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Percentage", style="yellow", justify="right")

    total = sum(stats.values())

    # Sort by count (descending)
    sorted_items = sorted(stats.items(), key=lambda x: x[1], reverse=True)

    for value, count in sorted_items:
        percentage = (count / total * 100) if total > 0 else 0
        table.add_row(str(value), f"{count:,}", f"{percentage:.1f}%")

    return table


async def main():
    """Main analytics function."""
    console = Console()

    console.print("[bold blue]🚀 TypeScript-Style Analytics in Python[/bold blue]")
    console.print("=" * 55)

    try:
        with StreamingIngester() as ingester:
            # TypeScript-style parallel aggregation
            console.print("\n[bold yellow]⚡ Running Parallel Aggregations...[/bold yellow]")

            document_type_stats, country_stats = await get_parallel_aggregations(ingester)

            # Display results in formatted tables
            console.print("\n")
            console.print(
                format_aggregation_results(document_type_stats, "📋 Document Type Distribution")
            )

            console.print("\n")
            console.print(format_aggregation_results(country_stats, "🌍 Country Distribution"))

            # Additional aggregations (like your expandable TypeScript queries)
            console.print("\n[bold yellow]📊 Additional Analytics:[/bold yellow]")

            language_stats = ingester.get_language_stats()
            court_stats = ingester.get_court_stats()

            console.print("\n")
            console.print(format_aggregation_results(language_stats, "🌐 Language Distribution"))

            # Show top 10 courts only
            if court_stats:
                top_courts = dict(
                    sorted(court_stats.items(), key=lambda x: x[1], reverse=True)[:10]
                )
                console.print("\n")
                console.print(format_aggregation_results(top_courts, "⚖️  Top 10 Courts"))

            # Summary statistics (like TypeScript aggregate totals)
            console.print("\n[bold yellow]📈 Summary Statistics:[/bold yellow]")

            total_documents = sum(document_type_stats.values())
            total_countries = len(country_stats)
            total_languages = len(language_stats)
            total_courts = len(court_stats)

            summary_table = Table(
                title="Collection Summary", show_header=True, header_style="bold cyan"
            )
            summary_table.add_column("Metric", style="white")
            summary_table.add_column("Value", style="green", justify="right")

            summary_table.add_row("Total Documents", f"{total_documents:,}")
            summary_table.add_row("Unique Countries", f"{total_countries}")
            summary_table.add_row("Languages", f"{total_languages}")
            summary_table.add_row("Courts", f"{total_courts}")

            console.print("\n")
            console.print(summary_table)

            # TypeScript-style data access examples
            console.print("\n[bold yellow]💻 TypeScript-Style Data Access:[/bold yellow]")

            if document_type_stats:
                most_common_type = max(document_type_stats.items(), key=lambda x: x[1])
                console.print(
                    f"[green]Most common document type:[/green] {most_common_type[0]} ({most_common_type[1]:,} docs)"
                )

            if country_stats:
                most_common_country = max(country_stats.items(), key=lambda x: x[1])
                console.print(
                    f"[green]Most common country:[/green] {most_common_country[0]} ({most_common_country[1]:,} docs)"
                )

            # Equivalent to TypeScript filtering/mapping
            judgment_types = {
                k: v for k, v in document_type_stats.items() if "judgment" in k.lower()
            }
            if judgment_types:
                console.print(
                    f"[green]Judgment documents:[/green] {sum(judgment_types.values()):,}"
                )

            # Show percentage distribution (like TypeScript calculations)
            console.print("\n[bold yellow]📊 Percentage Breakdown:[/bold yellow]")
            for doc_type, count in sorted(
                document_type_stats.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total_documents * 100) if total_documents > 0 else 0
                console.print(f"  • {doc_type}: {percentage:.1f}% ({count:,} documents)")

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        console.print("\n[yellow]💡 Troubleshooting:[/yellow]")
        console.print("1. Make sure Weaviate is running: docker-compose up -d")
        console.print("2. Ensure you have documents ingested in the collection")
        console.print("3. Check Weaviate connection settings")


if __name__ == "__main__":
    asyncio.run(main())
