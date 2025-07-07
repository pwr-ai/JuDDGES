#!/usr/bin/env python3
"""
Example script demonstrating collection analytics using StreamingIngester aggregation methods.

This script shows how to use the built-in aggregation methods to analyze
your legal document collection in Weaviate.
"""

from rich.console import Console

from juddges.data.stream_ingester import StreamingIngester


def main():
    """Run collection analytics."""
    console = Console()

    console.print("[bold blue]🔍 Legal Document Collection Analytics[/bold blue]")
    console.print("=" * 50)

    try:
        # Initialize the ingester (connects to Weaviate)
        with StreamingIngester() as ingester:
            # Option 1: Get individual statistics
            console.print("\n[bold yellow]📊 Individual Statistics:[/bold yellow]")

            # Document types
            doc_types = ingester.get_document_type_stats()
            console.print(f"Document types: {doc_types}")

            # Countries
            countries = ingester.get_country_stats()
            console.print(f"Countries: {countries}")

            # Languages
            languages = ingester.get_language_stats()
            console.print(f"Languages: {languages}")

            # Option 2: Get comprehensive statistics
            console.print("\n[bold yellow]📈 Comprehensive Analysis:[/bold yellow]")
            comprehensive_stats = ingester.get_comprehensive_stats()

            # Option 3: Print formatted summary
            console.print("\n[bold yellow]📋 Formatted Summary:[/bold yellow]")
            ingester.print_collection_summary()

            # Option 4: Access specific aggregation data
            console.print("\n[bold yellow]🎯 Specific Analysis Examples:[/bold yellow]")

            if comprehensive_stats.get("document_types"):
                most_common_type = max(
                    comprehensive_stats["document_types"].items(), key=lambda x: x[1]
                )
                console.print(
                    f"Most common document type: {most_common_type[0]} ({most_common_type[1]:,} documents)"
                )

            if comprehensive_stats.get("countries"):
                total_countries = len(comprehensive_stats["countries"])
                console.print(f"Documents span {total_countries} countries")

            if comprehensive_stats.get("languages"):
                languages_list = list(comprehensive_stats["languages"].keys())
                console.print(f"Available languages: {', '.join(languages_list)}")

            # Date range analysis
            date_stats = ingester.get_date_range_stats()
            if date_stats.get("total_with_dates", 0) > 0:
                console.print(
                    f"Document date range: {date_stats['earliest_date']} to {date_stats['latest_date']}"
                )

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        console.print(
            "\n[yellow]💡 Make sure Weaviate is running and you have documents ingested.[/yellow]"
        )
        console.print("[dim]Start Weaviate: docker-compose up -d (in weaviate/ directory)[/dim]")


if __name__ == "__main__":
    main()
