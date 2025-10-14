"""Check coverage for all properties in Weaviate LegalDocuments collection."""

from collections import defaultdict
from typing import Dict

from loguru import logger
from rich.console import Console
from rich.table import Table

from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase

console = Console()


def check_property_coverage(db: WeaviateLegalDocumentsDatabase, sample_size: int = 10000) -> Dict:
    """Check coverage for all properties in the LegalDocuments collection.

    Args:
        db: Weaviate database connection
        sample_size: Number of documents to sample (max 10K due to offset limit)

    Returns:
        Dict with coverage statistics for all properties
    """
    collection = db.legal_documents_collection

    # Get total count
    logger.info("Getting total document count...")
    total_response = collection.aggregate.over_all(total_count=True)
    total_count = total_response.total_count

    logger.info(f"Total documents in collection: {total_count:,}")
    logger.info(f"Sampling {sample_size:,} documents for property coverage analysis...")

    # Count populated fields
    field_counts = defaultdict(int)
    field_types = {}
    documents_checked = 0
    batch_size = 1000
    offset = 0

    # Limit to sample_size or 10K (Weaviate offset limit)
    max_offset = min(sample_size, 10000)

    while offset < max_offset:
        try:
            # Fetch batch of documents
            response = collection.query.fetch_objects(
                limit=batch_size,
                offset=offset
            )

            for obj in response.objects:
                documents_checked += 1
                properties = obj.properties

                # Check each property
                for field, value in properties.items():
                    # Check if field has meaningful content
                    is_populated = False

                    if value is not None:
                        if isinstance(value, str):
                            if value.strip():  # Non-empty string
                                is_populated = True
                        elif isinstance(value, list):
                            if value:  # Non-empty list
                                is_populated = True
                        elif isinstance(value, (int, float, bool)):
                            is_populated = True
                        else:
                            is_populated = True

                    if is_populated:
                        field_counts[field] += 1

                    # Track field type (from first occurrence)
                    if field not in field_types:
                        field_types[field] = type(value).__name__ if value is not None else "None"

            offset += batch_size
            logger.info(f"Processed {documents_checked:,} documents...")

        except Exception as e:
            logger.error(f"Error fetching documents at offset {offset}: {e}")
            break

    # Calculate coverage statistics
    field_stats = {}

    for field in sorted(field_counts.keys()):
        count = field_counts[field]
        coverage = (count / documents_checked * 100) if documents_checked > 0 else 0.0
        empty_count = documents_checked - count

        field_stats[field] = {
            "populated_count": count,
            "empty_count": empty_count,
            "coverage_percentage": round(coverage, 2),
            "type": field_types.get(field, "unknown"),
        }

    return {
        "total_documents": total_count,
        "sample_size": documents_checked,
        "field_stats": field_stats,
    }


def print_results(stats: Dict):
    """Print coverage statistics in a formatted table."""

    console.print("\n[bold cyan]Property Coverage Analysis[/bold cyan]\n")
    console.print(f"[bold]Total documents in collection:[/bold] {stats['total_documents']:,}")
    console.print(f"[bold]Sample size analyzed:[/bold] {stats['sample_size']:,}")
    console.print(f"[dim]Note: Sample limited to 10,000 documents due to Weaviate offset limit[/dim]\n")

    # Create table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", width=35)
    table.add_column("Type", style="blue", width=15)
    table.add_column("Populated", justify="right", style="green", width=12)
    table.add_column("Empty", justify="right", style="red", width=12)
    table.add_column("Coverage %", justify="right", style="yellow", width=12)

    # Sort by coverage percentage (ascending) to show most empty fields first
    sorted_fields = sorted(
        stats["field_stats"].items(),
        key=lambda x: x[1]["coverage_percentage"]
    )

    for field, field_stat in sorted_fields:
        coverage_pct = field_stat["coverage_percentage"]

        # Color code based on coverage
        if coverage_pct > 80:
            coverage_color = "green"
        elif coverage_pct > 50:
            coverage_color = "yellow"
        elif coverage_pct > 10:
            coverage_color = "orange1"
        else:
            coverage_color = "red"

        table.add_row(
            field,
            field_stat["type"],
            f"{field_stat['populated_count']:,}",
            f"{field_stat['empty_count']:,}",
            f"[{coverage_color}]{coverage_pct:.2f}%[/{coverage_color}]",
        )

    console.print(table)

    # Print summary statistics
    console.print("\n[bold cyan]Summary Statistics[/bold cyan]\n")

    # Count properties by coverage ranges
    coverage_ranges = {
        "0-10%": 0,
        "10-50%": 0,
        "50-80%": 0,
        "80-100%": 0,
    }

    for field_stat in stats["field_stats"].values():
        coverage_pct = field_stat["coverage_percentage"]
        if coverage_pct <= 10:
            coverage_ranges["0-10%"] += 1
        elif coverage_pct <= 50:
            coverage_ranges["10-50%"] += 1
        elif coverage_pct <= 80:
            coverage_ranges["50-80%"] += 1
        else:
            coverage_ranges["80-100%"] += 1

    for range_name, count in coverage_ranges.items():
        console.print(f"Properties with {range_name} coverage: [bold]{count}[/bold]")

    # Highlight extremely sparse fields (< 1% coverage)
    sparse_fields = [
        field for field, stat in stats["field_stats"].items()
        if stat["coverage_percentage"] < 1.0
    ]

    if sparse_fields:
        console.print(f"\n[bold red]Very sparse properties (< 1% coverage):[/bold red]")
        for field in sparse_fields:
            stat = stats["field_stats"][field]
            console.print(
                f"  • {field}: {stat['populated_count']:,} / {stats['sample_size']:,} "
                f"({stat['coverage_percentage']:.2f}%)"
            )


def main():
    """Main execution function."""
    logger.info("Connecting to Weaviate...")

    with WeaviateLegalDocumentsDatabase() as db:
        try:
            stats = check_property_coverage(db, sample_size=10000)
            print_results(stats)

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise


if __name__ == "__main__":
    main()
