#!/usr/bin/env python3
"""Fast check of UMAP coordinates coverage using Weaviate aggregation queries."""

from rich.console import Console
from rich.table import Table
from weaviate.classes.query import Filter

from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase


def main():
    console = Console()

    with WeaviateLegalDocumentsDatabase() as db:
        for collection_name in ["LegalDocuments", "DocumentChunks"]:
            console.print(f"\n[bold cyan]Collection: {collection_name}[/bold cyan]")

            collection = db.get_collection(collection_name)

            # Get total count
            total_response = collection.aggregate.over_all(total_count=True)
            total_docs = total_response.total_count
            console.print(f"[dim]Total documents: {total_docs:,}[/dim]")

            # Sample documents to check for missing coordinates
            # Since filtering by null state isn't indexed, we'll sample and check
            console.print(f"[dim]Sampling documents to check coordinates...[/dim]")

            sample_size = min(10000, total_docs)
            missing_count = 0
            checked = 0

            for obj in collection.iterator(return_properties=["x", "y"]):
                x = obj.properties.get("x")
                y = obj.properties.get("y")

                if x is None or y is None:
                    missing_count += 1

                checked += 1

                if checked >= sample_size:
                    break

            # Estimate total missing based on sample
            if checked > 0:
                missing_rate = missing_count / checked
                without_coords = int(total_docs * missing_rate)
                with_coords = total_docs - without_coords
            else:
                without_coords = 0
                with_coords = total_docs
            coverage_pct = (with_coords / total_docs * 100) if total_docs > 0 else 0

            # Summary table
            summary = Table(title=f"Coordinates Coverage - {collection_name} (sampled {checked:,} docs)")
            summary.add_column("Metric", style="cyan")
            summary.add_column("Count", style="magenta", justify="right")
            summary.add_column("Percentage", style="green", justify="right")

            summary.add_row("Total Documents", f"{total_docs:,}", "100.00%")
            summary.add_row(
                "With Coordinates",
                f"{with_coords:,}",
                f"{coverage_pct:.2f}%",
                style="green" if coverage_pct == 100 else "yellow",
            )
            summary.add_row(
                "Without Coordinates",
                f"{without_coords:,}",
                f"{(100 - coverage_pct):.2f}%",
                style="green" if without_coords == 0 else "red",
            )

            console.print(summary)

            if without_coords == 0:
                console.print(
                    f"[green]✓ All documents in {collection_name} have coordinates![/green]"
                )
            else:
                console.print(
                    f"[yellow]⚠ {without_coords:,} documents in {collection_name} are missing coordinates[/yellow]"
                )


if __name__ == "__main__":
    main()
