"""Check if all x and y coordinates are set in Weaviate legal documents collection."""

import os

# Override Weaviate host for external access BEFORE any imports
os.environ["WEAVIATE_HOST"] = os.getenv("WWEAVIATE_URL", "83.238.160.206")
os.environ["WEAVIATE_PORT"] = "8080"

from loguru import logger
from rich.console import Console
from rich.table import Table

from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase


def main() -> None:
    """Check UMAP coordinates coverage in Weaviate."""
    console = Console()

    with WeaviateLegalDocumentsDatabase() as db:
        collection = db.legal_documents_collection

        # Get total count
        total_response = collection.aggregate.over_all(total_count=True)
        total_docs = total_response.total_count

        logger.info(f"Total documents: {total_docs}")

        # Count documents with both x and y coordinates
        import weaviate.classes.query as wq

        # Documents with x coordinate set (not null)
        with_x_response = collection.aggregate.over_all(
            total_count=True,
            filters=wq.Filter.by_property("x").is_none(False),
        )
        with_x = with_x_response.total_count

        # Documents with y coordinate set (not null)
        with_y_response = collection.aggregate.over_all(
            total_count=True,
            filters=wq.Filter.by_property("y").is_none(False),
        )
        with_y = with_y_response.total_count

        # Documents with both x and y
        with_both_response = collection.aggregate.over_all(
            total_count=True,
            filters=wq.Filter.by_property("x").is_none(False) & wq.Filter.by_property("y").is_none(False),
        )
        with_both = with_both_response.total_count

        # Documents without x
        without_x = total_docs - with_x

        # Documents without y
        without_y = total_docs - with_y

        # Documents missing at least one coordinate
        missing_coords = total_docs - with_both

        # Create results table
        table = Table(title="UMAP Coordinates Coverage")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Percentage", style="green")

        table.add_row("Total Documents", str(total_docs), "100%")
        table.add_row(
            "With X Coordinate",
            str(with_x),
            f"{(with_x / total_docs * 100):.2f}%" if total_docs > 0 else "0%",
        )
        table.add_row(
            "With Y Coordinate",
            str(with_y),
            f"{(with_y / total_docs * 100):.2f}%" if total_docs > 0 else "0%",
        )
        table.add_row(
            "With Both X and Y",
            str(with_both),
            f"{(with_both / total_docs * 100):.2f}%" if total_docs > 0 else "0%",
        )
        table.add_row(
            "Without X Coordinate",
            str(without_x),
            f"{(without_x / total_docs * 100):.2f}%" if total_docs > 0 else "0%",
        )
        table.add_row(
            "Without Y Coordinate",
            str(without_y),
            f"{(without_y / total_docs * 100):.2f}%" if total_docs > 0 else "0%",
        )
        table.add_row(
            "Missing Any Coordinate",
            str(missing_coords),
            f"{(missing_coords / total_docs * 100):.2f}%" if total_docs > 0 else "0%",
            style="yellow" if missing_coords > 0 else "green",
        )

        console.print(table)

        # Summary
        if missing_coords == 0:
            console.print("\n[green]✓ All documents have UMAP coordinates set![/green]")
        else:
            console.print(f"\n[yellow]⚠ {missing_coords} documents are missing UMAP coordinates[/yellow]")


if __name__ == "__main__":
    main()
