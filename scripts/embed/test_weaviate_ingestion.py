import os

import typer
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from juddges.data.weaviate_db import WeaviateJudgementsDatabase
from juddges.settings import ROOT_PATH

console = Console()


def print_schema(db: WeaviateJudgementsDatabase) -> None:
    """Print the schema of both collections."""
    console.print("\n[bold blue]Weaviate Schema[/bold blue]")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}")
    ) as progress:
        task = progress.add_task("Fetching schema...", start=False)
        progress.start_task(task)

        for collection_name in [db.JUDGMENTS_COLLECTION, db.JUDGMENT_CHUNKS_COLLECTION]:
            collection = db.client.collections.get(collection_name)
            schema = collection.config.get()

            table = Table(title=f"Collection: {collection_name}")
            table.add_column("Property", style="cyan")
            table.add_column("Data Type", style="magenta")

            for prop in schema.properties:
                table.add_row(prop.name, prop.data_type.value)

            console.print(table)

        progress.stop_task(task)


def print_collection_stats(db: WeaviateJudgementsDatabase) -> None:
    """Print statistics about the collections."""
    console.print("\n[bold blue]Collection Statistics[/bold blue]")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}")
    ) as progress:
        task = progress.add_task("Calculating statistics...", start=False)
        progress.start_task(task)

        console.print("[green]Fetching judgements collection...[/green]")
        judgements_collection = db.client.collections.get(db.JUDGMENTS_COLLECTION)
        judgements_count = len(judgements_collection)
        console.print(
            f"[green]Judgements collection fetched. Count: {judgements_count:,}[/green]"
        )

        console.print("[green]Fetching chunks collection...[/green]")
        chunks_collection = db.client.collections.get(db.JUDGMENT_CHUNKS_COLLECTION)
        chunks_count = len(chunks_collection)
        console.print(
            f"[green]Chunks collection fetched. Count: {chunks_count:,}[/green]"
        )

        table = Table(title="Document Counts")
        table.add_column("Collection", style="cyan")
        table.add_column("Count", style="magenta")

        table.add_row(db.JUDGMENTS_COLLECTION, f"{judgements_count:,}")
        table.add_row(db.JUDGMENT_CHUNKS_COLLECTION, f"{chunks_count:,}")

        console.print(table)

        progress.stop_task(task)


def run_sample_queries(db: WeaviateJudgementsDatabase) -> None:
    """Run sample queries to verify search functionality."""
    console.print("\n[bold blue]Sample Queries[/bold blue]")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}")
    ) as progress:
        task = progress.add_task("Running sample queries...", start=False)
        progress.start_task(task)

        # Query on judgements collection
        judgements = db.judgements_collection
        response = judgements.query.fetch_objects(limit=3)

        console.print("[yellow]Sample Judgements:[/yellow]")
        for obj in response.objects:
            print(obj.properties)

        # Semantic search on chunks
        chunks = db.judgement_chunks_collection
        query = "Sprawa dotyczy narkotyków"  # Case involves drugs
        response = chunks.query.hybrid(query=query, limit=3)

        console.print("\n[yellow]Sample Semantic Search Results:[/yellow]")
        console.print(f"Query: {query}")
        for obj in response.objects:
            print(obj.properties)

        progress.stop_task(task)


def main() -> None:
    """
    End-to-end test for Weaviate ingestion process.
    Verifies schema, counts documents, and runs sample queries.
    """
    load_dotenv(ROOT_PATH / ".env", override=True)

    console.print("[bold green]Starting Weaviate Ingestion Test[/bold green]")
    console.print(
        f"Connecting to Weaviate at {os.environ['WV_URL']}:{os.environ['WV_PORT']} (gRPC: {os.environ['WV_GRPC_PORT']})"
    )

    with WeaviateJudgementsDatabase(
        os.environ["WV_URL"],
        os.environ["WV_PORT"],
        os.environ["WV_GRPC_PORT"],
        os.environ["WV_API_KEY"],
    ) as db:
        # Print schema
        print_schema(db)

        # Print collection statistics
        print_collection_stats(db)

        # Run sample queries
        run_sample_queries(db)


if __name__ == "__main__":
    typer.run(main)
