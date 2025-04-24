import os

import typer
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from sentence_transformers import SentenceTransformer

from ingest_to_weaviate import EnhancedWeaviateDB
from juddges.settings import ROOT_PATH

console = Console()


def print_collections(db: EnhancedWeaviateDB) -> None:
    """Print all collections in the database."""
    console.print("\n[bold blue]All Collections[/bold blue]")
    collections = db.client.collections.list_all()

    table = Table(title="Collections")
    table.add_column("Collection Name", style="cyan")

    for collection in collections:
        table.add_row(collection)

    console.print(table)


def print_schema(db: EnhancedWeaviateDB) -> None:
    """Print the schema of both collections."""
    console.print("\n[bold blue]Weaviate Schema[/bold blue]")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}")
    ) as progress:
        task = progress.add_task("Fetching schema...", start=False)
        progress.start_task(task)

        for collection_name in [db.LEGAL_DOCUMENTS_COLLECTION, db.DOCUMENT_CHUNKS_COLLECTION]:
            collection = db.client.collections.get(collection_name)
            schema = collection.config.get()

            table = Table(title=f"Collection: {collection_name}")
            table.add_column("Property", style="cyan")
            table.add_column("Data Type", style="magenta")

            for prop in schema.properties:
                table.add_row(prop.name, prop.data_type.value)

            console.print(table)

        progress.stop_task(task)


def print_collection_stats(db: EnhancedWeaviateDB) -> None:
    """Print statistics about the collections."""
    console.print("\n[bold blue]Collection Statistics[/bold blue]")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}")
    ) as progress:
        task = progress.add_task("Calculating statistics...", start=False)
        progress.start_task(task)

        console.print("[green]Fetching legal documents collection...[/green]")
        legal_docs_collection = db.legal_documents_collection
        legal_docs_count = db.get_collection_size(legal_docs_collection)
        console.print(f"[green]Legal documents collection fetched. Count: {legal_docs_count:,}[/green]")

        console.print("[green]Fetching document chunks collection...[/green]")
        chunks_collection = db.document_chunks_collection
        chunks_count = db.get_collection_size(chunks_collection)
        console.print(f"[green]Document chunks collection fetched. Count: {chunks_count:,}[/green]")

        table = Table(title="Document Counts")
        table.add_column("Collection", style="cyan")
        table.add_column("Count", style="magenta")

        table.add_row(db.LEGAL_DOCUMENTS_COLLECTION, f"{legal_docs_count:,}")
        table.add_row(db.DOCUMENT_CHUNKS_COLLECTION, f"{chunks_count:,}")

        console.print(table)

        progress.stop_task(task)


def run_sample_queries(db: EnhancedWeaviateDB) -> None:
    """Run sample queries to verify search functionality."""
    console.print("\n[bold blue]Sample Queries[/bold blue]")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}")
    ) as progress:
        task = progress.add_task("Running sample queries...", start=False)
        progress.start_task(task)

        # Query on legal documents collection
        legal_docs = db.legal_documents_collection
        response = legal_docs.query.fetch_objects(limit=3)

        console.print("[yellow]Sample Legal Documents:[/yellow]")
        for obj in response.objects:
            assert obj.properties.get("document_id"), "document_id is not set"
            print(obj.properties)

        console.print("\n[yellow]Sample Documents with Vectors:[/yellow]")
        response = legal_docs.query.fetch_objects(limit=3, include_vector=True)
        for obj in response.objects:
            assert obj.properties.get("document_id"), "document_id is not set"
            assert obj.vector is not None and len(obj.vector) > 0, "Vector is empty"
            print(obj.properties)
            print(f"Vector dimensions: {len(obj.vector)}")

        # Hybrid search on legal documents collection
        query = "Sprawa dotyczy narkotyków"  # Case involves drugs
        model = SentenceTransformer("sdadas/mmlw-roberta-large")
        query_vector = model.encode(query).tolist()  # Convert numpy array to list

        response = legal_docs.query.hybrid(
            query=query,
            alpha=0.5,  # Balance between vector and keyword search
            vector=query_vector,
            target_vector="base",
            limit=3,
        )

        console.print("\n[yellow]Sample Hybrid Search Results on Legal Documents:[/yellow]")
        console.print(f"Query: {query}")
        for obj in response.objects:
            assert obj.properties.get("document_id"), "document_id is not set"
            print(obj.properties)

        # Semantic search on chunks
        chunks = db.document_chunks_collection
        response = chunks.query.hybrid(
            query=query,
            alpha=0.5,  # Balance between vector and keyword search
            vector=query_vector,
            target_vector="base",
            limit=3,
        )

        console.print("\n[yellow]Sample Semantic Search Results on Chunks:[/yellow]")
        console.print(f"Query: {query}")
        for obj in response.objects:
            assert obj.properties.get("document_id"), "document_id is not set"
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
        f"Connecting to Weaviate at {os.environ['WV_HOST']}:{os.environ['WV_PORT']} (gRPC: {os.environ['WV_GRPC_PORT']})"
    )

    with EnhancedWeaviateDB(
        os.environ["WV_HOST"],
        os.environ["WV_PORT"],
        os.environ["WV_GRPC_PORT"],
        os.environ.get("WV_API_KEY"),
    ) as db:
        # Print all collections
        print_collections(db)

        # Print schema
        print_schema(db)

        # Print collection statistics
        print_collection_stats(db)

        # Run sample queries
        run_sample_queries(db)


if __name__ == "__main__":
    typer.run(main)
