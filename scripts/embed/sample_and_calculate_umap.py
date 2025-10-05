#!/usr/bin/env python3
"""
Sample documents from Weaviate, calculate UMAP coordinates, and update the database.

This script:
1. Samples up to 25k documents per unique value for 'country' and 'source_url' from Weaviate
2. Saves sampled documents with vectors to parquet files
3. Fits UMAP on the sampled data and saves the UMAP model to models/umap/
4. Calculates 2D coordinates for sampled documents
5. Updates Weaviate with the calculated coordinates
6. Applies UMAP model to remaining documents (optional)

Usage:
    # Sample, fit UMAP, and update sampled documents
    docker compose run --rm web python scripts/embed/sample_and_calculate_umap.py \
        --output-dir data/embeddings_samples \
        --sample-size 25000 \
        --collection both

    # Apply saved UMAP model to all remaining documents
    docker compose run --rm web python scripts/embed/sample_and_calculate_umap.py \
        --apply-saved-model models/umap/umap_model_LegalDocuments.pkl \
        --collection LegalDocuments
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from sklearn.preprocessing import normalize
from umap import UMAP

from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase
from juddges.settings import UMAP_MODELS_PATH, VectorName


def sample_documents_by_strata(
    db: WeaviateLegalDocumentsDatabase,
    collection_name: str,
    sample_size: int,
    vector_name: str,
    console: Console,
) -> Dict[str, List[Dict]]:
    """Sample documents from Weaviate stratified by appropriate fields.

    For LegalDocuments: stratifies by country and source_url
    For DocumentChunks: stratifies by language and document_type

    Args:
        db: Weaviate database instance
        collection_name: Name of collection to sample from
        sample_size: Maximum number of documents per unique value
        vector_name: Name of vector to retrieve
        console: Rich console for output

    Returns:
        Dictionary with strata as keys and list of documents as values
    """
    console.print(f"[bold cyan]Sampling documents from {collection_name}...[/bold cyan]")

    collection = db.get_collection(collection_name)

    # Determine strata fields based on collection
    is_chunks = collection_name == "DocumentChunks"

    if is_chunks:
        strata_fields = ["language", "document_type"]
        console.print("[dim]Fetching unique strata (language, document_type)...[/dim]")
    else:
        strata_fields = ["country", "source_url"]
        console.print("[dim]Fetching unique strata (country, source_url)...[/dim]")

    # Get all documents with strata fields using iterator (more efficient than offset pagination)
    strata_counts = defaultdict(int)
    strata_docs = defaultdict(list)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning documents...", total=None)

        # Use iterator for efficient streaming (avoids timeout issues)
        for obj in collection.iterator(
            include_vector=True,
            return_properties=strata_fields,
        ):
            # Extract strata values based on collection type
            if is_chunks:
                field1 = obj.properties.get("language", "unknown")
                field2 = obj.properties.get("document_type", "unknown")
            else:
                field1 = obj.properties.get("country", "unknown")
                field2 = obj.properties.get("source_url", "unknown")

            stratum = f"{field1}|{field2}"

            # Only add if we haven't reached sample_size for this stratum
            if strata_counts[stratum] < sample_size:
                # Get the vector
                vector = obj.vector.get(vector_name) if obj.vector else None

                if vector is not None:
                    doc_data = {
                        "uuid": str(obj.uuid),
                        strata_fields[0]: field1,
                        strata_fields[1]: field2,
                        "vector": vector,
                    }
                    strata_docs[stratum].append(doc_data)
                    strata_counts[stratum] += 1

            progress.update(task, advance=1)

            # Stop if all strata are full (optimization to avoid scanning entire collection)
            if strata_counts and all(count >= sample_size for count in strata_counts.values()):
                console.print("[green]All strata reached sample size, stopping early[/green]")
                break

    # Display strata statistics
    strata_label = "Language|Type" if is_chunks else "Country|Source"
    strata_table = Table(title=f"Sampled Strata from {collection_name}")
    strata_table.add_column(f"Stratum ({strata_label})", style="cyan")
    strata_table.add_column("Count", style="yellow", justify="right")

    total_sampled = 0
    for stratum, docs in sorted(strata_docs.items()):
        strata_table.add_row(stratum, str(len(docs)))
        total_sampled += len(docs)

    console.print(strata_table)
    console.print(f"[green]✓[/green] Total sampled: {total_sampled} documents")

    return strata_docs


def save_sampled_data(
    strata_docs: Dict[str, List[Dict]],
    output_dir: Path,
    collection_name: str,
    console: Console,
) -> Path:
    """Save sampled documents with vectors to parquet file.

    Args:
        strata_docs: Dictionary of strata to documents
        output_dir: Directory to save the parquet file
        collection_name: Name of collection
        console: Rich console for output

    Returns:
        Path to saved parquet file
    """
    console.print(f"[bold cyan]Saving sampled data to parquet...[/bold cyan]")

    # Flatten the strata_docs into a list
    all_docs = []
    for stratum, docs in strata_docs.items():
        all_docs.extend(docs)

    if not all_docs:
        raise ValueError("No documents to save")

    # Get field names from first document (excluding uuid and vector)
    field_names = [key for key in all_docs[0].keys() if key not in ["uuid", "vector"]]

    # Create DataFrame with dynamic fields
    df_data = {
        "uuid": [doc["uuid"] for doc in all_docs],
        "vector": [doc["vector"] for doc in all_docs],
    }

    # Add strata fields
    for field_name in field_names:
        df_data[field_name] = [doc[field_name] for doc in all_docs]

    df = pl.DataFrame(df_data)

    # Save to parquet
    output_path = output_dir / f"{collection_name}_sampled.parquet"
    df.write_parquet(output_path)

    console.print(f"[green]✓[/green] Saved {len(all_docs)} documents to {output_path}")

    return output_path


def fit_umap_and_calculate_coords(
    vectors: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    console: Console,
) -> Tuple[UMAP, np.ndarray]:
    """Fit UMAP model and calculate 2D coordinates.

    Args:
        vectors: Array of embeddings
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        console: Rich console for output

    Returns:
        Tuple of (fitted UMAP model, 2D coordinates)
    """
    console.print("[bold cyan]Normalizing vectors...[/bold cyan]")
    normalized_vectors = normalize(vectors, norm="l2", axis=1)

    norms = np.linalg.norm(normalized_vectors, axis=1)
    console.print(
        f"[green]✓[/green] Normalized {len(normalized_vectors)} vectors "
        f"(mean norm: {norms.mean():.6f}, std: {norms.std():.6f})"
    )

    console.print("[bold cyan]Fitting UMAP model...[/bold cyan]")
    console.print(
        f"[dim]Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric=cosine[/dim]"
    )

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
        n_components=2,
        verbose=True,
    )

    coords = umap_model.fit_transform(normalized_vectors)

    console.print(f"[green]✓[/green] Computed UMAP coordinates with shape {coords.shape}")
    console.print(
        f"[dim]X range: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
        f"Y range: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}][/dim]"
    )

    return umap_model, coords


def save_umap_model(
    umap_model: UMAP,
    model_name: str,
    console: Console,
) -> Path:
    """Save UMAP model to pickle file.

    Args:
        umap_model: Fitted UMAP model
        model_name: Name for the model file (without extension)
        console: Rich console for output

    Returns:
        Path to saved model file
    """
    UMAP_MODELS_PATH.mkdir(parents=True, exist_ok=True)
    model_path = UMAP_MODELS_PATH / f"{model_name}.pkl"

    console.print(f"[bold cyan]Saving UMAP model to {model_path}...[/bold cyan]")

    with open(model_path, "wb") as f:
        pickle.dump(umap_model, f)

    console.print(f"[green]✓[/green] Saved UMAP model to {model_path}")

    return model_path


def update_weaviate_batch(
    db: WeaviateLegalDocumentsDatabase,
    collection_name: str,
    uuids: List[str],
    coordinates: np.ndarray,
    batch_size: int,
    console: Console,
) -> Tuple[int, int]:
    """Update Weaviate documents with UMAP coordinates in batches.

    Args:
        db: Weaviate database instance
        collection_name: Name of collection to update
        uuids: List of document UUIDs
        coordinates: Array of (x, y) coordinates
        batch_size: Number of documents to update per batch
        console: Rich console for output

    Returns:
        Tuple of (successful_updates, failed_updates)
    """
    collection = db.get_collection(collection_name)
    successful_updates = 0
    failed_updates = 0

    console.print(
        f"[bold cyan]Updating {len(uuids)} documents in batches of {batch_size}...[/bold cyan]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Updating documents...", total=len(uuids))

        for i in range(0, len(uuids), batch_size):
            batch_uuids = uuids[i : i + batch_size]
            batch_coords = coordinates[i : i + batch_size]

            for uuid, coords in zip(batch_uuids, batch_coords):
                try:
                    collection.data.update(
                        uuid=uuid,
                        properties={
                            "x": float(coords[0]),
                            "y": float(coords[1]),
                        },
                    )
                    successful_updates += 1
                except Exception as e:
                    logger.error(f"Failed to update document {uuid}: {e}")
                    failed_updates += 1

                progress.update(task, advance=1)

    console.print(
        f"[green]✓[/green] Updated {successful_updates} documents "
        f"([red]{failed_updates} failed[/red])"
    )

    return successful_updates, failed_updates


def apply_saved_umap_to_all(
    db: WeaviateLegalDocumentsDatabase,
    collection_name: str,
    umap_model_path: Path,
    vector_name: str,
    batch_size: int,
    console: Console,
) -> Tuple[int, int]:
    """Apply saved UMAP model to all documents in Weaviate.

    Args:
        db: Weaviate database instance
        collection_name: Name of collection
        umap_model_path: Path to saved UMAP model
        vector_name: Name of vector to use
        batch_size: Batch size for updates
        console: Rich console for output

    Returns:
        Tuple of (successful_updates, failed_updates)
    """
    console.print(f"[bold cyan]Loading UMAP model from {umap_model_path}...[/bold cyan]")

    with open(umap_model_path, "rb") as f:
        umap_model = pickle.load(f)

    console.print("[green]✓[/green] UMAP model loaded successfully")

    collection = db.get_collection(collection_name)

    # Fetch all documents with vectors using iterator (more efficient, avoids timeout)
    console.print("[bold cyan]Fetching all documents with vectors...[/bold cyan]")

    all_uuids = []
    all_vectors = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching documents...", total=None)

        # Use iterator for efficient streaming
        for obj in collection.iterator(include_vector=True):
            vector = obj.vector.get(vector_name) if obj.vector else None
            if vector is not None:
                all_uuids.append(str(obj.uuid))
                all_vectors.append(vector)

            progress.update(task, advance=1)

    console.print(f"[green]✓[/green] Fetched {len(all_uuids)} documents with vectors")

    # Normalize and transform
    console.print("[bold cyan]Applying UMAP transformation...[/bold cyan]")
    vectors_array = np.vstack(all_vectors)
    normalized_vectors = normalize(vectors_array, norm="l2", axis=1)
    coords = umap_model.transform(normalized_vectors)

    console.print(f"[green]✓[/green] Transformed {len(coords)} vectors to 2D coordinates")

    # Update Weaviate
    successful, failed = update_weaviate_batch(
        db, collection_name, all_uuids, coords, batch_size, console
    )

    return successful, failed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sample documents from Weaviate, fit UMAP, and update coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/embeddings_samples",
        help="Directory to save sampled data and UMAP model (default: data/embeddings_samples)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2500,
        help="Maximum number of documents per unique country/source_url value (default: 2500)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        choices=["LegalDocuments", "DocumentChunks", "both"],
        default="both",
        help="Collection to process (default: both)",
    )
    parser.add_argument(
        "--vector-name",
        type=str,
        default=VectorName.BASE,
        help=f"Name of vector to use (default: {VectorName.BASE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for Weaviate updates (default: 500)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter (default: 15)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (default: 0.1)",
    )
    parser.add_argument(
        "--apply-saved-model",
        type=str,
        help="Path to saved UMAP model to apply to all documents",
    )

    args = parser.parse_args()

    console = Console()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Display configuration
    config_text = "[bold blue]UMAP Sampling and Calculation[/bold blue]\n\n"

    if args.apply_saved_model:
        config_text += f"Mode: Apply saved UMAP model\n"
        config_text += f"Model Path: {args.apply_saved_model}\n"
    else:
        config_text += f"Mode: Sample and fit new UMAP\n"
        config_text += f"Sample Size: {args.sample_size} per stratum\n"
        config_text += f"UMAP n_neighbors: {args.n_neighbors}\n"
        config_text += f"UMAP min_dist: {args.min_dist}\n"

    config_text += f"Collection: {args.collection}\n"
    config_text += f"Vector Name: {args.vector_name}\n"
    config_text += f"Output Dir: {output_dir}\n"
    config_text += f"Batch Size: {args.batch_size}\n"

    console.print(Panel.fit(config_text, title="Configuration", border_style="blue"))

    # Determine collections to process
    if args.collection == "both":
        collections = ["LegalDocuments", "DocumentChunks"]
    else:
        collections = [args.collection]

    with WeaviateLegalDocumentsDatabase() as db:
        for collection_name in collections:
            console.print(f"\n[bold magenta]{'=' * 60}[/bold magenta]")
            console.print(f"[bold magenta]Processing: {collection_name}[/bold magenta]")
            console.print(f"[bold magenta]{'=' * 60}[/bold magenta]\n")

            try:
                if args.apply_saved_model:
                    # Apply saved UMAP model to all documents
                    model_path = Path(args.apply_saved_model)
                    if not model_path.exists():
                        raise FileNotFoundError(f"UMAP model not found: {model_path}")

                    successful, failed = apply_saved_umap_to_all(
                        db,
                        collection_name,
                        model_path,
                        args.vector_name,
                        args.batch_size,
                        console,
                    )

                    console.print(
                        f"[bold green]✓ Applied UMAP to {successful} documents "
                        f"({failed} failed)[/bold green]"
                    )

                else:
                    # Sample documents
                    strata_docs = sample_documents_by_strata(
                        db,
                        collection_name,
                        args.sample_size,
                        args.vector_name,
                        console,
                    )

                    if not strata_docs:
                        console.print(
                            f"[bold yellow]No documents sampled for {collection_name}, skipping[/bold yellow]"
                        )
                        continue

                    # Save sampled data
                    parquet_path = save_sampled_data(
                        strata_docs, output_dir, collection_name, console
                    )

                    # Extract vectors for UMAP
                    all_docs = []
                    for docs in strata_docs.values():
                        all_docs.extend(docs)

                    uuids = [doc["uuid"] for doc in all_docs]
                    vectors = np.vstack([doc["vector"] for doc in all_docs])

                    # Fit UMAP and calculate coordinates
                    umap_model, coords = fit_umap_and_calculate_coords(
                        vectors, args.n_neighbors, args.min_dist, console
                    )

                    # Save UMAP model (only once, for the first collection)
                    if collection_name == collections[0]:
                        model_path = save_umap_model(
                            umap_model, f"umap_model_{collection_name}", console
                        )

                    # Update Weaviate
                    successful, failed = update_weaviate_batch(
                        db, collection_name, uuids, coords, args.batch_size, console
                    )

                    # Summary
                    summary_table = Table(title=f"{collection_name} Summary")
                    summary_table.add_column("Metric", style="cyan")
                    summary_table.add_column("Count", style="yellow", justify="right")

                    summary_table.add_row("Total Sampled", str(len(uuids)))
                    summary_table.add_row("Strata", str(len(strata_docs)))
                    summary_table.add_row("UMAP Coords Computed", str(len(coords)))
                    summary_table.add_row("Documents Updated", str(successful))
                    summary_table.add_row("Failed Updates", str(failed))

                    console.print(summary_table)

            except Exception as e:
                console.print(f"[bold red]Error processing {collection_name}: {e}[/bold red]")
                logger.exception(f"Failed to process {collection_name}")
                continue

    console.print(
        Panel.fit(
            "[bold green]✓ UMAP processing completed![/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
