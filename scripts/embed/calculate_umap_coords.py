#!/usr/bin/env python3
"""
Calculate UMAP 2D coordinates for Weaviate documents and update them in the database.

This script supports two workflows:

Workflow 1: Calculate and save coordinates (two steps)
  Step 1: Calculate and save
    docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
        --embeddings-dir data/embeddings/pl-court-raw-sample \
        --output-dir umap_coords \
        --skip-update

  Step 2: Update Weaviate from saved coordinates
    docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
        --load-from umap_coords/LegalDocuments_coords.parquet \
        --collection LegalDocuments

Workflow 2: Calculate and update in one step
    docker compose run --rm web python scripts/embed/calculate_umap_coords.py \
        --embeddings-dir data/embeddings/pl-court-raw-sample \
        --collection LegalDocuments

The script:
1. Loads embeddings from parquet files OR loads previously calculated coordinates
2. Normalizes vectors using L2 normalization (when calculating)
3. Computes UMAP 2D coordinates (when calculating)
4. Optionally saves coordinates to parquet file (uuid, x, y)
5. Updates documents with (x, y) coordinates in Weaviate (unless --skip-update)
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl
from datasets import load_dataset
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
from juddges.data.utils import generate_deterministic_uuid


@dataclass
class UMAPConfig:
    """Configuration for UMAP computation."""

    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 42
    n_components: int = 2


@dataclass
class ProcessingStats:
    """Statistics for the UMAP coordinate calculation process."""

    total_documents: int = 0
    vectors_extracted: int = 0
    vectors_normalized: int = 0
    umap_computed: int = 0
    documents_updated: int = 0
    failed_updates: int = 0
    skipped_no_vector: int = 0


def load_embeddings_from_parquet(
    embeddings_dir: Path,
    collection_name: str,
    console: Console,
) -> Tuple[List[str], np.ndarray]:
    """Load embeddings from parquet files.

    Args:
        embeddings_dir: Path to directory containing parquet files with embeddings
        collection_name: Name of collection (determines ID field and UUID generation)
        console: Rich console for output

    Returns:
        Tuple of (uuids, embeddings) where embeddings is a numpy array
    """
    console.print(
        f"[bold cyan]Loading embeddings from {embeddings_dir}...[/bold cyan]"
    )

    # Load dataset from parquet files
    dataset = load_dataset("parquet", data_dir=str(embeddings_dir), split="train")

    console.print(f"[green]✓[/green] Loaded {len(dataset)} records from parquet files")

    uuids: List[str] = []
    embeddings: List[np.ndarray] = []

    # Determine ID and chunk fields based on collection
    is_chunks = collection_name == "DocumentChunks"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Processing {len(dataset)} records...", total=len(dataset))

        for row in dataset:
            # Get document ID (try different possible field names)
            doc_id = row.get("judgment_id") or row.get("document_id") or row.get("id")
            if not doc_id:
                logger.warning(f"No document ID found in row: {row.keys()}")
                progress.update(task, advance=1)
                continue

            # For chunks, get chunk_id
            chunk_id = row.get("chunk_id") if is_chunks else None

            # Generate deterministic UUID
            uuid = generate_deterministic_uuid(doc_id, chunk_id)

            # Get embedding (try different possible field names)
            embedding = row.get("embedding") or row.get("embeddings") or row.get("vector")

            if embedding is not None and len(embedding) > 0:
                uuids.append(uuid)
                embeddings.append(np.array(embedding))
            else:
                logger.warning(f"No embedding found for document {doc_id}")

            progress.update(task, advance=1)

    if not embeddings:
        console.print("[bold red]No embeddings extracted![/bold red]")
        return [], np.array([])

    embeddings_array = np.vstack(embeddings)
    console.print(
        f"[green]✓[/green] Extracted {len(uuids)} embeddings with shape {embeddings_array.shape}"
    )

    return uuids, embeddings_array


def normalize_embeddings(embeddings: np.ndarray, console: Console) -> np.ndarray:
    """Normalize embeddings using L2 normalization.

    Args:
        embeddings: Array of embeddings to normalize
        console: Rich console for output

    Returns:
        L2-normalized embeddings
    """
    console.print("[bold cyan]Normalizing embeddings (L2 norm)...[/bold cyan]")

    # Normalize each embedding vector to unit length
    normalized = normalize(embeddings, norm="l2", axis=1)

    # Verify normalization
    norms = np.linalg.norm(normalized, axis=1)
    console.print(
        f"[green]✓[/green] Normalized {len(normalized)} vectors "
        f"(mean norm: {norms.mean():.6f}, std: {norms.std():.6f})"
    )

    return normalized


def compute_umap_coordinates(
    embeddings: np.ndarray, config: UMAPConfig, console: Console
) -> np.ndarray:
    """Compute UMAP 2D coordinates from embeddings.

    Args:
        embeddings: Normalized embeddings
        config: UMAP configuration
        console: Rich console for output

    Returns:
        2D coordinates array of shape (n_samples, 2)
    """
    console.print("[bold cyan]Computing UMAP 2D coordinates...[/bold cyan]")
    console.print(
        f"[dim]Parameters: n_neighbors={config.n_neighbors}, "
        f"min_dist={config.min_dist}, metric={config.metric}[/dim]"
    )

    umap_model = UMAP(
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        metric=config.metric,
        random_state=config.random_state,
        n_components=config.n_components,
        verbose=False,
    )

    coords = umap_model.fit_transform(embeddings)

    console.print(f"[green]✓[/green] Computed UMAP coordinates with shape {coords.shape}")
    console.print(
        f"[dim]X range: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
        f"Y range: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}][/dim]"
    )

    return coords


def update_weaviate_coordinates(
    db: WeaviateLegalDocumentsDatabase,
    collection_name: str,
    uuids: List[str],
    coordinates: np.ndarray,
    batch_size: int,
    console: Console,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Update Weaviate documents with UMAP coordinates.

    Args:
        db: Weaviate database instance
        collection_name: Name of collection to update
        uuids: List of document UUIDs
        coordinates: Array of (x, y) coordinates
        batch_size: Number of documents to update per batch
        console: Rich console for output
        dry_run: If True, preview changes without updating

    Returns:
        Tuple of (successful_updates, failed_updates)
    """
    if dry_run:
        console.print("[bold yellow]DRY RUN MODE - No updates will be made[/bold yellow]")
        # Preview first 5 updates
        preview_table = Table(title="Preview of Updates (first 5)")
        preview_table.add_column("UUID", style="cyan")
        preview_table.add_column("X", style="green")
        preview_table.add_column("Y", style="green")

        for i in range(min(5, len(uuids))):
            preview_table.add_row(
                uuids[i][:16] + "...",
                f"{coordinates[i, 0]:.4f}",
                f"{coordinates[i, 1]:.4f}",
            )

        console.print(preview_table)
        return len(uuids), 0

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
                    # Update only x and y properties
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


def save_coordinates(
    uuids: List[str],
    coordinates: np.ndarray,
    output_path: Path,
    console: Console,
) -> None:
    """Save UMAP coordinates to parquet file.

    Args:
        uuids: List of document/chunk UUIDs
        coordinates: Array of (x, y) coordinates
        output_path: Path to save the parquet file
        console: Rich console for output
    """
    console.print(f"[bold cyan]Saving coordinates to {output_path}...[/bold cyan]")

    # Create DataFrame with UUIDs and coordinates
    df = pl.DataFrame({
        "uuid": uuids,
        "x": coordinates[:, 0].tolist(),
        "y": coordinates[:, 1].tolist(),
    })

    # Save to parquet
    df.write_parquet(output_path)

    console.print(
        f"[green]✓[/green] Saved {len(uuids)} coordinates to {output_path}"
    )


def load_coordinates(
    input_path: Path,
    console: Console,
) -> Tuple[List[str], np.ndarray]:
    """Load UMAP coordinates from parquet file.

    Args:
        input_path: Path to parquet file with saved coordinates
        console: Rich console for output

    Returns:
        Tuple of (uuids, coordinates)
    """
    console.print(f"[bold cyan]Loading coordinates from {input_path}...[/bold cyan]")

    # Read parquet file
    df = pl.read_parquet(input_path)

    uuids = df["uuid"].to_list()
    coordinates = np.column_stack([df["x"].to_numpy(), df["y"].to_numpy()])

    console.print(
        f"[green]✓[/green] Loaded {len(uuids)} coordinates from {input_path}"
    )

    return uuids, coordinates


def display_statistics(stats: ProcessingStats, console: Console) -> None:
    """Display processing statistics."""
    stats_table = Table(title="Processing Statistics", show_header=True)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Count", style="green", justify="right")

    stats_table.add_row("Total Documents", str(stats.total_documents))
    stats_table.add_row("Vectors Extracted", str(stats.vectors_extracted))
    stats_table.add_row("Vectors Normalized", str(stats.vectors_normalized))
    stats_table.add_row("UMAP Coordinates Computed", str(stats.umap_computed))
    stats_table.add_row("Documents Updated", str(stats.documents_updated))
    stats_table.add_row("Failed Updates", str(stats.failed_updates))
    stats_table.add_row("Skipped (No Vector)", str(stats.skipped_no_vector))

    console.print(stats_table)


def main():
    """Main entry point for UMAP coordinate calculation."""
    parser = argparse.ArgumentParser(
        description="Calculate and update UMAP 2D coordinates for Weaviate documents",
        epilog="""
Examples:
  # Step 1: Calculate and save coordinates
  python calculate_umap_coords.py --embeddings-dir data/embeddings/pl-court-raw-sample --output-dir umap_coords

  # Step 2: Update Weaviate from saved coordinates
  python calculate_umap_coords.py --load-from umap_coords/LegalDocuments_coords.parquet --collection LegalDocuments
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        help="Path to directory containing embedding parquet files (e.g., data/embeddings/pl-court-raw-sample)",
    )
    parser.add_argument(
        "--load-from",
        type=str,
        help="Load previously calculated coordinates from parquet file instead of computing them",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save calculated coordinates (optional, creates {output-dir}/{collection}_coords.parquet)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        choices=["LegalDocuments", "DocumentChunks", "both"],
        default="both",
        help="Collection to process (default: both)",
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
        "--skip-update",
        action="store_true",
        help="Only calculate and save coordinates, don't update Weaviate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without updating Weaviate",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.load_from and args.embeddings_dir:
        raise ValueError("Cannot specify both --load-from and --embeddings-dir")

    if not args.load_from and not args.embeddings_dir:
        raise ValueError("Must specify either --embeddings-dir or --load-from")

    if args.load_from and args.collection == "both":
        raise ValueError("When using --load-from, must specify --collection (cannot be 'both')")

    # Validate embeddings directory exists (if specified)
    embeddings_base_dir = None
    if args.embeddings_dir:
        embeddings_base_dir = Path(args.embeddings_dir)
        if not embeddings_base_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_base_dir}")

    console = Console()

    # Prepare output directory if needed
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Display configuration
    mode = "Load from file" if args.load_from else "Calculate from embeddings"
    config_text = f"[bold blue]UMAP Coordinate Calculator[/bold blue]\n\nMode: {mode}\n"

    if args.load_from:
        config_text += f"Load From: {args.load_from}\n"
    else:
        config_text += f"Embeddings Dir: {embeddings_base_dir}\n"
        config_text += f"UMAP n_neighbors: {args.n_neighbors}\n"
        config_text += f"UMAP min_dist: {args.min_dist}\n"

    config_text += f"Collection: {args.collection}\n"

    if output_dir:
        config_text += f"Output Dir: {output_dir}\n"

    if not args.skip_update:
        config_text += f"Batch Size: {args.batch_size}\n"
        config_text += f"Dry Run: {args.dry_run}\n"
    else:
        config_text += "Skip Update: True\n"

    console.print(
        Panel.fit(
            config_text,
            title="Configuration",
            border_style="blue",
        )
    )

    # Determine which collections to process
    if args.collection == "both":
        collections = ["LegalDocuments", "DocumentChunks"]
    else:
        collections = [args.collection]

    umap_config = UMAPConfig(n_neighbors=args.n_neighbors, min_dist=args.min_dist)

    # Process each collection
    for collection_name in collections:
        console.print(f"\n[bold magenta]{'=' * 60}[/bold magenta]")
        console.print(f"[bold magenta]Processing: {collection_name}[/bold magenta]")
        console.print(f"[bold magenta]{'=' * 60}[/bold magenta]\n")

        stats = ProcessingStats()

        try:
            # Mode 1: Load from previously saved coordinates
            if args.load_from:
                coords_file = Path(args.load_from)
                if not coords_file.exists():
                    raise FileNotFoundError(f"Coordinates file not found: {coords_file}")

                uuids, coordinates = load_coordinates(coords_file, console)
                stats.total_documents = len(uuids)
                stats.umap_computed = len(coordinates)

            # Mode 2: Calculate from embeddings
            else:
                assert embeddings_base_dir is not None, "embeddings_base_dir must be set when calculating"

                # Determine embeddings subdirectory based on collection type
                if collection_name == "DocumentChunks":
                    embeddings_dir = embeddings_base_dir / "chunk_embeddings"
                else:  # LegalDocuments
                    embeddings_dir = embeddings_base_dir / "agg_embeddings"

                if not embeddings_dir.exists():
                    console.print(
                        f"[bold yellow]Skipping {collection_name} - embeddings directory not found: {embeddings_dir}[/bold yellow]"
                    )
                    continue

                # Phase 1: Load embeddings from parquet files
                uuids, embeddings = load_embeddings_from_parquet(
                    embeddings_dir, collection_name, console
                )

                if len(embeddings) == 0:
                    console.print(
                        f"[bold red]Skipping {collection_name} - no embeddings found[/bold red]"
                    )
                    continue

                stats.total_documents = len(uuids)
                stats.vectors_extracted = len(uuids)
                stats.skipped_no_vector = 0

                # Phase 2: Normalize embeddings
                normalized_embeddings = normalize_embeddings(embeddings, console)
                stats.vectors_normalized = len(normalized_embeddings)

                # Phase 3: Compute UMAP coordinates
                coordinates = compute_umap_coordinates(normalized_embeddings, umap_config, console)
                stats.umap_computed = len(coordinates)

                # Phase 3.5: Save coordinates if output directory specified
                if output_dir:
                    coords_filename = f"{collection_name}_coords.parquet"
                    coords_path = output_dir / coords_filename
                    save_coordinates(uuids, coordinates, coords_path, console)

            # Phase 4: Update Weaviate (unless skip-update flag is set)
            if not args.skip_update:
                with WeaviateLegalDocumentsDatabase() as db:
                    successful, failed = update_weaviate_coordinates(
                        db,
                        collection_name,
                        uuids,
                        coordinates,
                        args.batch_size,
                        console,
                        dry_run=args.dry_run,
                    )
                    stats.documents_updated = successful
                    stats.failed_updates = failed
            else:
                console.print("[bold yellow]Skipping Weaviate update (--skip-update flag)[/bold yellow]")

            # Display statistics
            console.print("")
            display_statistics(stats, console)

        except Exception as e:
            console.print(f"[bold red]Error processing {collection_name}: {e}[/bold red]")
            logger.exception(f"Failed to process {collection_name}")
            continue

    console.print(
        Panel.fit(
            "[bold green]✓ UMAP coordinate calculation completed![/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
