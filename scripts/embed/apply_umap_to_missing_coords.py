#!/usr/bin/env python3
"""
Apply saved UMAP model to documents with missing x,y coordinates in Weaviate.

This script efficiently processes documents in batches:
1. Single iterator pass - processes all documents in one scan (O(n) complexity)
2. Larger processing batches (1000 docs) for efficient UMAP transformation
3. Generator-based streaming to minimize memory usage

Usage:
    # Test mode: Apply to 1000 random documents with missing coordinates
    docker compose run --rm web python scripts/embed/apply_umap_to_missing_coords.py \
        --model-path models/umap/umap_model_LegalDocuments.pkl \
        --collection LegalDocuments \
        --test-mode \
        --test-sample-size 1000

    # Production mode: Apply to all documents with missing coordinates
    docker compose run --rm web python scripts/embed/apply_umap_to_missing_coords.py \
        --model-path models/umap/umap_model_LegalDocuments.pkl \
        --collection LegalDocuments \
        --process-batch-size 1000

    # Dry run mode: Preview without updating
    docker compose run --rm web python scripts/embed/apply_umap_to_missing_coords.py \
        --model-path models/umap/umap_model_LegalDocuments.pkl \
        --collection LegalDocuments \
        --test-mode \
        --dry-run
"""

import argparse
import pickle
from pathlib import Path
from typing import Generator, List, Tuple

import numpy as np
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

from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase
from juddges.settings import VectorName


def stream_documents_with_missing_coords(
    db: WeaviateLegalDocumentsDatabase,
    collection_name: str,
    vector_name: str,
    console: Console,
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """Stream documents with missing x or y coordinates using single iterator pass.

    Args:
        db: Weaviate database instance
        collection_name: Name of collection to query
        vector_name: Name of vector to retrieve
        console: Rich console for output

    Yields:
        Tuple of (uuid, vector) for each document with missing coordinates
    """
    collection = db.get_collection(collection_name)

    # Single pass through iterator - no re-scanning!
    for obj in collection.iterator(
        include_vector=True,
        return_properties=["x", "y"],
    ):
        x = obj.properties.get("x")
        y = obj.properties.get("y")

        # Only yield documents with missing coordinates
        if x is None or y is None:
            vector = obj.vector.get(vector_name) if obj.vector else None

            if vector is not None:
                yield str(obj.uuid), np.array(vector)
            else:
                logger.warning(f"Document {obj.uuid} has no vector, skipping")


def update_coordinates_batch(
    db: WeaviateLegalDocumentsDatabase,
    collection_name: str,
    uuids: List[str],
    coordinates: np.ndarray,
    console: Console,
) -> Tuple[int, int]:
    """Update x,y coordinates for a batch of documents.

    Note: Weaviate doesn't have a batch update API for property-only updates,
    so each document is updated individually.

    Args:
        db: Weaviate database instance
        collection_name: Name of collection to update
        uuids: List of document UUIDs
        coordinates: Array of (x, y) coordinates
        console: Rich console for output

    Returns:
        Tuple of (successful_updates, failed_updates)
    """
    collection = db.get_collection(collection_name)
    successful_updates = 0
    failed_updates = 0

    # Update documents one by one (Weaviate limitation for property-only updates)
    for uuid, coords in zip(uuids, coordinates):
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

    return successful_updates, failed_updates


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Apply saved UMAP model to documents with missing x,y coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved UMAP model pickle file (e.g., models/umap/umap_model_LegalDocuments.pkl)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        choices=["LegalDocuments", "DocumentChunks"],
        required=True,
        help="Collection to process",
    )
    parser.add_argument(
        "--vector-name",
        type=str,
        default=VectorName.BASE,
        help=f"Name of vector to use (default: {VectorName.BASE})",
    )
    parser.add_argument(
        "--process-batch-size",
        type=int,
        default=1000,
        help="Number of documents to process in memory at once (default: 1000)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: Process a subset of documents first",
    )
    parser.add_argument(
        "--test-sample-size",
        type=int,
        default=1000,
        help="Number of documents to process in test mode (default: 1000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without updating Weaviate",
    )

    args = parser.parse_args()

    # Validate model path exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"UMAP model not found: {model_path}")

    console = Console()

    # Display configuration
    mode = "Test Mode (Sample)" if args.test_mode else "Production Mode (All Documents)"
    config_text = f"[bold blue]Apply UMAP to Missing Coordinates[/bold blue]\n\n"
    config_text += f"Mode: {mode}\n"
    config_text += f"Model Path: {model_path}\n"
    config_text += f"Collection: {args.collection}\n"
    config_text += f"Vector Name: {args.vector_name}\n"
    config_text += f"Process Batch Size: {args.process_batch_size}\n"

    if args.test_mode:
        config_text += f"Test Sample Size: {args.test_sample_size}\n"

    if args.dry_run:
        config_text += "Dry Run: True\n"

    console.print(Panel.fit(config_text, title="Configuration", border_style="blue"))

    try:
        with WeaviateLegalDocumentsDatabase() as db:
            # Load UMAP model once
            console.print(f"[bold cyan]Loading UMAP model from {model_path}...[/bold cyan]")
            with open(model_path, "rb") as f:
                umap_model = pickle.load(f)
            console.print("[green]✓[/green] UMAP model loaded successfully")

            # Stream and process documents
            total_successful = 0
            total_failed = 0
            docs_processed = 0

            console.print(
                f"[bold cyan]Streaming documents and processing in batches of {args.process_batch_size}...[/bold cyan]"
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Streaming and processing...", total=None)

                # Create document stream (single iterator pass)
                doc_stream = stream_documents_with_missing_coords(
                    db, args.collection, args.vector_name, console
                )

                # Accumulate documents into process batches
                uuids_batch = []
                vectors_batch = []

                for uuid, vector in doc_stream:
                    uuids_batch.append(uuid)
                    vectors_batch.append(vector)

                    # Process when we have enough documents
                    if len(uuids_batch) >= args.process_batch_size:
                        # Convert to numpy array
                        vectors_array = np.vstack(vectors_batch)

                        # Apply UMAP transformation
                        normalized_vectors = normalize(vectors_array, norm="l2", axis=1)
                        coords = umap_model.transform(normalized_vectors)

                        # Update Weaviate
                        if not args.dry_run:
                            successful, failed = update_coordinates_batch(
                                db,
                                args.collection,
                                uuids_batch,
                                coords,
                                console,
                            )
                        else:
                            console.print(
                                f"[yellow]DRY RUN: Would update {len(uuids_batch)} documents[/yellow]"
                            )
                            successful, failed = len(uuids_batch), 0

                        total_successful += successful
                        total_failed += failed
                        docs_processed += len(uuids_batch)

                        progress.update(
                            task,
                            description=f"Processed {docs_processed} documents...",
                        )

                        # Clear batch
                        uuids_batch = []
                        vectors_batch = []

                    # Stop if in test mode and we've processed enough
                    if args.test_mode and docs_processed >= args.test_sample_size:
                        console.print(
                            f"[yellow]Test mode complete: Processed {docs_processed} documents[/yellow]"
                        )
                        break

                # Process remaining documents in final batch
                if len(uuids_batch) > 0:
                    vectors_array = np.vstack(vectors_batch)
                    normalized_vectors = normalize(vectors_array, norm="l2", axis=1)
                    coords = umap_model.transform(normalized_vectors)

                    if not args.dry_run:
                        successful, failed = update_coordinates_batch(
                            db,
                            args.collection,
                            uuids_batch,
                            coords,
                            console,
                        )
                    else:
                        console.print(
                            f"[yellow]DRY RUN: Would update {len(uuids_batch)} documents[/yellow]"
                        )
                        successful, failed = len(uuids_batch), 0

                    total_successful += successful
                    total_failed += failed
                    docs_processed += len(uuids_batch)

            # Display summary
            summary_table = Table(title="Processing Summary")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Count", style="yellow", justify="right")

            summary_table.add_row("Documents Processed", str(docs_processed))
            summary_table.add_row("Successful Updates", str(total_successful))
            summary_table.add_row("Failed Updates", str(total_failed))

            console.print(summary_table)

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        logger.exception("Failed to apply UMAP to missing coordinates")
        raise

    console.print(
        Panel.fit(
            "[bold green]✓ UMAP application completed![/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
