#!/usr/bin/env python3
"""
Update Weaviate documents with raw_content from HuggingFace datasets.

This script:
1. Connects to Weaviate
2. Loads the specified HuggingFace dataset
3. Builds an index for fast lookups
4. Updates all Weaviate documents with raw_content from the dataset
"""

import argparse
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import track

from juddges.data.dataset_mapper import DatasetToWeaviateMapper
from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase


console = Console()


def main():
    """Update Weaviate documents with raw_content from dataset."""
    parser = argparse.ArgumentParser(
        description="Update Weaviate documents with raw_content from HuggingFace dataset"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., 'juddges/pl-court-raw', 'AI-TAX/pl-eureka-raw')",
    )
    parser.add_argument(
        "--raw-text-field",
        type=str,
        default="text",
        help="Field in dataset containing raw text (default: 'text')",
    )
    parser.add_argument(
        "--id-field",
        type=str,
        default=None,
        help="Primary ID field in dataset (default: auto-detect based on dataset)",
    )
    parser.add_argument(
        "--secondary-id-field",
        type=str,
        default=None,
        help="Secondary ID field for fallback (default: auto-detect based on dataset)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for updates (default: 100)",
    )
    parser.add_argument(
        "--weaviate-host",
        type=str,
        default="localhost",
        help="Weaviate host (default: localhost)",
    )
    parser.add_argument(
        "--weaviate-port",
        type=int,
        default=8222,
        help="Weaviate HTTP port (default: 8222)",
    )
    parser.add_argument(
        "--weaviate-grpc-port",
        type=int,
        default=50051,
        help="Weaviate gRPC port (default: 50051)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )

    args = parser.parse_args()

    # Auto-detect ID fields based on dataset
    id_field = args.id_field
    secondary_id_field = args.secondary_id_field

    if id_field is None:
        if "court" in args.dataset_name.lower():
            id_field = "judgment_id"
            secondary_id_field = secondary_id_field or "docket_number"
        elif "eureka" in args.dataset_name.lower():
            id_field = "id"
            secondary_id_field = secondary_id_field or "docker_number"
        else:
            logger.error(
                "Could not auto-detect ID field. Please specify --id-field explicitly."
            )
            return

    logger.info(f"Using ID field: {id_field}, secondary: {secondary_id_field}")

    # Connect to Weaviate
    console.print(f"[bold blue]Connecting to Weaviate at {args.weaviate_host}:{args.weaviate_port}...")
    db = WeaviateLegalDocumentsDatabase(
        host=args.weaviate_host,
        port=args.weaviate_port,
        grpc_port=args.weaviate_grpc_port,
    )

    # Initialize mapper
    console.print(f"[bold blue]Loading dataset {args.dataset_name}...")
    mapper = DatasetToWeaviateMapper(
        db=db,
        dataset_name=args.dataset_name,
    )

    # Build index
    console.print("[bold blue]Building dataset index...")
    mapper.build_index(
        id_field=id_field,
        secondary_id_field=secondary_id_field,
    )

    # Check for missing raw_content
    console.print("[bold blue]Checking for documents missing raw_content...")
    missing_raw_content = mapper.get_missing_raw_content_documents()
    console.print(f"[yellow]Found {len(missing_raw_content)} documents missing raw_content")

    if args.dry_run:
        console.print("\n[bold yellow]DRY RUN MODE - No changes will be made")
        if missing_raw_content:
            console.print("\n[bold]Sample documents that would be updated:")
            for i, doc in enumerate(missing_raw_content[:5]):
                console.print(
                    f"  {i+1}. ID: {doc.get('document_id')}, "
                    f"Number: {doc.get('document_number')}"
                )
        console.print(
            f"\n[bold green]Would update {len(missing_raw_content)} documents with raw_content"
        )
        return

    # Update documents
    console.print(f"\n[bold blue]Updating documents with raw_content from field '{args.raw_content_field}'...")
    updated_count = mapper.update_raw_content_from_dataset(
        raw_content_field=args.raw_content_field,
        batch_size=args.batch_size,
        document_id_field=id_field,
    )

    console.print(f"\n[bold green]✓ Successfully updated {updated_count} documents")


if __name__ == "__main__":
    main()
