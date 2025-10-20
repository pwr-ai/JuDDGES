#!/usr/bin/env python3
"""Job coordinator for large-scale distributed extraction.

This coordinator:
1. Fetches document IDs from Weaviate based on search queries
2. Filters out already-extracted documents (optional)
3. Splits documents into small batches and queues them to Redis
4. Monitors worker progress

Usage:
    # Queue 1M documents for extraction (small batches for max parallelization)
    python scripts/extraction/coordinator.py \\
        --search-queries "kredyt frankowy" "IP Box" \\
        --max-documents 1000000 \\
        --job-batch-size 2 \\
        --redis-url redis://:PASSWORD@localhost:6379/0
"""

import argparse
import json
import os
import time
import uuid
from typing import List, Optional

import redis
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from juddges.extraction import ExtractionStorage, WeaviateRestClient
from juddges.settings import ROOT_PATH

# Load environment variables
load_dotenv(ROOT_PATH / ".env", override=True)

console = Console()


class ExtractionCoordinator:
    """Coordinates large-scale extraction by queuing jobs to Redis."""

    def __init__(
        self,
        redis_url: str,
        queue_name: str = "extraction_queue",
        job_batch_size: int = 2,
    ):
        """Initialize coordinator.

        Args:
            redis_url: Redis connection URL (supports redis://[:password@]host:port/db format)
            queue_name: Name of Redis queue
            job_batch_size: Number of documents per job (default: 2 for optimal parallelization)
        """
        self.queue_name = queue_name
        self.job_batch_size = job_batch_size

        # Connect to Redis with authentication support
        # Redis URL can include password: redis://:password@host:port/db
        logger.info(f"Connecting to Redis: {redis_url.split('@')[-1] if '@' in redis_url else redis_url}")

        try:
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except redis.exceptions.AuthenticationError as e:
            logger.error(
                f"Redis authentication failed. Please ensure REDIS_URL includes password: "
                f"redis://:YOUR_PASSWORD@host:port/db"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        # Initialize Weaviate client
        self.weaviate_client = WeaviateRestClient.from_env()

        # Initialize storage
        try:
            self.storage = ExtractionStorage()
            logger.info("Connected to extraction storage")
        except Exception as e:
            logger.warning(f"Storage not available: {e}")
            self.storage = None

    def coordinate_extraction(
        self,
        search_queries: List[str],
        max_documents: int,
        document_type_filter: Optional[str] = None,
        filter_already_extracted: bool = False,
        run_name: Optional[str] = None,
        search_mode: str = "hybrid",
        force_cursor: bool = False,
        skip_documents: int = 0,
        sort_by_creation_time: bool = False,
    ) -> str:
        """Coordinate large-scale extraction.

        Args:
            search_queries: List of search queries to find documents (ignored if force_cursor=True)
            max_documents: Maximum number of documents to extract
            document_type_filter: Optional document type filter
            filter_already_extracted: Skip documents that already have extracted data
            run_name: Optional name for this extraction run
            search_mode: Search mode - "keyword" (BM25), "semantic" (vector), or "hybrid" (default)
            force_cursor: Skip search queries and iterate through ALL documents using cursor pagination
            skip_documents: Number of documents to skip before starting extraction (useful for resuming)
            sort_by_creation_time: Sort documents by creation time (oldest first). Useful for processing in chronological order.

        Returns:
            Extraction run ID
        """
        # Create extraction run in database
        run_id = self._create_extraction_run(
            search_queries=search_queries,
            max_documents=max_documents,
            document_type_filter=document_type_filter,
            run_name=run_name,
        )

        console.print(f"\n[bold cyan]Extraction Coordinator[/bold cyan]")
        console.print(f"Run ID: {run_id}")
        console.print(f"Target documents: {max_documents:,}")
        if skip_documents > 0:
            console.print(f"Skip documents: [yellow]{skip_documents:,}[/yellow] (starting from offset)")
        if search_queries:
            console.print(f"Search queries: {', '.join(search_queries)}")
            console.print(f"Search mode: {search_mode}")
        else:
            console.print(f"Search queries: [yellow]None (fetching ALL documents)[/yellow]")
        console.print(f"Job batch size: {self.job_batch_size}")
        console.print(f"Redis queue: {self.queue_name}")
        if force_cursor:
            console.print(f"[yellow]Force cursor: enabled (bypassing 10K limit)[/yellow]")
        console.print()

        # Fetch all matching document IDs
        console.print("[cyan]Step 1:[/cyan] Fetching document IDs from Weaviate...")
        all_document_ids = []

        # If no search queries, fetch all documents with force_cursor
        if not search_queries:
            console.print(f"  Fetching ALL documents (no search filter)")
            doc_ids = self._fetch_document_ids(
                search_query=None,
                document_type_filter=document_type_filter,
                limit=max_documents,
                search_mode=search_mode,
                force_cursor=force_cursor,
                skip_documents=skip_documents,
                sort_by_creation_time=sort_by_creation_time,
            )
            all_document_ids.extend(doc_ids)
            console.print(f"  Found: {len(doc_ids):,} documents")
        else:
            # Fetch documents for each search query
            for query in search_queries:
                console.print(f"  Querying: '{query}' (mode: {search_mode})")
                doc_ids = self._fetch_document_ids(
                    search_query=query,
                    document_type_filter=document_type_filter,
                    limit=max_documents,
                    search_mode=search_mode,
                    force_cursor=force_cursor,
                    skip_documents=skip_documents,
                    sort_by_creation_time=sort_by_creation_time,
                )
                all_document_ids.extend(doc_ids)
                console.print(f"  Found: {len(doc_ids):,} documents")

        # Remove duplicates
        all_document_ids = list(set(all_document_ids))
        console.print(f"\n  Total unique documents: {len(all_document_ids):,}")

        # Limit to maximum documents
        if len(all_document_ids) > max_documents:
            all_document_ids = all_document_ids[:max_documents]
            console.print(f"  Limited to: {max_documents:,} documents")

        # Optional: Filter already-extracted documents
        if filter_already_extracted:
            console.print("\n[cyan]Step 2:[/cyan] Filtering already-extracted documents...")
            all_document_ids = self._filter_already_extracted(all_document_ids)
            console.print(f"  Documents needing extraction: {len(all_document_ids):,}")

        # Split into job batches and queue
        console.print(f"\n[cyan]Step 3:[/cyan] Queuing jobs to Redis...")
        num_jobs = (len(all_document_ids) + self.job_batch_size - 1) // self.job_batch_size

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Queuing jobs...", total=num_jobs)

            jobs_queued = 0
            for i in range(0, len(all_document_ids), self.job_batch_size):
                batch_doc_ids = all_document_ids[i : i + self.job_batch_size]

                job = {
                    "job_id": str(uuid.uuid4()),
                    "run_id": str(run_id),  # Convert UUID to string
                    "document_ids": batch_doc_ids,
                }

                # Push to Redis queue
                self.redis_client.lpush(self.queue_name, json.dumps(job))
                jobs_queued += 1

                progress.update(task, advance=1)

        console.print(f"\n[green]✓[/green] Queued {jobs_queued:,} jobs to Redis")
        console.print(f"[green]✓[/green] Total documents: {len(all_document_ids):,}")
        console.print(f"\nWorkers can now poll queue: {self.queue_name}")

        return run_id

    def _create_extraction_run(
        self,
        search_queries: List[str],
        max_documents: int,
        document_type_filter: Optional[str],
        run_name: Optional[str],
    ) -> str:
        """Create extraction run in database.

        Args:
            search_queries: List of search queries
            max_documents: Maximum number of documents
            document_type_filter: Optional document type filter
            run_name: Optional run name

        Returns:
            Run ID
        """
        if not self.storage:
            return str(uuid.uuid4())

        try:
            run_id = self.storage.create_extraction_run(
                model_name="gemini-2.5-pro",
                sample_size=max_documents,
                batch_size=self.job_batch_size,
                max_workers=0,  # Distributed workers
                weaviate_host=os.getenv("WEAVIATE_HOST", "localhost"),
                weaviate_port=int(os.getenv("WEAVIATE_PORT", "8084")),
                search_query="; ".join(search_queries) if search_queries else "ALL_DOCUMENTS",
                document_type_filter=document_type_filter,
                vertex_project=os.getenv("VERTEX_PROJECT", "insbay-b32351"),
                vertex_location=os.getenv("VERTEX_LOCATION", "us-central1"),
                temperature=0.0,
                prompt_template="Distributed extraction",
                extraction_schema={},
                random_seed=42,
                notes=run_name or f"Large-scale extraction: {max_documents:,} documents",
            )
            logger.info(f"Created extraction run: {run_id}")
            return run_id

        except Exception as e:
            logger.error(f"Failed to create extraction run: {e}")
            return str(uuid.uuid4())

    def _fetch_document_ids(
        self,
        search_query: Optional[str],
        document_type_filter: Optional[str],
        limit: int,
        search_mode: str = "hybrid",
        force_cursor: bool = False,
        skip_documents: int = 0,
        sort_by_creation_time: bool = False,
    ) -> List[str]:
        """Fetch document IDs from Weaviate.

        Args:
            search_query: Optional search query
            document_type_filter: Optional document type filter
            limit: Maximum number of documents
            search_mode: Search mode - "keyword", "semantic", or "hybrid"
            force_cursor: Force cursor pagination even with filters/search
            skip_documents: Number of documents to skip before collecting results
            sort_by_creation_time: Sort by creation time (oldest first)

        Returns:
            List of document IDs
        """
        # When using force_cursor without search, use exact limit
        # When using search queries, fetch 2x to account for filtering
        if force_cursor and not search_query:
            fetch_limit = limit
        else:
            fetch_limit = min(limit * 2, 100000)  # Weaviate has limits

        documents = self.weaviate_client.fetch_documents(
            max_documents=fetch_limit,
            search_query=search_query,
            document_type_filter=document_type_filter,
            search_mode=search_mode,
            force_cursor=force_cursor,
            skip_documents=skip_documents,
            sort_by="publication_date" if sort_by_creation_time else None,
            sort_order="asc" if sort_by_creation_time else "asc",
        )

        return [doc.get("document_id") for doc in documents if doc.get("document_id")]

    def _filter_already_extracted(self, document_ids: List[str]) -> List[str]:
        """Filter out documents that already have extracted data.

        Args:
            document_ids: List of document IDs

        Returns:
            List of document IDs needing extraction
        """
        if not self.storage:
            logger.warning("Storage not available, cannot filter already-extracted documents")
            return document_ids

        try:
            # Get all successfully processed document IDs from the database
            processed_ids = self.storage.get_processed_document_ids(status="success")

            # Convert input list to set for efficient filtering
            document_ids_set = set(document_ids)

            # Filter out already processed documents
            remaining_ids = document_ids_set - processed_ids

            filtered_count = len(document_ids) - len(remaining_ids)
            logger.info(
                f"Filtered {filtered_count:,} already-extracted documents. "
                f"Remaining: {len(remaining_ids):,} documents"
            )

            return list(remaining_ids)

        except Exception as e:
            logger.error(f"Failed to filter already-extracted documents: {e}")
            logger.warning("Returning all documents without filtering")
            return document_ids

    def monitor_progress(self, run_id: str, refresh_interval: int = 30):
        """Monitor extraction progress.

        Args:
            run_id: Extraction run ID to monitor
            refresh_interval: Seconds between refreshes
        """
        console.print(f"\n[bold cyan]Monitoring extraction run: {run_id}[/bold cyan]\n")

        try:
            while True:
                # Check queue length
                queue_length = self.redis_client.llen(self.queue_name)

                # Get statistics from storage
                if self.storage:
                    # TODO: Query storage for run statistics
                    console.print(f"Queue length: {queue_length:,} jobs remaining")
                else:
                    console.print(f"Queue length: {queue_length:,} jobs remaining")

                if queue_length == 0:
                    console.print("[green]✓ Queue empty - extraction complete![/green]")
                    break

                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped[/yellow]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Large-scale extraction coordinator")
    parser.add_argument(
        "--search-queries",
        nargs="+",
        default=None,
        help="Search queries to find documents (optional when using --force-cursor)",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=1000,
        help="Maximum number of documents to extract",
    )
    parser.add_argument(
        "--document-type",
        type=str,
        default=None,
        choices=["judgment", "tax_interpretation"],
        help="Optional document type filter",
    )
    parser.add_argument(
        "--filter-already-extracted",
        action="store_true",
        help="Skip documents that already have extracted data",
    )
    parser.add_argument(
        "--job-batch-size",
        type=int,
        default=2,
        help="Number of documents per job - smaller batches allow more parallelization (default: 2, recommended: 2-3)",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default=None,
        help="Redis connection URL (format: redis://:password@host:port/db, defaults to REDIS_URL env var)",
    )
    parser.add_argument(
        "--queue-name",
        type=str,
        default="extraction_queue",
        help="Redis queue name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for this extraction run",
    )
    parser.add_argument(
        "--search-mode",
        type=str,
        default="hybrid",
        choices=["keyword", "semantic", "hybrid"],
        help="Search mode: 'keyword' (BM25 for exact terms), 'semantic' (vector similarity), or 'hybrid' (both, default)",
    )
    parser.add_argument(
        "--force-cursor",
        action="store_true",
        help="Skip search queries and use cursor pagination to fetch ALL documents (bypasses 10K offset limit)",
    )
    parser.add_argument(
        "--skip-documents",
        type=int,
        default=0,
        help="Number of documents to skip before starting extraction (useful for resuming interrupted jobs, default: 0)",
    )
    parser.add_argument(
        "--sort-by-creation-time",
        action="store_true",
        help="Sort documents by publication_date (oldest first). Note: Sorting is LIMITED to 10K documents (Weaviate offset limit) and CANNOT be used with --force-cursor or hybrid/semantic search.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor progress after queuing jobs",
    )

    args = parser.parse_args()

    # Validate search queries requirement
    if not args.search_queries and not args.force_cursor:
        console.print(
            "[red]Error:[/red] --search-queries is required unless --force-cursor is specified.\n"
            "Either provide search queries or use --force-cursor to fetch ALL documents."
        )
        return

    # Default to empty list if no search queries provided
    search_queries = args.search_queries or []

    # Get Redis URL from args or environment
    redis_url = args.redis_url or os.getenv("REDIS_URL")
    if not redis_url:
        console.print(
            "[red]Error:[/red] Redis URL not provided. "
            "Set REDIS_URL environment variable or use --redis-url argument.\n"
            "Format: redis://:PASSWORD@host:port/db"
        )
        return

    # Create coordinator
    coordinator = ExtractionCoordinator(
        redis_url=redis_url,
        queue_name=args.queue_name,
        job_batch_size=args.job_batch_size,
    )

    # Queue extraction jobs
    run_id = coordinator.coordinate_extraction(
        search_queries=search_queries,
        max_documents=args.max_documents,
        document_type_filter=args.document_type,
        filter_already_extracted=args.filter_already_extracted,
        run_name=args.run_name,
        search_mode=args.search_mode,
        force_cursor=args.force_cursor,
        skip_documents=args.skip_documents,
        sort_by_creation_time=args.sort_by_creation_time,
    )

    # Optional: Monitor progress
    if args.monitor:
        coordinator.monitor_progress(run_id)


if __name__ == "__main__":
    main()
