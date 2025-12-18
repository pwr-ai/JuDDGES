#!/usr/bin/env python3
"""Generate Redis tasks for large-scale distributed extraction.

This script:
1. Fetches up to 1,000,000 document IDs from Weaviate
2. Filters out already-processed documents from PostgreSQL
3. Creates Redis tasks with 10 documents each
4. Tracks statistics on documents added vs already processed

Usage:
    # Generate tasks for 1M documents (10 docs per task)
    python scripts/extraction/generate_redis_tasks.py \
        --max-documents 1000000 \
        --task-size 10 \
        --redis-url redis://:PASSWORD@localhost:6379/0

    # With search query
    python scripts/extraction/generate_redis_tasks.py \
        --search-queries "kredyt frankowy" \
        --max-documents 100000 \
        --task-size 10

    # Process ALL documents using cursor pagination
    python scripts/extraction/generate_redis_tasks.py \
        --force-cursor \
        --max-documents 1000000 \
        --task-size 10
"""

import argparse
import json
import os
import time
import uuid
from typing import List, Optional, Set

import redis
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from juddges.extraction import ExtractionStorage, WeaviateRestClient
from juddges.settings import ROOT_PATH

# Load environment variables
load_dotenv(ROOT_PATH / ".env", override=True)

console = Console()


class RedisTaskGenerator:
    """Generates Redis tasks for distributed extraction with deduplication."""

    def __init__(
        self,
        redis_url: str,
        queue_name: str = "extraction_queue",
        task_size: int = 10,
    ):
        """Initialize task generator.

        Args:
            redis_url: Redis connection URL (supports redis://[:password@]host:port/db format)
            queue_name: Name of Redis queue
            task_size: Number of documents per task (default: 10)
        """
        self.queue_name = queue_name
        self.task_size = task_size

        # Connect to Redis with authentication support
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

        # Initialize storage for deduplication
        try:
            self.storage = ExtractionStorage()
            logger.info("Connected to extraction storage")
        except Exception as e:
            logger.warning(f"Storage not available: {e}")
            self.storage = None

    def generate_tasks(
        self,
        max_documents: int = 1_000_000,
        search_queries: Optional[List[str]] = None,
        document_type_filter: Optional[str] = None,
        search_mode: str = "hybrid",
        force_cursor: bool = False,
        skip_documents: int = 0,
        sort_by_creation_time: bool = False,
        run_name: Optional[str] = None,
    ) -> dict:
        """Generate Redis tasks with deduplication.

        Args:
            max_documents: Maximum number of documents to process
            search_queries: List of search queries to find documents (ignored if force_cursor=True)
            document_type_filter: Optional document type filter
            search_mode: Search mode - "keyword" (BM25), "semantic" (vector), or "hybrid" (default)
            force_cursor: Skip search queries and iterate through ALL documents using cursor pagination
            skip_documents: Number of documents to skip before starting
            sort_by_creation_time: Sort documents by creation time (oldest first)
            run_name: Optional name for this extraction run

        Returns:
            Dictionary with statistics about task generation
        """
        start_time = time.time()

        # Create extraction run in database
        run_id = self._create_extraction_run(
            search_queries=search_queries or [],
            max_documents=max_documents,
            document_type_filter=document_type_filter,
            run_name=run_name,
        )

        console.print(f"\n[bold cyan]Redis Task Generator[/bold cyan]")
        console.print(f"Run ID: {run_id}")
        console.print(f"Target documents: {max_documents:,}")
        console.print(f"Task size: {self.task_size} documents per task")
        console.print(f"Redis queue: {self.queue_name}")
        if skip_documents > 0:
            console.print(f"Skip documents: [yellow]{skip_documents:,}[/yellow]")
        if search_queries:
            console.print(f"Search queries: {', '.join(search_queries)}")
            console.print(f"Search mode: {search_mode}")
        else:
            console.print(f"Search queries: [yellow]None (fetching ALL documents)[/yellow]")
        if force_cursor:
            console.print(f"[yellow]Force cursor: enabled (bypassing 10K limit)[/yellow]")
        console.print()

        # Step 1: Fetch document IDs from Weaviate
        console.print("[cyan]Step 1:[/cyan] Fetching document IDs from Weaviate...")
        all_document_ids = self._fetch_document_ids(
            search_queries=search_queries or [],
            document_type_filter=document_type_filter,
            max_documents=max_documents,
            search_mode=search_mode,
            force_cursor=force_cursor,
            skip_documents=skip_documents,
            sort_by_creation_time=sort_by_creation_time,
        )

        total_fetched = len(all_document_ids)
        console.print(f"  Fetched: [green]{total_fetched:,}[/green] document IDs")

        # Step 2: Get already processed documents
        console.print("\n[cyan]Step 2:[/cyan] Checking already processed documents...")
        processed_ids = self._get_processed_documents()
        console.print(f"  Already processed: [yellow]{len(processed_ids):,}[/yellow] documents")

        # Step 3: Filter out processed documents
        console.print("\n[cyan]Step 3:[/cyan] Filtering documents...")
        document_ids_set = set(all_document_ids)
        remaining_ids = list(document_ids_set - processed_ids)

        filtered_count = len(all_document_ids) - len(remaining_ids)
        console.print(f"  Filtered out: [yellow]{filtered_count:,}[/yellow] already processed")
        console.print(f"  Remaining to process: [green]{len(remaining_ids):,}[/green] documents")

        if not remaining_ids:
            console.print("\n[yellow]No new documents to process![/yellow]")
            return {
                "run_id": str(run_id),
                "total_fetched": total_fetched,
                "already_processed": len(processed_ids),
                "filtered_count": filtered_count,
                "tasks_generated": 0,
                "documents_queued": 0,
                "duration_seconds": time.time() - start_time,
            }

        # Step 4: Generate Redis tasks
        console.print(f"\n[cyan]Step 4:[/cyan] Generating Redis tasks...")
        num_tasks = (len(remaining_ids) + self.task_size - 1) // self.task_size

        tasks_generated = 0
        documents_queued = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total} tasks"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Queuing tasks...", total=num_tasks)

            for i in range(0, len(remaining_ids), self.task_size):
                batch_doc_ids = remaining_ids[i : i + self.task_size]

                job = {
                    "job_id": str(uuid.uuid4()),
                    "run_id": str(run_id),
                    "document_ids": batch_doc_ids,
                }

                # Push to Redis queue
                self.redis_client.lpush(self.queue_name, json.dumps(job))
                tasks_generated += 1
                documents_queued += len(batch_doc_ids)

                progress.update(task, advance=1)

        duration = time.time() - start_time

        # Print summary table
        console.print("\n[bold green]✓ Task Generation Complete[/bold green]\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")

        table.add_row("Total fetched from Weaviate", f"{total_fetched:,}")
        table.add_row("Already processed", f"{len(processed_ids):,}")
        table.add_row("Filtered out (duplicates)", f"{filtered_count:,}")
        table.add_row("New documents to process", f"{len(remaining_ids):,}")
        table.add_row("Tasks generated", f"{tasks_generated:,}")
        table.add_row("Documents queued", f"{documents_queued:,}")
        table.add_row("Average docs per task", f"{documents_queued / tasks_generated:.1f}")
        table.add_row("Duration", f"{duration:.1f}s")

        console.print(table)
        console.print(f"\nWorkers can now poll queue: [cyan]{self.queue_name}[/cyan]\n")

        return {
            "run_id": str(run_id),
            "total_fetched": total_fetched,
            "already_processed": len(processed_ids),
            "filtered_count": filtered_count,
            "new_documents": len(remaining_ids),
            "tasks_generated": tasks_generated,
            "documents_queued": documents_queued,
            "duration_seconds": duration,
        }

    def _fetch_document_ids(
        self,
        search_queries: List[str],
        document_type_filter: Optional[str],
        max_documents: int,
        search_mode: str,
        force_cursor: bool,
        skip_documents: int,
        sort_by_creation_time: bool,
    ) -> List[str]:
        """Fetch document IDs from Weaviate."""
        all_document_ids = []

        if not search_queries:
            # Fetch ALL documents
            console.print(f"  Fetching ALL documents (no search filter)")
            doc_ids = self._fetch_documents_batch(
                search_query=None,
                document_type_filter=document_type_filter,
                limit=max_documents,
                search_mode=search_mode,
                force_cursor=force_cursor,
                skip_documents=skip_documents,
                sort_by_creation_time=sort_by_creation_time,
            )
            all_document_ids.extend(doc_ids)
        else:
            # Fetch documents for each search query
            for query in search_queries:
                console.print(f"  Querying: '{query}' (mode: {search_mode})")
                doc_ids = self._fetch_documents_batch(
                    search_query=query,
                    document_type_filter=document_type_filter,
                    limit=max_documents,
                    search_mode=search_mode,
                    force_cursor=force_cursor,
                    skip_documents=skip_documents,
                    sort_by_creation_time=sort_by_creation_time,
                )
                all_document_ids.extend(doc_ids)
                console.print(f"    Found: {len(doc_ids):,} documents")

        # Remove duplicates
        all_document_ids = list(set(all_document_ids))

        # Limit to maximum documents
        if len(all_document_ids) > max_documents:
            all_document_ids = all_document_ids[:max_documents]

        return all_document_ids

    def _fetch_documents_batch(
        self,
        search_query: Optional[str],
        document_type_filter: Optional[str],
        limit: int,
        search_mode: str,
        force_cursor: bool,
        skip_documents: int,
        sort_by_creation_time: bool,
    ) -> List[str]:
        """Fetch a batch of document IDs from Weaviate."""
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

    def _get_processed_documents(self) -> Set[str]:
        """Get set of already processed document IDs from PostgreSQL."""
        if not self.storage:
            logger.warning("Storage not available, cannot filter already-processed documents")
            return set()

        try:
            # Get all successfully processed document IDs
            processed_ids = self.storage.get_processed_document_ids(status="success")
            return processed_ids
        except Exception as e:
            logger.error(f"Failed to get processed documents: {e}")
            return set()

    def _create_extraction_run(
        self,
        search_queries: List[str],
        max_documents: int,
        document_type_filter: Optional[str],
        run_name: Optional[str],
    ) -> str:
        """Create extraction run in database."""
        if not self.storage:
            return str(uuid.uuid4())

        try:
            run_id = self.storage.create_extraction_run(
                model_name="gemini-2.5-pro",
                sample_size=max_documents,
                batch_size=self.task_size,
                max_workers=0,  # Distributed workers
                weaviate_host=os.getenv("WEAVIATE_HOST", "localhost"),
                weaviate_port=int(os.getenv("WEAVIATE_PORT", "8084")),
                search_query="; ".join(search_queries) if search_queries else "ALL_DOCUMENTS",
                document_type_filter=document_type_filter,
                vertex_project=os.getenv("VERTEX_PROJECT", "insbay-b32351"),
                vertex_location=os.getenv("VERTEX_LOCATION", "us-central1"),
                temperature=0.0,
                prompt_template="Distributed extraction with 10-doc tasks",
                extraction_schema={},
                random_seed=42,
                notes=run_name
                or f"Large-scale extraction: {max_documents:,} docs, {self.task_size} per task",
            )
            logger.info(f"Created extraction run: {run_id}")
            return run_id
        except Exception as e:
            logger.error(f"Failed to create extraction run: {e}")
            return str(uuid.uuid4())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate Redis tasks for distributed extraction")
    parser.add_argument(
        "--max-documents",
        type=int,
        default=1_000_000,
        help="Maximum number of documents to process (default: 1,000,000)",
    )
    parser.add_argument(
        "--task-size",
        type=int,
        default=10,
        help="Number of documents per Redis task (default: 10)",
    )
    parser.add_argument(
        "--search-queries",
        nargs="+",
        default=None,
        help="Search queries to find documents (optional when using --force-cursor)",
    )
    parser.add_argument(
        "--document-type",
        type=str,
        default=None,
        choices=["judgment", "tax_interpretation"],
        help="Optional document type filter",
    )
    parser.add_argument(
        "--search-mode",
        type=str,
        default="hybrid",
        choices=["keyword", "semantic", "hybrid"],
        help="Search mode: 'keyword' (BM25), 'semantic' (vector), or 'hybrid' (default)",
    )
    parser.add_argument(
        "--force-cursor",
        action="store_true",
        help="Skip search queries and use cursor pagination to fetch ALL documents",
    )
    parser.add_argument(
        "--skip-documents",
        type=int,
        default=0,
        help="Number of documents to skip before starting (default: 0)",
    )
    parser.add_argument(
        "--sort-by-creation-time",
        action="store_true",
        help="Sort documents by publication_date (oldest first)",
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
        help="Redis queue name (default: extraction_queue)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for this extraction run",
    )

    args = parser.parse_args()

    # Validate search queries requirement
    if not args.search_queries and not args.force_cursor:
        console.print(
            "[red]Error:[/red] --search-queries is required unless --force-cursor is specified.\n"
            "Either provide search queries or use --force-cursor to fetch ALL documents."
        )
        return

    # Get Redis URL from args or environment
    redis_url = args.redis_url or os.getenv("REDIS_URL")
    if not redis_url:
        console.print(
            "[red]Error:[/red] Redis URL not provided. "
            "Set REDIS_URL environment variable or use --redis-url argument.\n"
            "Format: redis://:PASSWORD@host:port/db"
        )
        return

    # Create generator
    generator = RedisTaskGenerator(
        redis_url=redis_url,
        queue_name=args.queue_name,
        task_size=args.task_size,
    )

    # Generate tasks
    stats = generator.generate_tasks(
        max_documents=args.max_documents,
        search_queries=args.search_queries,
        document_type_filter=args.document_type,
        search_mode=args.search_mode,
        force_cursor=args.force_cursor,
        skip_documents=args.skip_documents,
        sort_by_creation_time=args.sort_by_creation_time,
        run_name=args.run_name,
    )

    # Save stats to file for reference
    stats_file = ROOT_PATH / "data" / "extraction_runs" / f"{stats['run_id']}_task_generation.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    console.print(f"Statistics saved to: [cyan]{stats_file}[/cyan]")


if __name__ == "__main__":
    main()
