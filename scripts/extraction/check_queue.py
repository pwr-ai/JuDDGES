#!/usr/bin/env python3
"""Check Redis queue status for extraction jobs."""

import argparse
import json
import os

import redis
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.table import Table

from juddges.settings import ROOT_PATH

# Load environment variables
load_dotenv(ROOT_PATH / ".env", override=True)

console = Console()


def check_queue_status(redis_url: str, queue_name: str = "extraction_queue"):
    """Check the status of the Redis extraction queue.

    Args:
        redis_url: Redis connection URL
        queue_name: Name of the queue to check
    """
    try:
        # Connect to Redis
        client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        client.ping()

        # Get queue length
        queue_length = client.llen(queue_name)

        # Create summary table
        table = Table(title=f"Redis Queue Status: {queue_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Queue Length", f"{queue_length:,} jobs")

        # Calculate approximate documents (assuming 2 docs per job)
        if queue_length > 0:
            # Peek at first job to get actual batch size
            first_job = client.lindex(queue_name, 0)
            if first_job:
                job_data = json.loads(first_job)
                docs_per_job = len(job_data.get("document_ids", []))
                approx_docs = queue_length * docs_per_job
                table.add_row("Documents per job", str(docs_per_job))
                table.add_row("Approximate documents", f"{approx_docs:,}")
                table.add_row("Run ID (first job)", job_data.get("run_id", "N/A"))

        console.print(table)

        # Show more details if requested
        if queue_length == 0:
            console.print("\n[green]✓ Queue is empty - all jobs completed![/green]")
        else:
            console.print(f"\n[yellow]⚠ {queue_length:,} jobs remaining[/yellow]")

    except redis.exceptions.AuthenticationError:
        console.print("[red]Error:[/red] Redis authentication failed. Check REDIS_URL.")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check Redis queue status")
    parser.add_argument(
        "--redis-url",
        type=str,
        default=None,
        help="Redis connection URL (defaults to REDIS_URL env var)",
    )
    parser.add_argument(
        "--queue-name",
        type=str,
        default="extraction_queue",
        help="Redis queue name",
    )

    args = parser.parse_args()

    # Get Redis URL from args or environment
    redis_url = args.redis_url or os.getenv("REDIS_URL")
    if not redis_url:
        console.print(
            "[red]Error:[/red] Redis URL not provided. "
            "Set REDIS_URL environment variable or use --redis-url argument."
        )
        return

    check_queue_status(redis_url, args.queue_name)


if __name__ == "__main__":
    main()
