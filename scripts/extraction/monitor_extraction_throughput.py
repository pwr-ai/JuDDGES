"""Monitor extraction throughput in real-time from PostgreSQL logs.

This script calculates and displays:
- Documents processed per minute
- Extraction rate trends over time
- Current active extraction runs
- Historical performance statistics
"""

import argparse
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.table import Table

from juddges.extraction import ExtractionStorage

console = Console()


def calculate_throughput(
    storage: ExtractionStorage,
    run_id: Optional[str] = None,
    minutes: int = 5
) -> Dict:
    """Calculate extraction throughput statistics.

    Args:
        storage: ExtractionStorage instance
        run_id: Optional specific run ID to monitor (None = all recent runs)
        minutes: Number of minutes to look back for rate calculation

    Returns:
        Dictionary with throughput statistics
    """
    # Get recent extraction logs
    since = datetime.now() - timedelta(minutes=minutes)

    # Query extraction logs from PostgreSQL
    with storage.engine.connect() as conn:
        # Get document extraction logs for throughput calculation
        if run_id:
            query = """
                SELECT
                    COUNT(*) as total_docs,
                    COUNT(CASE WHEN status = 'success' THEN 1 END) as successful_docs,
                    COUNT(CASE WHEN status = 'error' THEN 1 END) as failed_docs,
                    MIN(created_at) as start_time,
                    MAX(created_at) as end_time
                FROM document_extractions
                WHERE run_id = :run_id
                AND created_at >= :since
            """
            result = conn.execute(query, {"run_id": run_id, "since": since}).fetchone()
        else:
            query = """
                SELECT
                    COUNT(*) as total_docs,
                    COUNT(CASE WHEN status = 'success' THEN 1 END) as successful_docs,
                    COUNT(CASE WHEN status = 'error' THEN 1 END) as failed_docs,
                    MIN(created_at) as start_time,
                    MAX(created_at) as end_time
                FROM document_extractions
                WHERE created_at >= :since
            """
            result = conn.execute(query, {"since": since}).fetchone()

        if not result or result.total_docs == 0:
            return {
                "total_docs": 0,
                "successful_docs": 0,
                "failed_docs": 0,
                "docs_per_minute": 0.0,
                "duration_minutes": minutes,
            }

        # Calculate rate
        total_docs = result.total_docs
        successful_docs = result.successful_docs
        failed_docs = result.failed_docs

        # Calculate actual duration
        if result.start_time and result.end_time:
            duration = (result.end_time - result.start_time).total_seconds() / 60.0
            if duration < 0.1:  # Less than 6 seconds
                duration = 0.1  # Minimum duration to avoid division by zero
        else:
            duration = minutes

        docs_per_minute = total_docs / duration if duration > 0 else 0.0

        return {
            "total_docs": total_docs,
            "successful_docs": successful_docs,
            "failed_docs": failed_docs,
            "docs_per_minute": round(docs_per_minute, 2),
            "duration_minutes": round(duration, 2),
            "start_time": result.start_time,
            "end_time": result.end_time,
        }


def get_active_runs(storage: ExtractionStorage) -> List[Dict]:
    """Get currently active extraction runs.

    Args:
        storage: ExtractionStorage instance

    Returns:
        List of active run information
    """
    with storage.engine.connect() as conn:
        query = """
            SELECT
                id,
                model_name,
                sample_size,
                batch_size,
                max_workers,
                started_at,
                total_documents,
                successful_extractions,
                failed_extractions
            FROM extraction_runs
            WHERE completed_at IS NULL
            ORDER BY started_at DESC
            LIMIT 10
        """
        results = conn.execute(query).fetchall()

        runs = []
        for row in results:
            elapsed = (datetime.now() - row.started_at).total_seconds() / 60.0
            progress = 0.0
            if row.sample_size and row.total_documents:
                progress = (row.total_documents / row.sample_size) * 100

            runs.append({
                "id": str(row.id),
                "model": row.model_name,
                "sample_size": row.sample_size,
                "total_docs": row.total_documents or 0,
                "successful": row.successful_extractions or 0,
                "failed": row.failed_extractions or 0,
                "batch_size": row.batch_size,
                "workers": row.max_workers,
                "started_at": row.started_at,
                "elapsed_minutes": round(elapsed, 1),
                "progress_pct": round(progress, 1),
            })

        return runs


def get_recent_completed_runs(storage: ExtractionStorage, limit: int = 5) -> List[Dict]:
    """Get recently completed extraction runs.

    Args:
        storage: ExtractionStorage instance
        limit: Number of recent runs to retrieve

    Returns:
        List of completed run information
    """
    with storage.engine.connect() as conn:
        query = """
            SELECT
                id,
                model_name,
                total_documents,
                successful_extractions,
                failed_extractions,
                duration_seconds,
                started_at,
                completed_at
            FROM extraction_runs
            WHERE completed_at IS NOT NULL
            ORDER BY completed_at DESC
            LIMIT :limit
        """
        results = conn.execute(query, {"limit": limit}).fetchall()

        runs = []
        for row in results:
            duration_minutes = row.duration_seconds / 60.0 if row.duration_seconds else 0.0
            docs_per_minute = (
                row.total_documents / duration_minutes
                if duration_minutes > 0 and row.total_documents
                else 0.0
            )

            runs.append({
                "id": str(row.id),
                "model": row.model_name,
                "total_docs": row.total_documents or 0,
                "successful": row.successful_extractions or 0,
                "failed": row.failed_extractions or 0,
                "duration_minutes": round(duration_minutes, 1),
                "docs_per_minute": round(docs_per_minute, 2),
                "completed_at": row.completed_at,
            })

        return runs


def create_dashboard_table(
    active_runs: List[Dict],
    recent_runs: List[Dict],
    current_throughput: Dict,
) -> Table:
    """Create a rich table for the monitoring dashboard.

    Args:
        active_runs: List of active extraction runs
        recent_runs: List of recently completed runs
        current_throughput: Current throughput statistics

    Returns:
        Rich Table object
    """
    table = Table(title="[bold cyan]Extraction Throughput Monitor[/bold cyan]", show_header=True)

    # Current throughput section
    table.add_row(
        "[bold yellow]Current Throughput (Last 5 minutes)[/bold yellow]",
        "",
        "",
        "",
        "",
    )
    table.add_row(
        "Total Documents",
        str(current_throughput["total_docs"]),
        "",
        "",
        "",
    )
    table.add_row(
        "Successful",
        f"[green]{current_throughput['successful_docs']}[/green]",
        "",
        "",
        "",
    )
    table.add_row(
        "Failed",
        f"[red]{current_throughput['failed_docs']}[/red]",
        "",
        "",
        "",
    )
    table.add_row(
        "Rate",
        f"[bold]{current_throughput['docs_per_minute']} docs/min[/bold]",
        "",
        "",
        "",
    )
    table.add_row("", "", "", "", "")  # Separator

    # Active runs section
    if active_runs:
        table.add_row(
            "[bold yellow]Active Runs[/bold yellow]",
            "",
            "",
            "",
            "",
        )
        for run in active_runs:
            table.add_row(
                f"Run {run['id'][:8]}...",
                f"{run['model']}",
                f"Progress: {run['progress_pct']:.1f}% ({run['total_docs']}/{run['sample_size']})",
                f"Elapsed: {run['elapsed_minutes']} min",
                f"Workers: {run['workers']}",
            )
        table.add_row("", "", "", "", "")  # Separator

    # Recent completed runs section
    if recent_runs:
        table.add_row(
            "[bold yellow]Recent Completed Runs[/bold yellow]",
            "",
            "",
            "",
            "",
        )
        for run in recent_runs:
            success_rate = (
                (run['successful'] / run['total_docs'] * 100)
                if run['total_docs'] > 0
                else 0
            )
            table.add_row(
                f"Run {run['id'][:8]}...",
                f"{run['model']}",
                f"Docs: {run['total_docs']} ({success_rate:.1f}% success)",
                f"Duration: {run['duration_minutes']:.1f} min",
                f"[bold]{run['docs_per_minute']:.2f} docs/min[/bold]",
            )

    return table


def monitor_live(storage: ExtractionStorage, refresh_seconds: int = 5):
    """Monitor extraction throughput in real-time with live updates.

    Args:
        storage: ExtractionStorage instance
        refresh_seconds: Refresh interval in seconds
    """
    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                # Fetch current data
                current_throughput = calculate_throughput(storage, minutes=5)
                active_runs = get_active_runs(storage)
                recent_runs = get_recent_completed_runs(storage, limit=5)

                # Create dashboard
                table = create_dashboard_table(active_runs, recent_runs, current_throughput)

                # Update live display
                live.update(table)

                # Wait before next update
                time.sleep(refresh_seconds)

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped by user[/yellow]")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Monitor extraction throughput")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Monitor specific extraction run ID",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=5,
        help="Number of minutes to look back for rate calculation (default: 5)",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=5,
        help="Refresh interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live monitoring mode with auto-refresh",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=5,
        help="Number of recent completed runs to show (default: 5)",
    )

    args = parser.parse_args()

    try:
        # Initialize storage
        logger.info("Connecting to extraction storage (PostgreSQL)...")
        storage = ExtractionStorage()

        if args.live:
            # Live monitoring mode
            console.print(
                f"\n[cyan]Starting live monitoring (refresh every {args.refresh}s)...[/cyan]"
            )
            console.print("[dim]Press Ctrl+C to stop[/dim]\n")
            monitor_live(storage, refresh_seconds=args.refresh)
        else:
            # Single snapshot mode
            current_throughput = calculate_throughput(storage, run_id=args.run_id, minutes=args.minutes)
            active_runs = get_active_runs(storage)
            recent_runs = get_recent_completed_runs(storage, limit=args.recent)

            # Display results
            table = create_dashboard_table(active_runs, recent_runs, current_throughput)
            console.print(table)

            # Additional summary
            console.print(f"\n[dim]Showing data from last {args.minutes} minutes[/dim]")
            if args.run_id:
                console.print(f"[dim]Filtered to run ID: {args.run_id}[/dim]")

    except Exception as e:
        logger.error(f"Error monitoring extraction: {e}")
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
