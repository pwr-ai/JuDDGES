"""Check health status of extraction workers.

This script monitors:
- Worker responsiveness
- Redis queue status
- Recent error patterns
- Processing rate consistency
"""

import argparse
import json
import time
from typing import Dict, List

import redis
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console()


def check_redis_queue(redis_url: str, queue_name: str = "extraction_queue") -> Dict:
    """Check Redis queue status.

    Args:
        redis_url: Redis connection URL
        queue_name: Name of the extraction queue

    Returns:
        Dictionary with queue status
    """
    try:
        client = redis.from_url(redis_url, decode_responses=True)

        # Get queue length
        queue_length = client.llen(queue_name)

        # Check if Redis is responsive
        ping_time = time.time()
        client.ping()
        response_time = (time.time() - ping_time) * 1000  # ms

        # Try to peek at a job without removing it
        sample_job = None
        if queue_length > 0:
            # Get the last item without removing
            jobs = client.lrange(queue_name, -1, -1)
            if jobs:
                try:
                    sample_job = json.loads(jobs[0])
                except json.JSONDecodeError:
                    pass

        return {
            "status": "healthy",
            "queue_length": queue_length,
            "response_time_ms": round(response_time, 2),
            "sample_job": sample_job,
        }

    except redis.ConnectionError as e:
        return {
            "status": "error",
            "error": f"Cannot connect to Redis: {e}",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def check_worker_stats(redis_url: str, worker_stats_key: str = "worker_stats") -> List[Dict]:
    """Check worker statistics from Redis.

    Args:
        redis_url: Redis connection URL
        worker_stats_key: Redis key pattern for worker stats

    Returns:
        List of worker statistics
    """
    try:
        client = redis.from_url(redis_url, decode_responses=True)

        # Get all worker stat keys
        keys = client.keys(f"{worker_stats_key}:*")

        workers = []
        for key in keys:
            try:
                stats_json = client.get(key)
                if stats_json:
                    stats = json.loads(stats_json)
                    stats["key"] = key
                    workers.append(stats)
            except json.JSONDecodeError:
                pass

        return workers

    except Exception as e:
        logger.error(f"Failed to get worker stats: {e}")
        return []


def monitor_queue_rate(
    redis_url: str,
    queue_name: str = "extraction_queue",
    duration_seconds: int = 60
) -> Dict:
    """Monitor queue consumption rate.

    Args:
        redis_url: Redis connection URL
        queue_name: Queue name
        duration_seconds: How long to monitor

    Returns:
        Dictionary with rate statistics
    """
    try:
        client = redis.from_url(redis_url, decode_responses=True)

        # Get initial queue length
        initial_length = client.llen(queue_name)
        start_time = time.time()

        console.print(f"\n[cyan]Monitoring queue for {duration_seconds} seconds...[/cyan]")

        # Wait
        time.sleep(duration_seconds)

        # Get final queue length
        final_length = client.llen(queue_name)
        elapsed = time.time() - start_time

        # Calculate rate
        consumed = initial_length - final_length
        rate_per_sec = consumed / elapsed if elapsed > 0 else 0
        rate_per_min = rate_per_sec * 60

        return {
            "initial_length": initial_length,
            "final_length": final_length,
            "consumed": consumed,
            "elapsed_seconds": round(elapsed, 1),
            "rate_per_second": round(rate_per_sec, 2),
            "rate_per_minute": round(rate_per_min, 2),
        }

    except Exception as e:
        logger.error(f"Failed to monitor queue rate: {e}")
        return {}


def display_queue_status(status: Dict):
    """Display queue status in a table.

    Args:
        status: Queue status dictionary
    """
    console.print("\n[bold cyan]Redis Queue Status[/bold cyan]\n")

    if status.get("status") == "error":
        console.print(f"[red]✗ Error: {status.get('error')}[/red]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="yellow")

    queue_length = status.get("queue_length", 0)
    status_color = "green" if queue_length < 1000 else "yellow" if queue_length < 5000 else "red"

    table.add_row("Status", f"[{status_color}]● Online[/{status_color}]")
    table.add_row("Queue Length", f"[{status_color}]{queue_length:,}[/{status_color}]")
    table.add_row("Response Time", f"{status.get('response_time_ms', 0):.2f} ms")

    if status.get("sample_job"):
        job = status["sample_job"]
        table.add_row("Sample Job ID", job.get("job_id", "N/A"))
        table.add_row("Sample Job Docs", str(len(job.get("document_ids", []))))

    console.print(table)


def display_worker_stats(workers: List[Dict]):
    """Display worker statistics.

    Args:
        workers: List of worker stats
    """
    console.print("\n[bold cyan]Worker Statistics[/bold cyan]\n")

    if not workers:
        console.print("[yellow]No worker statistics found[/yellow]")
        console.print("[dim]Workers may not be reporting stats or have not started yet[/dim]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Worker", style="cyan")
    table.add_column("Jobs", justify="right", style="white")
    table.add_column("Success", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Rate (docs/min)", justify="right", style="yellow")
    table.add_column("Status", style="white")

    for worker in workers:
        worker_id = worker.get("worker_id", "?")
        jobs = worker.get("jobs_processed", 0)
        success = worker.get("documents_extracted", 0)
        failed = worker.get("documents_failed", 0)
        rate = worker.get("docs_per_min", 0)
        last_update = worker.get("last_update", "unknown")

        # Determine status
        if rate > 10:
            status = "[green]● Active[/green]"
        elif rate > 0:
            status = "[yellow]● Slow[/yellow]"
        else:
            status = "[red]● Idle[/red]"

        table.add_row(
            f"Worker {worker_id}",
            str(jobs),
            str(success),
            str(failed),
            f"{rate:.1f}",
            status,
        )

    console.print(table)


def display_rate_monitoring(rate_stats: Dict):
    """Display queue consumption rate.

    Args:
        rate_stats: Rate statistics dictionary
    """
    console.print("\n[bold cyan]Queue Consumption Rate[/bold cyan]\n")

    if not rate_stats:
        console.print("[yellow]No rate data available[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="yellow")

    consumed = rate_stats.get("consumed", 0)
    rate_per_min = rate_stats.get("rate_per_minute", 0)

    # Determine health
    if rate_per_min > 20:
        health = "[green]Healthy[/green]"
    elif rate_per_min > 5:
        health = "[yellow]Moderate[/yellow]"
    elif rate_per_min > 0:
        health = "[red]Slow[/red]"
    else:
        health = "[red]Stalled[/red]"

    table.add_row("Initial Queue Length", f"{rate_stats.get('initial_length', 0):,}")
    table.add_row("Final Queue Length", f"{rate_stats.get('final_length', 0):,}")
    table.add_row("Jobs Consumed", str(consumed))
    table.add_row("Elapsed Time", f"{rate_stats.get('elapsed_seconds', 0)} seconds")
    table.add_row("Rate", f"[bold]{rate_per_min:.2f} jobs/min[/bold]")
    table.add_row("Health", health)

    console.print(table)


def provide_health_recommendations(queue_status: Dict, rate_stats: Dict, workers: List[Dict]):
    """Provide health recommendations based on monitoring.

    Args:
        queue_status: Queue status
        rate_stats: Rate statistics
        workers: Worker statistics
    """
    console.print("\n[bold cyan]Health Assessment[/bold cyan]\n")

    issues = []

    # Check queue length
    queue_length = queue_status.get("queue_length", 0)
    if queue_length > 10000:
        issues.append("⚠ Queue is very large (> 10K jobs) - consider adding more workers")
    elif queue_length > 5000:
        issues.append("⚠ Queue is growing - monitor for bottlenecks")

    # Check rate
    rate = rate_stats.get("rate_per_minute", 0)
    if rate < 5 and queue_length > 100:
        issues.append("⚠ Processing rate is slow - check for errors or rate limiting")
    elif rate == 0 and queue_length > 0:
        issues.append("⚠ Queue is stalled - workers may have crashed or stopped")

    # Check workers
    if not workers:
        issues.append("⚠ No worker statistics available - workers may not be running")
    else:
        active_workers = sum(1 for w in workers if w.get("docs_per_min", 0) > 5)
        if active_workers == 0:
            issues.append("⚠ No active workers detected - check worker processes")

    # Display issues or healthy status
    if issues:
        for issue in issues:
            console.print(f"[yellow]{issue}[/yellow]")

        console.print("\n[bold]Recommended Actions:[/bold]")
        console.print("  • Check worker logs for errors")
        console.print("  • Run: [cyan]python scripts/extraction/check_errors_and_limits.py[/cyan]")
        console.print("  • Monitor Vertex AI quotas and rate limits")
        console.print("  • Consider scaling up workers if queue is growing")
    else:
        console.print("[green]✓ All systems appear healthy[/green]")
        console.print("  • Queue is being processed")
        console.print("  • Workers are active")
        console.print("  • No immediate issues detected")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Check extraction worker health")
    parser.add_argument(
        "--redis-url",
        type=str,
        default="redis://localhost:6379",
        help="Redis connection URL (default: redis://localhost:6379)",
    )
    parser.add_argument(
        "--queue-name",
        type=str,
        default="extraction_queue",
        help="Queue name to monitor (default: extraction_queue)",
    )
    parser.add_argument(
        "--monitor-rate",
        action="store_true",
        help="Monitor queue consumption rate for 60 seconds",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration to monitor rate in seconds (default: 60)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Live monitoring mode with auto-refresh",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=5,
        help="Refresh interval for live mode in seconds (default: 5)",
    )

    args = parser.parse_args()

    if args.live:
        # Live monitoring mode
        console.print(f"\n[cyan]Starting live health monitoring (refresh every {args.refresh}s)...[/cyan]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")

        try:
            while True:
                # Clear screen
                console.clear()

                # Check status
                queue_status = check_redis_queue(args.redis_url, args.queue_name)
                workers = check_worker_stats(args.redis_url)

                # Display
                display_queue_status(queue_status)
                display_worker_stats(workers)

                # Sleep
                time.sleep(args.refresh)

        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped by user[/yellow]")

    else:
        # Single check mode
        queue_status = check_redis_queue(args.redis_url, args.queue_name)
        display_queue_status(queue_status)

        workers = check_worker_stats(args.redis_url)
        display_worker_stats(workers)

        if args.monitor_rate:
            rate_stats = monitor_queue_rate(args.redis_url, args.queue_name, args.duration)
            display_rate_monitoring(rate_stats)
        else:
            rate_stats = {}

        # Provide recommendations
        provide_health_recommendations(queue_status, rate_stats, workers)


if __name__ == "__main__":
    main()
