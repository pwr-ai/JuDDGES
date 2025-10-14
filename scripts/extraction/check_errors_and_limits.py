"""Check extraction logs for timeouts, rate limiting, and errors.

This script analyzes extraction logs and database records to identify:
- Rate limiting incidents (429 errors)
- Timeout errors
- Server errors (500, 503)
- Retry patterns
- Error frequency and trends
"""

import argparse
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()

try:
    from juddges.extraction import ExtractionStorage
    HAS_DB = True
except ImportError:
    HAS_DB = False
    logger.warning("ExtractionStorage not available - DB analysis disabled")


def parse_log_file(log_file: Path) -> Dict:
    """Parse log file for errors and rate limiting.

    Args:
        log_file: Path to log file

    Returns:
        Dictionary with error statistics
    """
    stats = {
        "rate_limits": 0,
        "timeouts": 0,
        "server_errors": 0,
        "retries": 0,
        "429_errors": 0,
        "503_errors": 0,
        "500_errors": 0,
        "total_errors": 0,
        "error_examples": [],
    }

    # Patterns to detect (use word boundaries to avoid false positives)
    patterns = {
        "rate_limit": re.compile(r"rate.?limit|\b429\b", re.IGNORECASE),
        "timeout": re.compile(r"timeout|timed.?out", re.IGNORECASE),
        # Server errors: match HTTP status codes, not in timestamps or IDs
        "server_error": re.compile(r"(?:status|code|error).*?(?:500|503|502)|\b(?:500|503|502)\b.*?(?:error|failed)", re.IGNORECASE),
        "retry": re.compile(r"retry|retrying|attempt \d+", re.IGNORECASE),
        "error": re.compile(r"ERROR|CRITICAL|Failed|Exception", re.IGNORECASE),
    }

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                # Check for rate limiting
                if patterns["rate_limit"].search(line):
                    stats["rate_limits"] += 1
                    if "429" in line:
                        stats["429_errors"] += 1
                    if len(stats["error_examples"]) < 10:
                        stats["error_examples"].append({
                            "type": "rate_limit",
                            "line": line_num,
                            "message": line.strip()[:200],
                        })

                # Check for timeouts
                if patterns["timeout"].search(line):
                    stats["timeouts"] += 1
                    if len(stats["error_examples"]) < 10:
                        stats["error_examples"].append({
                            "type": "timeout",
                            "line": line_num,
                            "message": line.strip()[:200],
                        })

                # Check for server errors
                if patterns["server_error"].search(line):
                    stats["server_errors"] += 1
                    if "503" in line:
                        stats["503_errors"] += 1
                    if "500" in line:
                        stats["500_errors"] += 1
                    if len(stats["error_examples"]) < 10:
                        stats["error_examples"].append({
                            "type": "server_error",
                            "line": line_num,
                            "message": line.strip()[:200],
                        })

                # Check for retries
                if patterns["retry"].search(line):
                    stats["retries"] += 1

                # Check for general errors
                if patterns["error"].search(line):
                    stats["total_errors"] += 1

    except Exception as e:
        logger.error(f"Failed to parse log file {log_file}: {e}")

    return stats


def check_database_errors(
    storage: "ExtractionStorage",
    run_id: Optional[str] = None,
    hours: int = 24
) -> Dict:
    """Check database for extraction errors.

    Args:
        storage: ExtractionStorage instance
        run_id: Optional specific run ID to check
        hours: Number of hours to look back

    Returns:
        Dictionary with error statistics from database
    """
    if not HAS_DB or not storage:
        return {}

    since = datetime.now() - timedelta(hours=hours)

    try:
        with storage.engine.connect() as conn:
            # Query for failed extractions
            if run_id:
                query = """
                    SELECT
                        COUNT(*) as total_failed,
                        COUNT(CASE WHEN error LIKE '%429%' OR error LIKE '%rate limit%' THEN 1 END) as rate_limit_errors,
                        COUNT(CASE WHEN error LIKE '%timeout%' THEN 1 END) as timeout_errors,
                        COUNT(CASE WHEN error LIKE '%500%' OR error LIKE '%503%' THEN 1 END) as server_errors,
                        COUNT(CASE WHEN attempts > 1 THEN 1 END) as retried_docs
                    FROM document_extractions
                    WHERE run_id = :run_id
                    AND status = 'error'
                    AND created_at >= :since
                """
                result = conn.execute(query, {"run_id": run_id, "since": since}).fetchone()
            else:
                query = """
                    SELECT
                        COUNT(*) as total_failed,
                        COUNT(CASE WHEN error LIKE '%429%' OR error LIKE '%rate limit%' THEN 1 END) as rate_limit_errors,
                        COUNT(CASE WHEN error LIKE '%timeout%' THEN 1 END) as timeout_errors,
                        COUNT(CASE WHEN error LIKE '%500%' OR error LIKE '%503%' THEN 1 END) as server_errors,
                        COUNT(CASE WHEN attempts > 1 THEN 1 END) as retried_docs
                    FROM document_extractions
                    WHERE status = 'error'
                    AND created_at >= :since
                """
                result = conn.execute(query, {"since": since}).fetchone()

            if result:
                return {
                    "total_failed": result.total_failed,
                    "rate_limit_errors": result.rate_limit_errors,
                    "timeout_errors": result.timeout_errors,
                    "server_errors": result.server_errors,
                    "retried_docs": result.retried_docs,
                }

    except Exception as e:
        logger.error(f"Failed to query database: {e}")

    return {}


def get_error_timeline(
    storage: "ExtractionStorage",
    run_id: Optional[str] = None,
    hours: int = 24
) -> List[Dict]:
    """Get timeline of errors over time.

    Args:
        storage: ExtractionStorage instance
        run_id: Optional specific run ID
        hours: Number of hours to look back

    Returns:
        List of error counts by hour
    """
    if not HAS_DB or not storage:
        return []

    since = datetime.now() - timedelta(hours=hours)

    try:
        with storage.engine.connect() as conn:
            if run_id:
                query = """
                    SELECT
                        date_trunc('hour', created_at) as hour,
                        COUNT(*) as errors,
                        COUNT(CASE WHEN error LIKE '%429%' OR error LIKE '%rate limit%' THEN 1 END) as rate_limits,
                        COUNT(CASE WHEN error LIKE '%timeout%' THEN 1 END) as timeouts
                    FROM document_extractions
                    WHERE run_id = :run_id
                    AND status = 'error'
                    AND created_at >= :since
                    GROUP BY hour
                    ORDER BY hour
                """
                results = conn.execute(query, {"run_id": run_id, "since": since}).fetchall()
            else:
                query = """
                    SELECT
                        date_trunc('hour', created_at) as hour,
                        COUNT(*) as errors,
                        COUNT(CASE WHEN error LIKE '%429%' OR error LIKE '%rate limit%' THEN 1 END) as rate_limits,
                        COUNT(CASE WHEN error LIKE '%timeout%' THEN 1 END) as timeouts
                    FROM document_extractions
                    WHERE status = 'error'
                    AND created_at >= :since
                    GROUP BY hour
                    ORDER BY hour
                """
                results = conn.execute(query, {"since": since}).fetchall()

            return [
                {
                    "hour": row.hour.strftime("%Y-%m-%d %H:00"),
                    "errors": row.errors,
                    "rate_limits": row.rate_limits,
                    "timeouts": row.timeouts,
                }
                for row in results
            ]

    except Exception as e:
        logger.error(f"Failed to get error timeline: {e}")

    return []


def display_log_results(log_file: Path, stats: Dict):
    """Display log file analysis results.

    Args:
        log_file: Path to log file
        stats: Statistics dictionary
    """
    console.print(f"\n[bold cyan]Log File Analysis: {log_file.name}[/bold cyan]\n")

    # Summary table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Error Type", style="cyan", width=25)
    table.add_column("Count", justify="right", style="yellow")
    table.add_column("Status", style="white")

    # Rate limits
    rate_limit_status = "[red]⚠ DETECTED[/red]" if stats["rate_limits"] > 0 else "[green]✓ None[/green]"
    table.add_row("Rate Limiting (429)", str(stats["rate_limits"]), rate_limit_status)

    # Timeouts
    timeout_status = "[red]⚠ DETECTED[/red]" if stats["timeouts"] > 0 else "[green]✓ None[/green]"
    table.add_row("Timeouts", str(stats["timeouts"]), timeout_status)

    # Server errors
    server_status = "[red]⚠ DETECTED[/red]" if stats["server_errors"] > 0 else "[green]✓ None[/green]"
    table.add_row("Server Errors (5xx)", str(stats["server_errors"]), server_status)

    # Retries
    retry_status = "[yellow]↻ Active[/yellow]" if stats["retries"] > 0 else "[green]✓ None[/green]"
    table.add_row("Retry Attempts", str(stats["retries"]), retry_status)

    # Total errors
    table.add_row("Total Errors", str(stats["total_errors"]), "")

    console.print(table)

    # Show error examples
    if stats["error_examples"]:
        console.print("\n[bold cyan]Error Examples:[/bold cyan]\n")
        for i, example in enumerate(stats["error_examples"][:5], 1):
            console.print(f"[yellow]{i}. Line {example['line']} ({example['type']}):[/yellow]")
            console.print(f"   {example['message']}\n")


def display_database_results(db_stats: Dict, timeline: List[Dict]):
    """Display database analysis results.

    Args:
        db_stats: Database statistics
        timeline: Error timeline data
    """
    console.print(f"\n[bold cyan]Database Analysis (Last 24 hours)[/bold cyan]\n")

    if not db_stats:
        console.print("[yellow]No database statistics available[/yellow]")
        return

    # Summary table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Count", justify="right", style="yellow")
    table.add_column("Percentage", justify="right", style="white")

    total = db_stats.get("total_failed", 0)

    def calc_pct(count):
        return f"{(count / total * 100):.1f}%" if total > 0 else "0%"

    table.add_row("Total Failed Extractions", str(total), "100%")
    table.add_row(
        "Rate Limit Errors",
        str(db_stats.get("rate_limit_errors", 0)),
        calc_pct(db_stats.get("rate_limit_errors", 0))
    )
    table.add_row(
        "Timeout Errors",
        str(db_stats.get("timeout_errors", 0)),
        calc_pct(db_stats.get("timeout_errors", 0))
    )
    table.add_row(
        "Server Errors",
        str(db_stats.get("server_errors", 0)),
        calc_pct(db_stats.get("server_errors", 0))
    )
    table.add_row(
        "Retried Documents",
        str(db_stats.get("retried_docs", 0)),
        calc_pct(db_stats.get("retried_docs", 0))
    )

    console.print(table)

    # Show timeline if available
    if timeline:
        console.print("\n[bold cyan]Error Timeline:[/bold cyan]\n")
        timeline_table = Table(show_header=True, header_style="bold magenta")
        timeline_table.add_column("Hour", style="cyan")
        timeline_table.add_column("Total Errors", justify="right", style="red")
        timeline_table.add_column("Rate Limits", justify="right", style="yellow")
        timeline_table.add_column("Timeouts", justify="right", style="orange1")

        for entry in timeline[-24:]:  # Show last 24 hours
            timeline_table.add_row(
                entry["hour"],
                str(entry["errors"]),
                str(entry["rate_limits"]),
                str(entry["timeouts"]),
            )

        console.print(timeline_table)


def provide_recommendations(log_stats: Dict, db_stats: Dict):
    """Provide recommendations based on error analysis.

    Args:
        log_stats: Log file statistics
        db_stats: Database statistics
    """
    console.print("\n[bold cyan]Recommendations:[/bold cyan]\n")

    has_rate_limits = (
        log_stats.get("rate_limits", 0) > 0 or
        db_stats.get("rate_limit_errors", 0) > 0
    )
    has_timeouts = (
        log_stats.get("timeouts", 0) > 0 or
        db_stats.get("timeout_errors", 0) > 0
    )
    has_server_errors = (
        log_stats.get("server_errors", 0) > 0 or
        db_stats.get("server_errors", 0) > 0
    )

    if has_rate_limits:
        console.print("[red]⚠[/red] [bold]Rate Limiting Detected[/bold]")
        console.print("  • Reduce [cyan]--max-workers[/cyan] (try 3-5 instead of 10)")
        console.print("  • Increase [cyan]--batch-size[/cyan] (process more docs per request)")
        console.print("  • Add delays between batches")
        console.print("  • Check Vertex AI quota limits\n")

    if has_timeouts:
        console.print("[yellow]⚠[/yellow] [bold]Timeout Errors Detected[/bold]")
        console.print("  • Reduce [cyan]max_text_length[/cyan] (current: 150000)")
        console.print("  • Decrease [cyan]--batch-size[/cyan] (try 3-5 for long documents)")
        console.print("  • Check network connectivity to Vertex AI")
        console.print("  • Increase timeout settings in GeminiExtractionChain\n")

    if has_server_errors:
        console.print("[red]⚠[/red] [bold]Server Errors Detected[/bold]")
        console.print("  • Check Vertex AI service status")
        console.print("  • Retry logic is active (3 attempts with exponential backoff)")
        console.print("  • Monitor for sustained issues")
        console.print("  • Consider switching to backup model\n")

    if not (has_rate_limits or has_timeouts or has_server_errors):
        console.print("[green]✓[/green] [bold]No major issues detected[/bold]")
        console.print("  • Extraction appears to be running smoothly")
        console.print("  • Continue monitoring for any changes\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Check extraction logs for timeouts and rate limiting"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file to analyze",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory containing log files (default: logs)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Check database for specific run ID",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Number of hours to look back in database (default: 24)",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip database analysis (only analyze log files)",
    )

    args = parser.parse_args()

    # Analyze log files
    log_stats = {}
    if args.log_file:
        log_path = Path(args.log_file)
        if log_path.exists():
            log_stats = parse_log_file(log_path)
            display_log_results(log_path, log_stats)
        else:
            console.print(f"[red]Log file not found: {args.log_file}[/red]")
    elif Path(args.log_dir).exists():
        # Find most recent log file
        log_files = sorted(Path(args.log_dir).glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if log_files:
            log_path = log_files[0]
            console.print(f"[dim]Analyzing most recent log: {log_path}[/dim]")
            log_stats = parse_log_file(log_path)
            display_log_results(log_path, log_stats)
        else:
            console.print(f"[yellow]No log files found in {args.log_dir}[/yellow]")

    # Analyze database
    db_stats = {}
    timeline = []
    if not args.no_db and HAS_DB:
        try:
            storage = ExtractionStorage()
            db_stats = check_database_errors(storage, args.run_id, args.hours)
            timeline = get_error_timeline(storage, args.run_id, args.hours)
            display_database_results(db_stats, timeline)
        except Exception as e:
            logger.error(f"Failed to analyze database: {e}")
            console.print(f"[yellow]Database analysis skipped: {e}[/yellow]")

    # Provide recommendations
    if log_stats or db_stats:
        provide_recommendations(log_stats, db_stats)


if __name__ == "__main__":
    main()
