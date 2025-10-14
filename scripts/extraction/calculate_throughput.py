"""Calculate extraction throughput from log files or extraction results.

This is a simpler alternative that works without PostgreSQL access,
analyzing extraction result files to calculate processing rates.
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()


def parse_extraction_results(results_file: Path) -> List[Dict]:
    """Parse extraction results from JSONL file.

    Args:
        results_file: Path to extraction results JSONL file

    Returns:
        List of extraction result dictionaries
    """
    results = []

    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")

    return results


def calculate_throughput_from_results(results: List[Dict]) -> Dict:
    """Calculate throughput statistics from extraction results.

    Args:
        results: List of extraction result dictionaries

    Returns:
        Dictionary with throughput statistics
    """
    if not results:
        return {
            "total_docs": 0,
            "successful_docs": 0,
            "failed_docs": 0,
            "docs_per_minute": 0.0,
            "duration_minutes": 0.0,
        }

    # Count statuses
    successful = sum(1 for r in results if r.get("extraction_status") == "success")
    failed = len(results) - successful

    # Extract timestamps if available
    timestamps = []
    for result in results:
        if "timestamp" in result:
            try:
                ts = datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))
                timestamps.append(ts)
            except (ValueError, AttributeError):
                pass

    # Calculate duration from timestamps
    if len(timestamps) >= 2:
        duration_seconds = (max(timestamps) - min(timestamps)).total_seconds()
        duration_minutes = max(duration_seconds / 60.0, 0.1)  # Minimum 0.1 minutes
    else:
        # Fallback: estimate based on typical extraction time
        # Assume ~1 document per minute as fallback
        duration_minutes = len(results)

    docs_per_minute = len(results) / duration_minutes if duration_minutes > 0 else 0.0

    return {
        "total_docs": len(results),
        "successful_docs": successful,
        "failed_docs": failed,
        "docs_per_minute": round(docs_per_minute, 2),
        "duration_minutes": round(duration_minutes, 2),
        "success_rate": round((successful / len(results) * 100) if len(results) > 0 else 0.0, 1),
        "has_timestamps": len(timestamps) >= 2,
    }


def analyze_extraction_directory(directory: Path) -> List[Dict]:
    """Analyze all extraction result files in a directory.

    Args:
        directory: Directory containing extraction results

    Returns:
        List of analysis results for each file
    """
    analyses = []

    # Find all extraction result files
    result_files = list(directory.glob("*extracted*.jsonl"))

    if not result_files:
        logger.warning(f"No extraction result files found in {directory}")
        return analyses

    for result_file in result_files:
        logger.info(f"Analyzing {result_file.name}...")

        try:
            results = parse_extraction_results(result_file)
            stats = calculate_throughput_from_results(results)

            analyses.append({
                "file": result_file.name,
                "stats": stats,
            })

        except Exception as e:
            logger.error(f"Failed to analyze {result_file.name}: {e}")

    return analyses


def display_throughput_table(analyses: List[Dict]):
    """Display throughput analysis results in a table.

    Args:
        analyses: List of analysis results
    """
    table = Table(title="[bold cyan]Extraction Throughput Analysis[/bold cyan]", show_header=True)

    table.add_column("File", style="cyan", width=40)
    table.add_column("Total Docs", justify="right", style="white")
    table.add_column("Success", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Success Rate", justify="right", style="yellow")
    table.add_column("Duration (min)", justify="right", style="blue")
    table.add_column("Docs/Min", justify="right", style="bold magenta")

    for analysis in analyses:
        file_name = analysis["file"]
        stats = analysis["stats"]

        # Add timestamp indicator
        timestamp_indicator = " ✓" if stats.get("has_timestamps") else " [dim](estimated)[/dim]"

        table.add_row(
            file_name,
            str(stats["total_docs"]),
            str(stats["successful_docs"]),
            str(stats["failed_docs"]),
            f"{stats['success_rate']:.1f}%",
            f"{stats['duration_minutes']:.1f}{timestamp_indicator}",
            f"[bold]{stats['docs_per_minute']:.2f}[/bold]",
        )

    console.print("\n")
    console.print(table)
    console.print("\n")


def estimate_completion_time(
    remaining_docs: int,
    current_rate: float,
    success_rate: float = 100.0
) -> Dict:
    """Estimate completion time based on current throughput.

    Args:
        remaining_docs: Number of documents remaining to process
        current_rate: Current processing rate (docs/min)
        success_rate: Expected success rate percentage

    Returns:
        Dictionary with time estimates
    """
    if current_rate <= 0:
        return {
            "eta_minutes": float("inf"),
            "eta_hours": float("inf"),
            "eta_days": float("inf"),
        }

    # Adjust for expected failures
    effective_docs = remaining_docs * (100.0 / success_rate) if success_rate > 0 else remaining_docs

    eta_minutes = effective_docs / current_rate
    eta_hours = eta_minutes / 60.0
    eta_days = eta_hours / 24.0

    return {
        "eta_minutes": round(eta_minutes, 1),
        "eta_hours": round(eta_hours, 1),
        "eta_days": round(eta_days, 2),
        "completion_time": datetime.now() + timedelta(minutes=eta_minutes),
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Calculate extraction throughput from result files"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="data/extraction_results",
        help="Directory containing extraction results (default: data/extraction_results)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Specific result file to analyze",
    )
    parser.add_argument(
        "--estimate",
        type=int,
        default=None,
        help="Estimate completion time for remaining documents (provide number of docs)",
    )
    parser.add_argument(
        "--success-rate",
        type=float,
        default=100.0,
        help="Expected success rate percentage for estimation (default: 100.0)",
    )

    args = parser.parse_args()

    if args.file:
        # Analyze single file
        file_path = Path(args.file)
        if not file_path.exists():
            console.print(f"[red]File not found: {args.file}[/red]")
            return

        logger.info(f"Analyzing {file_path}...")
        results = parse_extraction_results(file_path)
        stats = calculate_throughput_from_results(results)

        analyses = [{"file": file_path.name, "stats": stats}]
        display_throughput_table(analyses)

        # Calculate average rate for estimation
        avg_rate = stats["docs_per_minute"]

    else:
        # Analyze directory
        directory = Path(args.directory)
        if not directory.exists():
            console.print(f"[red]Directory not found: {args.directory}[/red]")
            return

        logger.info(f"Analyzing extraction results in {directory}...")
        analyses = analyze_extraction_directory(directory)

        if not analyses:
            console.print("[yellow]No extraction results found to analyze[/yellow]")
            return

        display_throughput_table(analyses)

        # Calculate average rate across all files
        total_docs = sum(a["stats"]["total_docs"] for a in analyses)
        total_duration = sum(a["stats"]["duration_minutes"] for a in analyses)
        avg_rate = total_docs / total_duration if total_duration > 0 else 0.0

    # Display average statistics
    if len(analyses) > 1:
        console.print("[bold cyan]Average Statistics:[/bold cyan]")
        console.print(f"Average throughput: [bold]{avg_rate:.2f} docs/min[/bold]")

    # Estimate completion time if requested
    if args.estimate:
        console.print(f"\n[bold cyan]Completion Time Estimate[/bold cyan]")
        console.print(f"Remaining documents: {args.estimate:,}")
        console.print(f"Current rate: {avg_rate:.2f} docs/min")
        console.print(f"Expected success rate: {args.success_rate:.1f}%")

        estimates = estimate_completion_time(args.estimate, avg_rate, args.success_rate)

        if estimates["eta_minutes"] == float("inf"):
            console.print("\n[red]Cannot estimate - no processing rate available[/red]")
        else:
            console.print(f"\n[yellow]Estimated Time:[/yellow]")
            console.print(f"  • {estimates['eta_minutes']:.1f} minutes")
            console.print(f"  • {estimates['eta_hours']:.1f} hours")
            console.print(f"  • {estimates['eta_days']:.2f} days")
            console.print(
                f"\n[yellow]Expected completion:[/yellow] "
                f"{estimates['completion_time'].strftime('%Y-%m-%d %H:%M:%S')}"
            )


if __name__ == "__main__":
    main()
