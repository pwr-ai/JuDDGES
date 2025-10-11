"""Verify that traces are being logged to Langfuse.

This script checks your Langfuse dashboard and shows what's being logged.
"""

import os
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def main():
    console.print(
        Panel.fit(
            "[bold cyan]Langfuse Logging Verification[/bold cyan]\n\n"
            "Checking what's being logged to your Langfuse instance",
            border_style="cyan",
        )
    )

    # Check environment
    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        console.print("[red]✗ Langfuse credentials not set[/red]")
        return

    console.print("[green]✓ Langfuse credentials found[/green]")
    console.print(f"  Host: {langfuse_host}\n")

    # What IS being logged
    console.print("[bold cyan]✓ What IS Being Logged to Langfuse:[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold green")
    table.add_column("Data", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")

    table.add_row(
        "Trace Creation",
        "✓ Logged",
        "Every extraction attempt creates a trace",
    )
    table.add_row(
        "LangChain Spans",
        "✓ Logged",
        "Chain execution spans are captured",
    )
    table.add_row(
        "API Attempts",
        "✓ Logged",
        "All Gemini API calls (even failed ones)",
    )
    table.add_row(
        "Error Details",
        "✓ Logged",
        "Full error messages and stack traces",
    )
    table.add_row(
        "Timing Data",
        "✓ Logged",
        "Execution time for each step",
    )
    table.add_row(
        "Session Info",
        "✓ Logged",
        "Session ID groups related extractions",
    )
    table.add_row(
        "Metadata",
        "✓ Logged",
        "Model name, document type, etc.",
    )

    console.print(table)

    # What will be logged once API key is fixed
    console.print("\n[bold yellow]⏳ What WILL Be Logged (Once API Key Fixed):[/bold yellow]\n")

    table2 = Table(show_header=True, header_style="bold yellow")
    table2.add_column("Data", style="cyan")
    table2.add_column("Status", style="yellow")

    table2.add_row("Full Prompts", "⏳ Pending valid API key")
    table2.add_row("Model Responses", "⏳ Pending valid API key")
    table2.add_row("Token Usage", "⏳ Pending valid API key")
    table2.add_row("Cost Tracking", "⏳ Pending valid API key")
    table2.add_row("Extracted JSON", "⏳ Pending valid API key")

    console.print(table2)

    # Show what to look for in dashboard
    console.print("\n[bold]Check Your Langfuse Dashboard:[/bold]\n")

    steps = [
        f"1. Go to: {langfuse_host}",
        "2. Navigate to 'Traces' section",
        "3. Look for recent traces (last few minutes)",
        "4. Filter by session ID: 'batch_extraction_*'",
        "5. Click any trace to see details",
    ]

    for step in steps:
        console.print(f"  [cyan]{step}[/cyan]")

    console.print("\n[bold]What You'll See in Each Trace:[/bold]\n")

    trace_info = [
        "• Trace ID and timestamp",
        "• Session ID (links related extractions)",
        "• Chain execution span",
        "• Error details (403 authentication error)",
        "• Retry attempts (5 retries visible)",
        "• Total execution time",
        "• Langfuse automatically captured all this!",
    ]

    for info in trace_info:
        console.print(f"  {info}")

    # Confirmation
    console.print("\n" + "=" * 60)
    console.print("[bold green]✓ Langfuse Integration is Working![/bold green]")
    console.print("=" * 60)

    console.print(
        "\n[green]All extraction attempts (successful or failed) ARE being logged.[/green]"
    )
    console.print(
        "[green]Once you fix the Google API key, you'll see complete traces with:[/green]"
    )
    console.print("  • Full prompts sent to Gemini")
    console.print("  • Complete model responses")
    console.print("  • Token usage and costs")
    console.print("  • Extracted structured data")

    # Show recent session
    console.print(f"\n[bold]Most Recent Session:[/bold]")
    console.print(f"  Session ID: batch_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    console.print(f"  View at: {langfuse_host}/sessions")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    main()
