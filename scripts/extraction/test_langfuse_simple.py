"""Simple Langfuse integration test for Gemini extraction.

This is a simplified test that works with the current Langfuse API.

Usage:
    # Make sure .env has your credentials set
    python scripts/extraction/test_langfuse_simple.py
"""

import os
import sys
from pathlib import Path

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

console = Console()


def main():
    """Run simple Langfuse test."""
    console.print(
        Panel.fit(
            "[bold cyan]Langfuse + Gemini Extraction - Simple Test[/bold cyan]\n\n"
            "Testing basic Langfuse observability integration",
            border_style="cyan",
        )
    )

    # Check environment
    console.print("\n[bold]Checking environment...[/bold]")
    required_vars = ["GOOGLE_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        console.print(f"[red]✗ Missing environment variables: {', '.join(missing)}[/red]")
        console.print("\n[yellow]Make sure .env file has:[/yellow]")
        for var in missing:
            console.print(f"  {var}=...")
        sys.exit(1)

    console.print("[green]✓ All environment variables set[/green]")
    console.print(f"  Host: {os.getenv('LANGFUSE_HOST')}")

    # Import
    console.print("\n[bold]Importing modules...[/bold]")
    try:
        from langfuse.langchain import CallbackHandler

        from juddges.extraction import GeminiExtractionChain
        from juddges.extraction.gemini_chain import DocumentType, ExtractionSchema

        console.print("[green]✓ Modules imported successfully[/green]")
    except ImportError as e:
        console.print(f"[red]✗ Import failed: {e}[/red]")
        sys.exit(1)

    # Test connection
    console.print("\n[bold]Testing Langfuse connection...[/bold]")
    try:
        handler = CallbackHandler()
        console.print("[green]✓ Langfuse handler created successfully[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to create Langfuse handler: {e}[/red]")
        sys.exit(1)

    # Create extraction chain
    console.print("\n[bold]Creating Gemini extraction chain...[/bold]")
    chain = GeminiExtractionChain(
        model_name="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),  # Explicitly pass API key
        cache_path=".cache/langfuse_simple_test.db",
        temperature=0.0,
    )
    console.print("[green]✓ Chain created[/green]")

    # Define schema
    schema = ExtractionSchema(
        fields={
            "verdict_date": "date as ISO 8601",
            "court": "string, name of the court",
            "case_number": "string, case identifier",
        },
        instructions="Extract factual information only",
        language="polish",
    )

    # Sample judgment
    judgment_text = """
    WYROK
    W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ

    Dnia 15 stycznia 2024 r.

    Sąd Okręgowy w Warszawie, V Wydział Cywilny
    Sygn. akt V C 123/2023

    I. Zasądza od pozwanego na rzecz powoda kwotę 50.000 zł.
    """

    console.print("\n[bold]Extracting with Langfuse tracing...[/bold]")
    console.print("[dim]This extraction will be traced in your Langfuse dashboard[/dim]")

    try:
        # Extract with Langfuse handler
        result = chain.extract(
            document_type=DocumentType.JUDGMENT,
            text=judgment_text,
            schema=schema,
            langfuse_handler=handler,
        )

        console.print("\n[bold green]✓ Extraction successful![/bold green]")
        console.print("\n[bold]Extracted Data:[/bold]")
        console.print(JSON.from_data(result, indent=2))

        console.print("\n[bold green]✓ Test completed successfully![/bold green]")
        console.print(f"\n[bold]View trace in Langfuse:[/bold]")
        console.print(f"  {os.getenv('LANGFUSE_HOST')}")
        console.print("\n[dim]Look for the most recent trace in your dashboard[/dim]")

    except Exception as e:
        console.print(f"\n[red]✗ Extraction failed: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    # Load .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv

        load_dotenv()
        console.print("[dim]Loaded environment from .env file[/dim]")
    except ImportError:
        pass  # python-dotenv not installed, assume env vars are set

    main()
