"""Test Langfuse integration with Gemini extraction chain.

This script demonstrates and tests Langfuse observability integration.

Prerequisites:
    1. Langfuse account: https://cloud.langfuse.com (or self-hosted)
    2. Create a project and get API keys
    3. Set environment variables:
       - LANGFUSE_PUBLIC_KEY
       - LANGFUSE_SECRET_KEY
       - LANGFUSE_HOST (optional, defaults to https://cloud.langfuse.com)
       - GOOGLE_API_KEY (for Gemini)

Usage:
    export GOOGLE_API_KEY="your-google-key"
    export LANGFUSE_PUBLIC_KEY="pk-lf-..."
    export LANGFUSE_SECRET_KEY="sk-lf-..."

    python scripts/extraction/test_langfuse.py
"""

import os
import sys
from pathlib import Path
from time import sleep

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

console = Console()


def check_environment() -> tuple[bool, bool]:
    """Check if required environment variables are set.

    Returns:
        (google_api_key_set, langfuse_keys_set)
    """
    google_key = bool(os.getenv("GOOGLE_API_KEY"))
    langfuse_public = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret = bool(os.getenv("LANGFUSE_SECRET_KEY"))

    table = Table(title="Environment Check")
    table.add_column("Variable", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Value", style="dim")

    def status(is_set):
        return "✓ Set" if is_set else "✗ Not Set"

    def show_val(key):
        val = os.getenv(key, "")
        if val:
            return f"{val[:20]}..." if len(val) > 20 else val
        return "[red]missing[/red]"

    table.add_row("GOOGLE_API_KEY", status(google_key), show_val("GOOGLE_API_KEY"))
    table.add_row("LANGFUSE_PUBLIC_KEY", status(langfuse_public), show_val("LANGFUSE_PUBLIC_KEY"))
    table.add_row("LANGFUSE_SECRET_KEY", status(langfuse_secret), show_val("LANGFUSE_SECRET_KEY"))
    table.add_row(
        "LANGFUSE_HOST",
        "✓ Set" if os.getenv("LANGFUSE_HOST") else "Default",
        os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

    console.print(table)

    return google_key, (langfuse_public and langfuse_secret)


def test_langfuse_import():
    """Test that Langfuse can be imported."""
    console.print("\n[bold cyan]Test 1: Import Langfuse[/bold cyan]")

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        console.print("[green]✓ Langfuse imported successfully[/green]")
        console.print(f"[dim]Langfuse module: {Langfuse.__module__}[/dim]")
        console.print(f"[dim]CallbackHandler: {CallbackHandler.__module__}[/dim]")
        return True
    except ImportError as e:
        console.print(f"[red]✗ Failed to import Langfuse: {e}[/red]")
        console.print("\n[yellow]Install Langfuse:[/yellow]")
        console.print("  uv pip install langfuse")
        return False


def test_langfuse_connection():
    """Test connection to Langfuse server."""
    console.print("\n[bold cyan]Test 2: Connect to Langfuse[/bold cyan]")

    from langfuse.langchain import CallbackHandler

    try:
        # Try to create a callback handler (this will validate credentials)
        console.print("[yellow]Testing Langfuse connection...[/yellow]")

        handler = CallbackHandler()

        console.print("[green]✓ Successfully connected to Langfuse[/green]")
        console.print(f"[dim]Host: {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}[/dim]")
        console.print(f"[dim]Public Key: {os.getenv('LANGFUSE_PUBLIC_KEY', '')[:20]}...[/dim]")

        return True
    except Exception as e:
        console.print(f"[red]✗ Failed to connect to Langfuse: {e}[/red]")
        console.print("\n[yellow]Check your credentials:[/yellow]")
        console.print("  - LANGFUSE_PUBLIC_KEY should start with 'pk-lf-'")
        console.print("  - LANGFUSE_SECRET_KEY should start with 'sk-lf-'")
        console.print(f"  - LANGFUSE_HOST should be accessible: {os.getenv('LANGFUSE_HOST', 'N/A')}")
        return False


def test_extraction_with_langfuse():
    """Test extraction with Langfuse tracing."""
    console.print("\n[bold cyan]Test 3: Extract with Langfuse Tracing[/bold cyan]")

    from langfuse.langchain import CallbackHandler

    from juddges.extraction import GeminiExtractionChain
    from juddges.extraction.gemini_chain import DocumentType, ExtractionSchema

    # Create Langfuse handler
    console.print("[yellow]Creating Langfuse callback handler...[/yellow]")
    handler = CallbackHandler(
        trace_name="gemini_extraction_test",
        session_id="test_session",
        user_id="test_user",
        metadata={
            "test": True,
            "environment": "development",
            "model": "gemini-2.5-flash",
        },
        tags=["test", "extraction", "gemini"],
    )

    console.print("[green]✓ Langfuse handler created[/green]")

    # Create extraction chain
    console.print("\n[yellow]Creating Gemini extraction chain...[/yellow]")
    chain = GeminiExtractionChain(
        model_name="gemini-2.5-flash",
        cache_path=".cache/langfuse_test.db",
        temperature=0.0,
    )
    console.print("[green]✓ Chain created[/green]")

    # Define schema
    schema = ExtractionSchema(
        fields={
            "verdict_date": "date as ISO 8601, when verdict was issued",
            "court": "string, name of the court",
            "case_number": "string, case identifier",
        },
        instructions="Extract factual information from the judgment",
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

    console.print("\n[yellow]Extracting with Langfuse tracing...[/yellow]")
    console.print("[dim]This will appear in your Langfuse dashboard[/dim]")

    try:
        result = chain.extract(
            document_type=DocumentType.JUDGMENT,
            text=judgment_text,
            schema=schema,
            langfuse_handler=handler,
        )

        console.print("\n[bold green]✓ Extraction successful![/bold green]")
        console.print("\n[bold]Extracted Data:[/bold]")
        from rich.json import JSON
        console.print(JSON.from_data(result, indent=2))

        # Flush Langfuse to ensure trace is sent
        console.print("\n[yellow]Flushing Langfuse traces...[/yellow]")
        handler.langfuse.flush()

        console.print("[green]✓ Trace sent to Langfuse[/green]")

        # Get trace info
        trace_url = f"{os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}/traces"
        console.print(f"\n[bold]View in Langfuse:[/bold] {trace_url}")

        return True

    except Exception as e:
        console.print(f"\n[red]✗ Extraction failed: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return False


def test_multiple_traces():
    """Test multiple extractions with Langfuse tracing."""
    console.print("\n[bold cyan]Test 4: Multiple Traces[/bold cyan]")
    console.print("[dim]Testing session tracking across multiple extractions[/dim]")

    from langfuse.langchain import CallbackHandler

    from juddges.extraction import GeminiExtractionChain
    from juddges.extraction.gemini_chain import DocumentType, ExtractionSchema

    chain = GeminiExtractionChain(model_name="gemini-2.5-flash")
    schema = ExtractionSchema(
        fields={"court": "string, court name"},
        language="polish",
    )

    session_id = "test_batch_session"

    judgments = [
        ("Judgment 1", "Wyrok Sądu Okręgowego w Warszawie z dnia 2024-01-15"),
        ("Judgment 2", "Wyrok Sądu Rejonowego w Krakowie z dnia 2024-02-20"),
        ("Judgment 3", "Wyrok Sądu Apelacyjnego w Gdańsku z dnia 2024-03-10"),
    ]

    console.print(f"\n[yellow]Processing {len(judgments)} judgments with session: {session_id}[/yellow]")

    try:
        for i, (name, text) in enumerate(judgments, 1):
            console.print(f"\n[cyan]Processing {name}...[/cyan]")

            # Create handler for this extraction
            handler = CallbackHandler(
                trace_name=f"extraction_{i}",
                session_id=session_id,
                metadata={"judgment": name, "index": i},
                tags=["batch", "test"],
            )

            result = chain.extract(
                document_type=DocumentType.JUDGMENT,
                text=text,
                schema=schema,
                langfuse_handler=handler,
            )

            console.print(f"[green]✓ {name}: {result.get('court', 'N/A')}[/green]")

            # Flush after each extraction
            handler.langfuse.flush()

        console.print(f"\n[bold green]✓ All {len(judgments)} extractions completed![/bold green]")
        console.print(f"\n[bold]View session in Langfuse:[/bold]")
        console.print(f"  Session ID: {session_id}")
        console.print(f"  Look for traces with tags: ['batch', 'test']")

        return True

    except Exception as e:
        console.print(f"[red]✗ Batch processing failed: {e}[/red]")
        return False


def show_langfuse_dashboard_guide():
    """Show guide for viewing traces in Langfuse dashboard."""
    console.print("\n" + "="*60)
    console.print("[bold cyan]Viewing Traces in Langfuse Dashboard[/bold cyan]")
    console.print("="*60)

    langfuse_host = os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')

    console.print(f"\n1. Open Langfuse: {langfuse_host}")
    console.print("\n2. Navigate to your project")
    console.print("\n3. Go to 'Traces' section")
    console.print("\n4. Look for traces with:")
    console.print("   • Name: 'gemini_extraction_test'")
    console.print("   • Tags: ['test', 'extraction', 'gemini']")
    console.print("   • Session: 'test_session' or 'test_batch_session'")
    console.print("\n5. Click on a trace to see:")
    console.print("   • Input prompt sent to Gemini")
    console.print("   • Model response")
    console.print("   • Execution time")
    console.print("   • Token usage")
    console.print("   • Metadata")
    console.print("\n6. Analyze:")
    console.print("   • Performance metrics")
    console.print("   • Cost per extraction")
    console.print("   • Error rates")
    console.print("   • Session timeline")

    console.print("\n[bold]Useful Filters:[/bold]")
    console.print("  • Filter by tag: test")
    console.print("  • Filter by session: test_session")
    console.print("  • Filter by user: test_user")
    console.print("  • Sort by: timestamp, duration, cost")


def main():
    """Run all Langfuse integration tests."""
    console.print(
        Panel.fit(
            "[bold cyan]Langfuse Integration Test Suite[/bold cyan]\n\n"
            "Testing Langfuse observability with Gemini extraction chain",
            border_style="cyan",
        )
    )

    # Check environment
    google_key_set, langfuse_keys_set = check_environment()

    if not google_key_set:
        console.print("\n[red]✗ GOOGLE_API_KEY not set[/red]")
        console.print("Get your key from: https://ai.google.dev/gemini-api/docs/api-key")
        console.print("Then set: export GOOGLE_API_KEY='your-key'")
        sys.exit(1)

    if not langfuse_keys_set:
        console.print("\n[red]✗ Langfuse keys not set[/red]")
        console.print("\nTo get Langfuse keys:")
        console.print("1. Sign up at: https://cloud.langfuse.com")
        console.print("2. Create a project")
        console.print("3. Go to Settings > API Keys")
        console.print("4. Copy your keys and set:")
        console.print("   export LANGFUSE_PUBLIC_KEY='pk-lf-...'")
        console.print("   export LANGFUSE_SECRET_KEY='sk-lf-...'")
        sys.exit(1)

    console.print("\n[bold green]✓ All environment variables set![/bold green]")

    # Run tests
    results = {}

    # Test 1: Import
    results["import"] = test_langfuse_import()
    if not results["import"]:
        console.print("\n[red]Stopping tests - Langfuse not installed[/red]")
        sys.exit(1)

    # Test 2: Connection
    results["connection"] = test_langfuse_connection()
    if not results["connection"]:
        console.print("\n[red]Stopping tests - Cannot connect to Langfuse[/red]")
        sys.exit(1)

    # Test 3: Extraction with tracing
    console.print("\n[yellow]Waiting 2 seconds before extraction test...[/yellow]")
    sleep(2)
    results["extraction"] = test_extraction_with_langfuse()

    # Test 4: Multiple traces
    if results["extraction"]:
        console.print("\n[yellow]Waiting 2 seconds before batch test...[/yellow]")
        sleep(2)
        results["multiple"] = test_multiple_traces()

    # Summary
    console.print("\n\n" + "="*60)
    console.print("[bold]Test Summary[/bold]")
    console.print("="*60 + "\n")

    for name, passed in results.items():
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        console.print(f"  {status} - {name.replace('_', ' ').title()}")

    total = len(results)
    passed = sum(1 for r in results.values() if r)

    console.print(f"\n[bold]Total: {passed}/{total} tests passed[/bold]")

    if passed == total:
        console.print("\n[bold green]🎉 All Langfuse tests passed![/bold green]")
        show_langfuse_dashboard_guide()
    else:
        console.print(f"\n[bold yellow]⚠ {total - passed} test(s) failed[/bold yellow]")


if __name__ == "__main__":
    main()
