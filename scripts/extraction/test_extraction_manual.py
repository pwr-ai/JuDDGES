"""Manual test script for Gemini extraction chain.

This script allows you to test the extraction chain interactively
without pytest, making it easier to debug and see results.

Usage:
    export GOOGLE_API_KEY="your-key"
    python scripts/extraction/test_extraction_manual.py
"""

import os
import sys
from pathlib import Path

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from juddges.extraction import GeminiExtractionChain
from juddges.extraction.gemini_chain import DocumentType, ExtractionSchema

console = Console()


def check_api_key() -> bool:
    """Check if API key is set."""
    if not os.getenv("GOOGLE_API_KEY"):
        console.print(
            "[red]✗ GOOGLE_API_KEY not set![/red]\n"
            "Please set your API key:\n"
            "  export GOOGLE_API_KEY='your-api-key'\n\n"
            "Get your key from: https://ai.google.dev/gemini-api/docs/api-key"
        )
        return False
    console.print("[green]✓ GOOGLE_API_KEY found[/green]")
    return True


def test_basic_extraction():
    """Test basic extraction functionality."""
    console.print("\n[bold cyan]Test 1: Basic Judgment Extraction[/bold cyan]")

    # Sample judgment text
    judgment_text = """
    WYROK
    W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ

    Dnia 15 stycznia 2024 r.

    Sąd Okręgowy w Warszawie, V Wydział Cywilny
    w składzie:
    Przewodniczący: SSO Anna Kowalska
    Sędziowie: SSO Jan Nowak, SSR del. Piotr Wiśniewski

    Sprawa z powództwa Jana Kowalskiego
    przeciwko Bankowi XYZ S.A.
    o zapłatę

    Sygn. akt V C 123/2023

    I. Zasądza od pozwanego Banku XYZ S.A. na rzecz powoda Jana Kowalskiego
    kwotę 50.000 zł wraz z odsetkami ustawowymi.
    """

    console.print("\n[dim]Judgment text:[/dim]")
    console.print(Panel(judgment_text.strip(), border_style="dim"))

    # Create chain
    console.print("\n[yellow]Creating extraction chain...[/yellow]")
    chain = GeminiExtractionChain(
        model_name="gemini-2.5-flash",
        cache_path=".cache/manual_test.db",
        temperature=0.0,
    )

    # Define schema
    schema = ExtractionSchema(
        fields={
            "verdict_date": "date as ISO 8601, when the verdict was issued",
            "court": "string, full name of the court",
            "case_number": "string, case signature/identifier",
            "judge_names": "List[string], names of all judges",
            "parties": "List[string], all involved parties",
        },
        instructions="Extract all factual information present in the text",
        language="polish",
    )

    console.print("[yellow]Schema fields:[/yellow]")
    for field, desc in schema.fields.items():
        console.print(f"  • {field}: [dim]{desc}[/dim]")

    # Extract
    console.print("\n[yellow]Calling Gemini API for extraction...[/yellow]")
    try:
        result = chain.extract(
            document_type=DocumentType.JUDGMENT,
            text=judgment_text,
            schema=schema,
        )

        console.print("\n[bold green]✓ Extraction successful![/bold green]")
        console.print("\n[bold]Extracted Information:[/bold]")
        console.print(JSON.from_data(result, indent=2))

        return True

    except Exception as e:
        console.print(f"\n[bold red]✗ Extraction failed:[/bold red] {e}")
        import traceback

        console.print(traceback.format_exc())
        return False


def test_caching():
    """Test caching functionality."""
    console.print("\n\n[bold cyan]Test 2: Caching Performance[/bold cyan]")

    if not Confirm.ask("Run caching test? (makes 2 identical API calls)"):
        return True

    import time

    chain = GeminiExtractionChain(
        model_name="gemini-2.5-flash",
        cache_path=".cache/manual_test.db",
    )

    schema = ExtractionSchema(
        fields={"court": "string, court name"},
        language="polish",
    )

    text = "Sąd Okręgowy w Warszawie wydał wyrok dnia 2024-01-15"

    # First call
    console.print("\n[yellow]First call (API)...[/yellow]")
    start = time.time()
    result1 = chain.extract(DocumentType.JUDGMENT, text, schema)
    time1 = time.time() - start
    console.print(f"[dim]Time: {time1:.3f}s[/dim]")

    # Second call
    console.print("\n[yellow]Second call (should hit cache)...[/yellow]")
    start = time.time()
    result2 = chain.extract(DocumentType.JUDGMENT, text, schema)
    time2 = time.time() - start
    console.print(f"[dim]Time: {time2:.3f}s[/dim]")

    # Compare
    if result1 == result2:
        console.print("[green]✓ Results match[/green]")
    else:
        console.print("[red]✗ Results differ![/red]")
        return False

    if time2 < time1 * 0.5:
        console.print(
            f"[green]✓ Cache speedup: {time1/time2:.1f}x faster[/green]"
        )
    else:
        console.print(
            f"[yellow]⚠ Cache might not be working (speedup: {time1/time2:.1f}x)[/yellow]"
        )

    return True


def test_batch_extraction():
    """Test batch extraction."""
    console.print("\n\n[bold cyan]Test 3: Batch Extraction[/bold cyan]")

    if not Confirm.ask("Run batch extraction test? (processes 3 judgments)"):
        return True

    chain = GeminiExtractionChain(model_name="gemini-2.5-flash")

    schema = ExtractionSchema(
        fields={
            "verdict_date": "date as ISO 8601",
            "court": "string, court name",
        },
        language="polish",
    )

    texts = [
        "Wyrok Sądu Okręgowego w Warszawie z dnia 2024-01-15 w sprawie I C 1/2024",
        "Wyrok Sądu Rejonowego w Krakowie z dnia 2024-02-20 w sprawie II C 2/2024",
        "Wyrok Sądu Apelacyjnego w Gdańsku z dnia 2024-03-10 w sprawie III C 3/2024",
    ]

    console.print(f"\n[yellow]Processing {len(texts)} judgments...[/yellow]")

    try:
        results = chain.batch_extract(
            document_type=DocumentType.JUDGMENT,
            texts=texts,
            schema=schema,
        )

        console.print(f"\n[green]✓ Processed {len(results)} documents[/green]")
        for i, result in enumerate(results, 1):
            console.print(f"\n[bold]Document {i}:[/bold]")
            console.print(JSON.from_data(result, indent=2))

        return True

    except Exception as e:
        console.print(f"[red]✗ Batch extraction failed: {e}[/red]")
        return False


def test_with_real_judgment():
    """Test with real judgment from dataset."""
    console.print("\n\n[bold cyan]Test 4: Real Judgment Data[/bold cyan]")

    sample_file = Path("data/sample_data/judgements-konfiskata-10-sample.csv")

    if not sample_file.exists():
        console.print(
            f"[yellow]⚠ Sample file not found: {sample_file}[/yellow]\n"
            "  Run 'dvc pull' to download sample data"
        )
        return True

    if not Confirm.ask(f"Test with real judgment from {sample_file.name}?"):
        return True

    import pandas as pd

    console.print(f"\n[yellow]Loading sample data from {sample_file}...[/yellow]")
    df = pd.read_csv(sample_file)

    if "excerpt" not in df.columns:
        console.print("[red]✗ 'excerpt' column not found in CSV[/red]")
        return False

    # Get first judgment with content
    judgment_row = df[df["excerpt"].notna()].iloc[0]
    judgment_text = str(judgment_row["excerpt"])[:5000]  # First 5000 chars

    console.print(f"\n[dim]Using judgment: {judgment_row.get('signature', 'N/A')}[/dim]")
    console.print(f"[dim]Text length: {len(judgment_text)} chars[/dim]")

    chain = GeminiExtractionChain(model_name="gemini-2.5-flash")

    schema = ExtractionSchema(
        fields={
            "verdict_date": "date as ISO 8601",
            "court": "string, court name",
            "case_type": "string, type of case (criminal, civil, etc.)",
            "summary": "string, brief summary of the case",
        },
        language="polish",
    )

    console.print("\n[yellow]Extracting...[/yellow]")

    try:
        result = chain.extract(
            document_type=DocumentType.JUDGMENT,
            text=judgment_text,
            schema=schema,
        )

        console.print("\n[bold green]✓ Extraction successful![/bold green]")
        console.print("\n[bold]Result:[/bold]")
        console.print(JSON.from_data(result, indent=2))

        return True

    except Exception as e:
        console.print(f"[red]✗ Extraction failed: {e}[/red]")
        return False


def main():
    """Run all tests."""
    console.print(
        Panel.fit(
            "[bold cyan]Gemini Extraction Chain - Manual Test Suite[/bold cyan]\n\n"
            "This script tests the extraction chain with real API calls.\n"
            "Each test can be run independently.",
            border_style="cyan",
        )
    )

    # Check prerequisites
    if not check_api_key():
        sys.exit(1)

    # Run tests
    tests = [
        ("Basic Extraction", test_basic_extraction),
        ("Caching", test_caching),
        ("Batch Processing", test_batch_extraction),
        ("Real Data", test_with_real_judgment),
    ]

    results = {}
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = success
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Tests interrupted by user[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]✗ Test failed with error: {e}[/red]")
            results[name] = False

    # Summary
    console.print("\n\n" + "=" * 60)
    console.print("[bold]Test Summary:[/bold]\n")

    for name, success in results.items():
        status = "[green]✓ PASS[/green]" if success else "[red]✗ FAIL[/red]"
        console.print(f"  {status} - {name}")

    total = len(results)
    passed = sum(1 for s in results.values() if s)

    console.print(
        f"\n[bold]Total: {passed}/{total} tests passed[/bold]"
    )

    if passed == total:
        console.print("\n[bold green]🎉 All tests passed![/bold green]")
    else:
        console.print(
            f"\n[bold yellow]⚠ {total - passed} test(s) failed[/bold yellow]"
        )


if __name__ == "__main__":
    main()
