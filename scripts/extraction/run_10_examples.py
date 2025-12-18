"""Run 10 extraction examples with Langfuse tracing.

This script demonstrates batch extraction with full Langfuse observability.
All 10 extractions will be tracked in a single session in your Langfuse dashboard.

Usage:
    python scripts/extraction/run_10_examples.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

console = Console()


def get_sample_judgments():
    """Get 10 sample judgment texts for extraction."""
    return [
        {
            "id": 1,
            "name": "Sprawa cywilna - zapłata",
            "text": """
            WYROK W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ
            Dnia 15 stycznia 2024 r.
            Sąd Okręgowy w Warszawie, V Wydział Cywilny
            w składzie: Przewodniczący: SSO Anna Kowalska
            Sygn. akt V C 123/2023

            Sprawa z powództwa Jana Kowalskiego przeciwko Bankowi XYZ S.A. o zapłatę

            I. Zasądza od pozwanego Banku XYZ S.A. na rzecz powoda Jana Kowalskiego
            kwotę 50.000 zł wraz z odsetkami ustawowymi od dnia 1 stycznia 2023 r.
            """,
        },
        {
            "id": 2,
            "name": "Sprawa karna - kradzież",
            "text": """
            WYROK W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ
            Dnia 20 lutego 2024 r.
            Sąd Rejonowy w Krakowie, II Wydział Karny
            SSR Jan Nowak
            Sygn. akt II K 456/2023

            W sprawie Piotra Nowackiego oskarżonego o kradzież z art. 278 k.k.

            I. Uznaje oskarżonego za winnego popełnienia zarzucanego mu czynu
            II. Wymierza karę 2 lat pozbawienia wolności z warunkowym zawieszeniem
            """,
        },
        {
            "id": 3,
            "name": "Sprawa administracyjna",
            "text": """
            WYROK W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ
            Dnia 10 marca 2024 r.
            Wojewódzki Sąd Administracyjny w Gdańsku
            SSA Maria Wiśniewska
            Sygn. akt I SA 789/2023

            W sprawie skargi Adam Kowalczyk na decyzję Wójta Gminy Pruszcz

            I. Uchyla zaskarżoną decyzję
            II. Zasądza zwrot kosztów postępowania
            """,
        },
        {
            "id": 4,
            "name": "Sprawa rozwodowa",
            "text": """
            WYROK W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ
            Dnia 5 kwietnia 2024 r.
            Sąd Okręgowy w Poznaniu, III Wydział Rodzinny i Nieletnich
            SSO Katarzyna Lewandowska
            Sygn. akt III RC 234/2023

            W sprawie z powództwa Anny Nowak przeciwko Markowi Nowakowi o rozwód

            I. Orzeka rozwód małżonków
            II. Powierza wykonywanie władzy rodzicielskiej matce
            III. Zasądza alimenty na rzecz dziecka w wysokości 1.500 zł miesięcznie
            """,
        },
        {
            "id": 5,
            "name": "Sprawa odszkodowawcza",
            "text": """
            WYROK W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ
            Dnia 12 maja 2024 r.
            Sąd Apelacyjny w Katowicach, II Wydział Cywilny
            SSA Tomasz Zieliński
            Sygn. akt II ACa 567/2024

            W sprawie apelacji od wyroku Sądu Okręgowego w Katowicach
            Powód: Firma ABC Sp. z o.o., Pozwany: Ubezpieczyciel DEF S.A.

            I. Oddala apelację
            II. Zasądza od skarżącego koszty postępowania odwoławczego
            """,
        },
        {
            "id": 6,
            "name": "Sprawa o naruszenie dóbr osobistych",
            "text": """
            WYROK W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ
            Dnia 18 czerwca 2024 r.
            Sąd Okręgowy we Wrocławiu, I Wydział Cywilny
            SSO Paweł Kowalski
            Sygn. akt I C 890/2023

            Sprawa z powództwa Marii Lewandowskiej przeciwko Wydawnictwu XYZ
            o naruszenie dóbr osobistych przez publikację nieprawdziwych informacji

            I. Nakazuje pozwanemu opublikowanie przeprosin
            II. Zasądza zadośćuczynienie w kwocie 20.000 zł
            """,
        },
        {
            "id": 7,
            "name": "Sprawa o eksmisję",
            "text": """
            WYROK W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ
            Dnia 25 lipca 2024 r.
            Sąd Rejonowy w Gdańsku, IV Wydział Cywilny
            SSR Agnieszka Mazur
            Sygn. akt IV C 345/2024

            Sprawa z powództwa Jana Kowalczyka przeciwko Piotrowi Nowakowi
            o eksmisję z lokalu mieszkalnego przy ul. Długiej 15

            I. Nakazuje pozwanemu opróżnienie lokalu w terminie 3 miesięcy
            II. Przyznaje prawo do lokalu socjalnego
            """,
        },
        {
            "id": 8,
            "name": "Sprawa pracownicza",
            "text": """
            WYROK W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ
            Dnia 30 sierpnia 2024 r.
            Sąd Rejonowy w Łodzi, X Wydział Pracy
            SSR Ewa Nowacka
            Sygn. akt X P 678/2024

            Sprawa z powództwa Tomasza Wiśniewskiego przeciwko Firmie ABC Sp. z o.o.
            o przywrócenie do pracy i wynagrodzenie za czas pozostawania bez pracy

            I. Przywraca powoda do pracy na poprzednich warunkach
            II. Zasądza wynagrodzenie za 3 miesiące w kwocie 15.000 zł
            """,
        },
        {
            "id": 9,
            "name": "Sprawa o podział majątku",
            "text": """
            WYROK W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ
            Dnia 10 września 2024 r.
            Sąd Okręgowy w Białymstoku, II Wydział Cywilny
            SSO Michał Wojciechowski
            Sygn. akt II C 234/2024

            Sprawa z wniosku Anny Kowalskiej i Marka Kowalskiego
            o podział majątku wspólnego po ustaniu wspólności majątkowej małżeńskiej

            I. Przyznaje wnioskodawczyni nieruchomość przy ul. Lipowej 5
            II. Zasądza od wnioskodawczyni na rzecz uczestnika spłatę w kwocie 250.000 zł
            """,
        },
        {
            "id": 10,
            "name": "Sprawa karna - wypadek drogowy",
            "text": """
            WYROK W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ
            Dnia 15 października 2024 r.
            Sąd Rejonowy w Rzeszowie, III Wydział Karny
            SSR Katarzyna Krawczyk
            Sygn. akt III K 901/2024

            W sprawie Pawła Zielińskiego oskarżonego o spowodowanie wypadku
            drogowego z art. 177 § 1 k.k. w zw. z art. 178 k.k.

            I. Uznaje oskarżonego za winnego
            II. Wymierza karę 1 roku pozbawienia wolności z warunkowym zawieszeniem
            III. Orzeka zakaz prowadzenia pojazdów mechanicznych na okres 3 lat
            IV. Zasądza nawiązkę na rzecz pokrzywdzonego w kwocie 30.000 zł
            """,
        },
    ]


def main():
    """Run 10 extraction examples."""
    console.print(
        Panel.fit(
            "[bold cyan]10 Extraction Examples with Langfuse[/bold cyan]\n\n"
            "Running batch extraction with full observability",
            border_style="cyan",
        )
    )

    # Check environment
    console.print("\n[bold]Checking environment...[/bold]")
    required_vars = ["GOOGLE_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        console.print(f"[red]✗ Missing: {', '.join(missing)}[/red]")
        console.print("\n[yellow]Set in .env file or export manually[/yellow]")
        sys.exit(1)

    console.print("[green]✓ Environment configured[/green]")

    # Import modules
    try:
        from langfuse.langchain import CallbackHandler

        from juddges.extraction import GeminiExtractionChain
        from juddges.extraction.gemini_chain import DocumentType, ExtractionSchema
    except ImportError as e:
        console.print(f"[red]✗ Import failed: {e}[/red]")
        sys.exit(1)

    # Create session ID
    session_id = f"batch_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    console.print(f"\n[bold]Session ID:[/bold] [cyan]{session_id}[/cyan]")

    # Create chain
    console.print("\n[bold]Initializing extraction chain...[/bold]")
    chain = GeminiExtractionChain(
        model_name="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),  # Explicitly pass API key
        cache_path=".cache/batch_extraction.db",
        temperature=0.0,
    )

    # Define schema
    schema = ExtractionSchema(
        fields={
            "verdict_date": "date as ISO 8601, when verdict was issued",
            "court": "string, full name of the court",
            "case_number": "string, case signature/identifier",
            "case_type": "string, type of case (civil, criminal, administrative, family, etc.)",
            "verdict_summary": "string, brief summary of the verdict",
        },
        instructions="Extract key facts from Polish court judgments",
        language="polish",
    )

    # Get sample judgments
    judgments = get_sample_judgments()

    # Results storage
    results = []
    errors = []

    # Process with progress bar
    console.print(f"\n[bold]Extracting from {len(judgments)} judgments...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing...", total=len(judgments))

        for judgment in judgments:
            progress.update(
                task,
                description=f"[cyan]Processing: {judgment['name']}",
            )

            try:
                # Create handler for this extraction
                handler = CallbackHandler()

                # Extract
                result = chain.extract(
                    document_type=DocumentType.JUDGMENT,
                    text=judgment["text"],
                    schema=schema,
                    langfuse_handler=handler,
                )

                results.append(
                    {
                        "id": judgment["id"],
                        "name": judgment["name"],
                        "result": result,
                        "status": "success",
                    }
                )

            except Exception as e:
                console.print(f"\n[red]✗ Error on {judgment['name']}: {e}[/red]")
                errors.append({"id": judgment["id"], "name": judgment["name"], "error": str(e)})
                results.append(
                    {
                        "id": judgment["id"],
                        "name": judgment["name"],
                        "result": None,
                        "status": "error",
                    }
                )

            progress.advance(task)

    # Display results
    console.print("\n[bold green]✓ Extraction completed![/bold green]")
    console.print(f"\nProcessed: {len(results)}")
    console.print(f"Successful: {len([r for r in results if r['status'] == 'success'])}")
    console.print(f"Errors: {len(errors)}")

    # Show results table
    if results:
        console.print("\n[bold]Extraction Results:[/bold]")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("ID", style="dim", width=4)
        table.add_column("Case Name", width=30)
        table.add_column("Date", width=12)
        table.add_column("Court", width=30)
        table.add_column("Status", width=10)

        for r in results:
            if r["status"] == "success" and r["result"]:
                table.add_row(
                    str(r["id"]),
                    r["name"][:28] + "..." if len(r["name"]) > 28 else r["name"],
                    r["result"].get("verdict_date", "N/A"),
                    r["result"].get("court", "N/A")[:28] + "..."
                    if r["result"].get("court") and len(r["result"].get("court", "")) > 28
                    else r["result"].get("court", "N/A"),
                    "[green]✓[/green]",
                )
            else:
                table.add_row(
                    str(r["id"]),
                    r["name"][:28] + "..." if len(r["name"]) > 28 else r["name"],
                    "-",
                    "-",
                    "[red]✗[/red]",
                )

        console.print(table)

    # Show detailed successful extractions
    console.print("\n[bold]Sample Extractions:[/bold]")
    successful = [r for r in results if r["status"] == "success"][:3]  # Show first 3

    for r in successful:
        console.print(f"\n[cyan]━━━ {r['name']} ━━━[/cyan]")
        from rich.json import JSON

        console.print(JSON.from_data(r["result"], indent=2))

    # Show Langfuse dashboard info
    console.print("\n[bold green]✓ All extractions traced in Langfuse![/bold green]")
    console.print(f"\n[bold]View in Langfuse Dashboard:[/bold]")
    console.print(f"  URL: {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}")
    console.print(f"  Session: {session_id}")
    console.print("\n[dim]Filter by session ID to see all 10 extractions together[/dim]")

    # Summary stats
    if successful:
        console.print("\n[bold]Extraction Quality Summary:[/bold]")
        dates_extracted = sum(
            1 for r in successful if r["result"].get("verdict_date") and r["result"]["verdict_date"] != ""
        )
        courts_extracted = sum(
            1 for r in successful if r["result"].get("court") and r["result"]["court"] != ""
        )

        console.print(f"  Dates extracted: {dates_extracted}/{len(successful)}")
        console.print(f"  Courts extracted: {courts_extracted}/{len(successful)}")


if __name__ == "__main__":
    # Load .env file
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    main()
