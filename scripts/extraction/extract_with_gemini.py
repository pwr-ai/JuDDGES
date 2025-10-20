"""Example script for extracting information using Gemini 2.5 Pro.

This script demonstrates how to use the GeminiExtractionChain for extracting
structured information from legal documents with caching and Langfuse tracing.

Usage:
    # Basic usage without Langfuse
    python scripts/extraction/extract_with_gemini.py

    # With Langfuse tracing (set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY env vars)
    python scripts/extraction/extract_with_gemini.py --use-langfuse

    # Using tax interpretation documents
    python scripts/extraction/extract_with_gemini.py --document-type tax_interpretation

    # Batch processing
    python scripts/extraction/extract_with_gemini.py --batch-size 10
"""

import os
from pathlib import Path

from langfuse.langchain import CallbackHandler
from loguru import logger
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

from juddges.extraction import GeminiExtractionChain
from juddges.extraction.gemini_chain import DocumentType, ExtractionSchema

console = Console()


def create_judgment_schema() -> ExtractionSchema:
    """Create example schema for judgment extraction."""
    return ExtractionSchema(
        fields={
            "verdict_date": "date as ISO 8601, when the verdict was issued",
            "verdict": "string, text representing the verdict of the judgment",
            "verdict_summary": "string, concise summary of the verdict",
            "verdict_id": "string, official case identifier",
            "court": "string, name of the court that issued the judgment",
            "parties": "List[string], names of involved parties",
            "judge_names": "List[string], names of judges",
            "legal_basis": "List[string], referenced laws and articles",
            "appeal_basis": "string, basis for appeal if applicable",
        },
        instructions=(
            "Focus on extracting factual information only. "
            "For dates, ensure ISO 8601 format. "
            "For lists, include all mentioned items. "
            "If information is not explicitly stated, use empty values."
        ),
        language="polish",
    )


def create_tax_interpretation_schema() -> ExtractionSchema:
    """Create example schema for tax interpretation extraction."""
    return ExtractionSchema(
        fields={
            "interpretation_date": "date as ISO 8601, when issued",
            "interpretation_number": "string, official document number",
            "tax_authority": "string, issuing tax authority",
            "applicant": "string, who requested the interpretation",
            "subject_matter": "string, brief description of the tax issue",
            "legal_basis": "List[string], referenced tax laws and articles",
            "interpretation_content": "string, main content of the interpretation",
            "conclusion": "string, final ruling or conclusion",
        },
        instructions=(
            "Extract key information from the tax interpretation document. "
            "Focus on the legal basis and conclusion. "
            "Maintain accuracy of legal references."
        ),
        language="polish",
    )


def example_judgment_text() -> str:
    """Example judgment text for demonstration."""
    return """
    WYROK
    W IMIENIU RZECZYPOSPOLITEJ POLSKIEJ

    Dnia 15 stycznia 2024 r.

    Sąd Okręgowy w Warszawie, V Wydział Cywilny
    w składzie:
    Przewodniczący: SSO Anna Kowalska
    Sędziowie: SSO Jan Nowak, SSR del. Piotr Wiśniewski

    po rozpoznaniu w dniu 10 stycznia 2024 r. w Warszawie
    na rozprawie
    sprawy z powództwa Jana Kowalskiego
    przeciwko Bankowi XYZ S.A.
    o zapłatę

    I. Zasądza od pozwanego Banku XYZ S.A. na rzecz powoda Jana Kowalskiego
    kwotę 50.000 zł (pięćdziesiąt tysięcy złotych) wraz z odsetkami ustawowymi
    za opóźnienie od dnia 1 stycznia 2023 r. do dnia zapłaty.

    II. Zasądza od pozwanego na rzecz powoda kwotę 5.000 zł tytułem zwrotu
    kosztów procesu.

    UZASADNIENIE

    Powód wniósł o zasądzenie od pozwanego Banku kwoty 50.000 zł tytułem
    zwrotu nienależnie pobranych opłat za prowadzenie rachunku bankowego
    w latach 2020-2022.

    Sąd ustalił, że Bank pobierał opłaty niezgodne z umową, co stanowiło
    podstawę do uwzględnienia powództwa w całości zgodnie z art. 410 k.c.
    w związku z art. 405 k.c.
    """


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract information using Gemini 2.5")
    parser.add_argument(
        "--document-type",
        type=str,
        default="judgment",
        choices=["judgment", "tax_interpretation"],
        help="Type of document to process",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        choices=["gemini-2.5-pro", "gemini-2.5-flash"],
        help="Gemini model to use",
    )
    parser.add_argument(
        "--use-langfuse",
        action="store_true",
        help="Enable Langfuse tracing",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=".cache/extraction_gemini.db",
        help="Path to SQLite cache file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of documents to process (for demo)",
    )

    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold cyan]Gemini 2.5 Legal Document Extraction[/bold cyan]\n"
            f"Model: {args.model}\n"
            f"Document Type: {args.document_type}\n"
            f"Cache: {args.cache_path}",
            border_style="cyan",
        )
    )

    # Initialize chain
    chain = GeminiExtractionChain(
        model_name=args.model,
        cache_path=args.cache_path,
        temperature=0.0,
    )

    # Select document type and schema
    if args.document_type == "judgment":
        doc_type = DocumentType.JUDGMENT
        schema = create_judgment_schema()
        text = example_judgment_text()
    else:
        doc_type = DocumentType.TAX_INTERPRETATION
        schema = create_tax_interpretation_schema()
        text = "Tax interpretation example text..."  # Add your example

    # Optional: Initialize Langfuse
    langfuse_handler = None
    if args.use_langfuse:
        if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
            console.print(
                "[yellow]Warning: Langfuse keys not set. "
                "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY env vars.[/yellow]"
            )
        else:
            langfuse_handler = CallbackHandler()
            console.print("[green]✓[/green] Langfuse tracing enabled")

    # Single extraction
    if args.batch_size == 1:
        console.print("\n[bold]Extracting information...[/bold]")

        result = chain.extract(
            document_type=doc_type,
            text=text,
            schema=schema,
            langfuse_handler=langfuse_handler,
        )

        console.print("\n[bold green]Extraction Result:[/bold green]")
        console.print(JSON.from_data(result, indent=2))

    # Batch extraction demo
    else:
        console.print(f"\n[bold]Batch extracting from {args.batch_size} documents...[/bold]")

        # Create batch of example texts (in real use, load from dataset)
        texts = [text] * args.batch_size

        results = chain.batch_extract(
            document_type=doc_type,
            texts=texts,
            schema=schema,
            langfuse_handler=langfuse_handler,
        )

        console.print(f"\n[bold green]Extracted {len(results)} documents[/bold green]")
        console.print("\n[bold]First result:[/bold]")
        console.print(JSON.from_data(results[0], indent=2))

    console.print("\n[bold green]✓ Extraction completed successfully![/bold green]")

    # Cache info
    cache_path = Path(args.cache_path)
    if cache_path.exists():
        cache_size = cache_path.stat().st_size / 1024  # KB
        console.print(f"\n[dim]Cache size: {cache_size:.2f} KB at {cache_path}[/dim]")


if __name__ == "__main__":
    main()
