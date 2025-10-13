#!/usr/bin/env python3
"""
Add extracted_factual_state and extracted_legal_state properties to LegalDocuments collection.

This script adds new properties needed for storing LLM-extracted information.
"""

import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from loguru import logger

from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase
from juddges.settings import ROOT_PATH
import weaviate.classes.config as wvcc

load_dotenv(ROOT_PATH / ".env", override=True)


def add_extraction_properties():
    """Add new properties for extraction fields."""
    console = Console()

    # Header
    console.print(
        Panel.fit(
            "🔧 Add Extraction Properties to Weaviate Schema",
            style="bold blue",
            border_style="bright_blue",
        )
    )

    console.print(
        "\n[cyan]This script will add:[/cyan]\n"
        "  1. factual_state (TEXT) - Factual circumstances\n"
        "  2. legal_state (TEXT) - Legal basis and applicable law\n"
    )

    try:
        with WeaviateLegalDocumentsDatabase() as db:
            console.print("✅ [green]Connected to Weaviate[/green]")

            collection = db.legal_documents_collection

            # Check existing properties
            existing_props = db.legal_documents_properties
            console.print(f"\n[cyan]Current properties:[/cyan] {len(existing_props)} total")

            # Track additions
            added_props = []

            # Add factual_state
            if "factual_state" not in existing_props:
                console.print("\n[yellow]Adding property: factual_state...[/yellow]")
                try:
                    collection.config.add_property(
                        wvcc.Property(
                            name="factual_state",
                            data_type=wvcc.DataType.TEXT,
                            description="Factual circumstances and background of the case (LLM-extracted)",
                            skip_vectorization=False,  # Enable semantic search
                            vectorize_property_name=False,
                            index_searchable=True,
                        )
                    )
                    console.print("[green]✓ Added factual_state[/green]")
                    added_props.append("factual_state")
                except Exception as e:
                    console.print(f"[red]✗ Failed to add factual_state: {e}[/red]")
            else:
                console.print("[dim]• factual_state already exists[/dim]")

            # Add legal_state
            if "legal_state" not in existing_props:
                console.print("\n[yellow]Adding property: legal_state...[/yellow]")
                try:
                    collection.config.add_property(
                        wvcc.Property(
                            name="legal_state",
                            data_type=wvcc.DataType.TEXT,
                            description="Legal basis and applicable law (LLM-extracted)",
                            skip_vectorization=False,  # Enable semantic search
                            vectorize_property_name=False,
                            index_searchable=True,
                        )
                    )
                    console.print("[green]✓ Added legal_state[/green]")
                    added_props.append("legal_state")
                except Exception as e:
                    console.print(f"[red]✗ Failed to add legal_state: {e}[/red]")
            else:
                console.print("[dim]• legal_state already exists[/dim]")

            # Verify changes
            console.print("\n[cyan]Verifying changes...[/cyan]")
            updated_props = db.legal_documents_properties
            console.print(f"  • Total properties: {len(updated_props)}")

            verification_passed = True

            if "factual_state" in updated_props:
                console.print("[green]  ✓ factual_state confirmed[/green]")
            elif "factual_state" in added_props:
                console.print("[red]  ✗ factual_state not found after addition[/red]")
                verification_passed = False

            if "legal_state" in updated_props:
                console.print("[green]  ✓ legal_state confirmed[/green]")
            elif "legal_state" in added_props:
                console.print("[red]  ✗ legal_state not found after addition[/red]")
                verification_passed = False

            # Summary
            console.print("\n" + "=" * 60)
            if added_props:
                console.print(
                    f"[bold green]✅ Successfully added {len(added_props)} properties![/bold green]"
                )
                console.print("\n[cyan]Added properties:[/cyan]")
                for prop in added_props:
                    console.print(f"  • {prop}")
            else:
                console.print(
                    "[yellow]ℹ️  No new properties were added (all already exist)[/yellow]"
                )

            if not verification_passed:
                console.print(
                    "\n[bold red]⚠️  Verification failed - some properties may not have been added correctly[/bold red]"
                )
                sys.exit(1)

            console.print(
                "\n[bold green]You can now ingest extracted data to these properties![/bold green]"
            )

    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        logger.exception("Schema update failed")
        sys.exit(1)


if __name__ == "__main__":
    add_extraction_properties()
