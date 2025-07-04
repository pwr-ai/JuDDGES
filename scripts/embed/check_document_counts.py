#!/usr/bin/env python3
"""
Check document counts in Weaviate database collections.
This script uses the existing WeaviateLegalDocumentsDatabase class to query collection sizes.
"""

import sys
from typing import Dict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase


def check_document_counts() -> Dict[str, int]:
    """
    Check document counts in both legal_documents and document_chunks collections.
    
    Returns:
        Dict[str, int]: Dictionary with collection names as keys and document counts as values.
    """
    console = Console()
    
    # Header
    console.print(
        Panel.fit("🗄️ Weaviate Document Count Check", style="bold blue", border_style="bright_blue")
    )
    
    counts = {}
    
    try:
        # Initialize database connection
        with WeaviateLegalDocumentsDatabase() as db:
            console.print("✅ [green]Connected to Weaviate database[/green]")
            
            # Check legal_documents collection
            try:
                legal_docs_count = db.get_collection_size(db.legal_documents_collection)
                counts["legal_documents"] = legal_docs_count
                console.print(f"📊 [cyan]Legal documents collection:[/cyan] [bold yellow]{legal_docs_count:,}[/bold yellow] documents")
            except Exception as e:
                console.print(f"❌ [red]Error getting legal documents count:[/red] {e}")
                counts["legal_documents"] = 0
            
            # Check document_chunks collection
            try:
                chunks_count = db.get_collection_size(db.document_chunks_collection)
                counts["document_chunks"] = chunks_count
                console.print(f"📄 [cyan]Document chunks collection:[/cyan] [bold yellow]{chunks_count:,}[/bold yellow] chunks")
            except Exception as e:
                console.print(f"❌ [red]Error getting document chunks count:[/red] {e}")
                counts["document_chunks"] = 0
            
            # Summary table
            table = Table(
                title="📊 Collection Summary", 
                show_header=True, 
                header_style="bold magenta",
                border_style="bright_blue"
            )
            table.add_column("Collection", style="cyan", width=20)
            table.add_column("Count", style="yellow", width=15, justify="right")
            table.add_column("Type", style="white", width=25)
            
            table.add_row("⚖️ legal_documents", f"{counts['legal_documents']:,}", "Full legal documents")
            table.add_row("📄 document_chunks", f"{counts['document_chunks']:,}", "Text chunks for search")
            
            console.print(table)
            
            # Total summary
            total_count = counts["legal_documents"] + counts["document_chunks"]
            summary_text = Text()
            summary_text.append("📊 Total Objects: ", style="bold cyan")
            summary_text.append(f"{total_count:,}", style="bold yellow")
            summary_text.append("\n🏗️ Collections: ", style="bold cyan")
            summary_text.append("2", style="bold yellow")
            
            console.print(Panel(summary_text, title="📈 Database Summary", border_style="green"))
            
            # Show ratio if both collections have data
            if counts["legal_documents"] > 0 and counts["document_chunks"] > 0:
                ratio = counts["document_chunks"] / counts["legal_documents"]
                console.print(f"\n📊 [bold green]Chunks per document ratio:[/bold green] [yellow]{ratio:.1f}[/yellow] chunks per document")
            
    except Exception as e:
        console.print(f"❌ [red]Database connection error:[/red] {e}")
        console.print("\n💡 [yellow]To start Weaviate:[/yellow]")
        console.print("cd weaviate/ && docker-compose up -d")
        return {"legal_documents": 0, "document_chunks": 0}
    
    return counts


def main():
    """Main function to run the document count check."""
    counts = check_document_counts()
    
    # Return appropriate exit code
    if counts["legal_documents"] == 0 and counts["document_chunks"] == 0:
        print("\nNo documents found in either collection.")
        sys.exit(1)
    else:
        print(f"\nFound {counts['legal_documents']} legal documents and {counts['document_chunks']} document chunks.")
        sys.exit(0)


if __name__ == "__main__":
    main()