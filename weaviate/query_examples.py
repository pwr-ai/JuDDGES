#!/usr/bin/env python3
"""
Example queries for Weaviate legal documents database.
Demonstrates various query patterns for legal document search.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

# Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8084")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
HEADERS = {"Authorization": f"Bearer {WEAVIATE_API_KEY}"}


def run_graphql_query(query: str, description: str):
    """Execute a GraphQL query and display results."""
    console.print(f"\n[bold cyan]Query: {description}[/bold cyan]")
    console.print(Panel(Syntax(query, "graphql", theme="monokai", line_numbers=False)))

    try:
        response = requests.post(
            f"{WEAVIATE_URL}/v1/graphql",
            json={"query": query},
            headers=HEADERS,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()

            # Check for GraphQL errors
            if "errors" in data:
                console.print(f"[red]GraphQL Errors:[/red]")
                for error in data["errors"]:
                    console.print(f"  • {error.get('message', 'Unknown error')}")
                return None

            console.print("[green]✓[/green] Query successful")
            return data.get("data", {})
        else:
            console.print(f"[red]✗[/red] Query failed: {response.status_code}")
            console.print(f"  Response: {response.text}")
            return None

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        return None


def display_documents(data: dict, collection_name: str, limit: int = 3):
    """Display document results in a readable format."""
    documents = data.get("Get", {}).get(collection_name, [])

    if not documents:
        console.print("[yellow]No results found[/yellow]")
        return

    console.print(f"\n[bold green]Found {len(documents)} result(s):[/bold green]\n")

    for i, doc in enumerate(documents[:limit], 1):
        console.print(f"[bold]Result {i}:[/bold]")
        for key, value in doc.items():
            if value is not None:
                # Truncate long text fields
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                elif isinstance(value, list) and len(value) > 5:
                    value = value[:5] + ["..."]

                console.print(f"  [cyan]{key}:[/cyan] {value}")
        console.print()


def query_1_simple_document_fetch():
    """Fetch a few documents with basic fields."""
    query = """
    {
      Get {
        LegalDocuments(limit: 3) {
          document_id
          title
          country
          language
          date_issued
          document_type
        }
      }
    }
    """

    data = run_graphql_query(query, "Simple Document Fetch (first 3 documents)")
    if data:
        display_documents(data, "LegalDocuments")


def query_2_keyword_search():
    """Search documents by keyword."""
    query = """
    {
      Get {
        LegalDocuments(
          where: {
            path: ["full_text"]
            operator: Like
            valueText: "*umowa*"
          }
          limit: 3
        ) {
          document_id
          title
          country
          language
          date_issued
          full_text
        }
      }
    }
    """

    data = run_graphql_query(query, "Keyword Search - Documents containing 'umowa' (contract)")
    if data:
        display_documents(data, "LegalDocuments")


def query_3_filter_by_country():
    """Filter documents by country."""
    query = """
    {
      Get {
        LegalDocuments(
          where: {
            path: ["country"]
            operator: Equal
            valueText: "Poland"
          }
          limit: 5
        ) {
          document_id
          title
          country
          language
          court
          date_issued
        }
      }
    }
    """

    data = run_graphql_query(query, "Filter by Country - Polish documents")
    if data:
        display_documents(data, "LegalDocuments", limit=5)


def query_4_document_chunks():
    """Query document chunks."""
    query = """
    {
      Get {
        DocumentChunks(limit: 5) {
          document_id
          chunk_id
          chunk_text
          position
          language
          country
        }
      }
    }
    """

    data = run_graphql_query(query, "Document Chunks - First 5 chunks")
    if data:
        display_documents(data, "DocumentChunks", limit=5)


def query_5_date_range():
    """Query documents by date range."""
    query = """
    {
      Get {
        LegalDocuments(
          where: {
            operator: And
            operands: [
              {
                path: ["date_issued"]
                operator: GreaterThanEqual
                valueDate: "2023-01-01T00:00:00Z"
              }
              {
                path: ["date_issued"]
                operator: LessThanEqual
                valueDate: "2023-12-31T23:59:59Z"
              }
            ]
          }
          limit: 5
        ) {
          document_id
          title
          date_issued
          country
          court
        }
      }
    }
    """

    data = run_graphql_query(query, "Date Range Filter - Documents from 2023")
    if data:
        display_documents(data, "LegalDocuments", limit=5)


def query_6_aggregate_stats():
    """Get aggregate statistics."""
    query = """
    {
      Aggregate {
        LegalDocuments {
          meta {
            count
          }
          country {
            count
            topOccurrences(limit: 5) {
              value
              occurs
            }
          }
          language {
            count
            topOccurrences(limit: 5) {
              value
              occurs
            }
          }
        }
      }
    }
    """

    data = run_graphql_query(query, "Aggregate Statistics - Count and top values")
    if data:
        agg = data.get("Aggregate", {}).get("LegalDocuments", [])
        if agg:
            console.print("\n[bold green]Statistics:[/bold green]\n")

            # Total count
            meta = agg[0].get("meta", {})
            console.print(f"  [cyan]Total Documents:[/cyan] {meta.get('count', 0):,}")

            # Country distribution
            country_data = agg[0].get("country", {})
            if country_data:
                console.print(f"\n  [cyan]Top Countries:[/cyan]")
                for item in country_data.get("topOccurrences", []):
                    console.print(f"    • {item['value']}: {item['occurs']:,} documents")

            # Language distribution
            language_data = agg[0].get("language", {})
            if language_data:
                console.print(f"\n  [cyan]Top Languages:[/cyan]")
                for item in language_data.get("topOccurrences", []):
                    console.print(f"    • {item['value']}: {item['occurs']:,} documents")

            console.print()


def query_7_near_text_search():
    """Perform semantic search using nearText."""
    query = """
    {
      Get {
        LegalDocuments(
          nearText: {
            concepts: ["odszkodowanie za szkodę", "odpowiedzialność cywilna"]
          }
          limit: 3
        ) {
          document_id
          title
          country
          language
          full_text
          _additional {
            distance
          }
        }
      }
    }
    """

    data = run_graphql_query(query, "Semantic Search - nearText for 'compensation for damage'")
    if data:
        console.print("\n[yellow]Note:[/yellow] nearText requires vectorizer configuration.")
        console.print("If results are empty, the collection may not have vectors configured.\n")
        display_documents(data, "LegalDocuments")


def query_8_bm25_search():
    """Perform BM25 keyword search."""
    query = """
    {
      Get {
        LegalDocuments(
          bm25: {
            query: "wyrok sądu najwyższego"
          }
          limit: 5
        ) {
          document_id
          title
          court
          date_issued
          full_text
          _additional {
            score
          }
        }
      }
    }
    """

    data = run_graphql_query(query, "BM25 Search - 'Supreme Court judgment'")
    if data:
        documents = data.get("Get", {}).get("LegalDocuments", [])
        if documents:
            console.print(f"\n[bold green]Found {len(documents)} result(s):[/bold green]\n")

            for i, doc in enumerate(documents, 1):
                console.print(f"[bold]Result {i}:[/bold]")
                console.print(f"  [cyan]document_id:[/cyan] {doc.get('document_id')}")
                console.print(f"  [cyan]title:[/cyan] {doc.get('title', 'N/A')}")
                console.print(f"  [cyan]court:[/cyan] {doc.get('court', 'N/A')}")
                console.print(f"  [cyan]date_issued:[/cyan] {doc.get('date_issued', 'N/A')}")

                # Show BM25 score
                additional = doc.get("_additional", {})
                if "score" in additional:
                    console.print(f"  [cyan]BM25 score:[/cyan] {additional['score']:.4f}")

                # Show text preview
                full_text = doc.get("full_text", "")
                if full_text:
                    preview = full_text[:200] + "..." if len(full_text) > 200 else full_text
                    console.print(f"  [cyan]text preview:[/cyan] {preview}")

                console.print()
        else:
            console.print("[yellow]No results found[/yellow]")


def main():
    """Run example queries."""
    console.print("[bold magenta]═══════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]  Weaviate Example Queries[/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════[/bold magenta]")

    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  URL: {WEAVIATE_URL}")
    console.print(f"  API Key: {'*' * (len(WEAVIATE_API_KEY) - 4) + WEAVIATE_API_KEY[-4:]}")

    queries = [
        ("1", "Simple Document Fetch", query_1_simple_document_fetch),
        ("2", "Keyword Search", query_2_keyword_search),
        ("3", "Filter by Country", query_3_filter_by_country),
        ("4", "Document Chunks", query_4_document_chunks),
        ("5", "Date Range Filter", query_5_date_range),
        ("6", "Aggregate Statistics", query_6_aggregate_stats),
        ("7", "Semantic Search (nearText)", query_7_near_text_search),
        ("8", "BM25 Keyword Search", query_8_bm25_search),
    ]

    console.print("\n[bold]Available queries:[/bold]")
    for num, name, _ in queries:
        console.print(f"  {num}. {name}")

    console.print("\n[bold cyan]Running all queries...[/bold cyan]")
    console.print("=" * 60)

    for num, name, query_func in queries:
        console.print(f"\n[bold yellow]{'=' * 60}[/bold yellow]")
        console.print(f"[bold yellow]Query {num}: {name}[/bold yellow]")
        console.print(f"[bold yellow]{'=' * 60}[/bold yellow]")

        try:
            query_func()
        except Exception as e:
            console.print(f"[red]Error running query: {e}[/red]")
            logger.exception(f"Query {num} failed")

    console.print(f"\n[bold yellow]{'=' * 60}[/bold yellow]")
    console.print("[bold green]✓ All queries completed![/bold green]")


if __name__ == "__main__":
    main()
