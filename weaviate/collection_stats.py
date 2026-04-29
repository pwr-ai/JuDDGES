#!/usr/bin/env python3
"""
Quick statistics script - shows how we got the collection counts.
"""

import os
import requests
from rich.console import Console
from rich.table import Table

console = Console()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8084")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
HEADERS = {"Authorization": f"Bearer {WEAVIATE_API_KEY}"}


def get_collection_count(collection_name: str) -> int:
    """Get count for a specific collection."""
    query = f"""
    {{
      Aggregate {{
        {collection_name} {{
          meta {{
            count
          }}
        }}
      }}
    }}
    """

    try:
        response = requests.post(
            f"{WEAVIATE_URL}/v1/graphql",
            json={"query": query},
            headers=HEADERS,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            count = (
                data.get("data", {})
                .get("Aggregate", {})
                .get(collection_name, [{}])[0]
                .get("meta", {})
                .get("count", 0)
            )
            return count
        else:
            console.print(f"[red]Error:[/red] {response.status_code}")
            return 0
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 0


def get_schema_info(collection_name: str) -> int:
    """Get number of properties in schema."""
    try:
        response = requests.get(
            f"{WEAVIATE_URL}/v1/schema",
            headers=HEADERS,
            timeout=5
        )

        if response.status_code == 200:
            schema = response.json()
            for cls in schema.get("classes", []):
                if cls.get("class") == collection_name:
                    return len(cls.get("properties", []))
        return 0
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 0


def main():
    console.print("\n[bold cyan]Weaviate Collection Statistics[/bold cyan]")
    console.print("=" * 50)
    console.print("\nThis shows how we got the collection counts.\n")

    # Get statistics for both collections
    collections = ["LegalDocuments", "DocumentChunks"]

    table = Table(title="Collection Statistics")
    table.add_column("Collection", style="cyan", no_wrap=True)
    table.add_column("Object Count", style="green", justify="right")
    table.add_column("Properties", style="yellow", justify="right")

    for collection in collections:
        console.print(f"[dim]Querying {collection}...[/dim]")

        count = get_collection_count(collection)
        properties = get_schema_info(collection)

        table.add_row(
            collection,
            f"{count:,}",
            str(properties)
        )

    console.print()
    console.print(table)

    # Show the exact query used
    console.print("\n[bold]Exact GraphQL Query Used:[/bold]")
    console.print("""
{
  Aggregate {
    LegalDocuments {
      meta {
        count
      }
    }
  }
}
    """)

    console.print("\n[bold]Or via curl:[/bold]")
    console.print("""
curl -H "Authorization: Bearer $WEAVIATE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "{ Aggregate { LegalDocuments { meta { count } } } }"}' \\
  http://localhost:8084/v1/graphql
    """)


if __name__ == "__main__":
    main()
