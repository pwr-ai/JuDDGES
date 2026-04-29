#!/usr/bin/env python3
"""
Check Weaviate database status and document counts with rich formatting.
"""

import os

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def check_weaviate_status():
    """Check Weaviate status and document counts."""

    console = Console()
    weaviate_url = "http://localhost:8084"
    api_key = os.getenv("WEAVIATE_API_KEY")
    if not api_key:
        raise RuntimeError("WEAVIATE_API_KEY env var is required")
    headers = {"Authorization": f"Bearer {api_key}"}

    # Header
    console.print(
        Panel.fit("🗄️ Weaviate Database Status Check", style="bold blue", border_style="bright_blue")
    )

    try:
        # Check connection
        response = requests.get(f"{weaviate_url}/v1/meta", headers=headers, timeout=5)
        if response.status_code != 200:
            console.print("❌ [red]Cannot connect to Weaviate[/red]")
            return

        meta_info = response.json()
        console.print(
            f"✅ [green]Connected to Weaviate v{meta_info.get('version', 'unknown')}[/green]"
        )

        # Get schema
        schema_response = requests.get(f"{weaviate_url}/v1/schema", headers=headers)
        if schema_response.status_code != 200:
            console.print("❌ [red]Cannot retrieve schema[/red]")
            return

        schema = schema_response.json()
        collections = [cls["class"] for cls in schema.get("classes", [])]

        if not collections:
            console.print("📭 [yellow]No collections found in Weaviate[/yellow]")
            return

        # Create collections table
        collections_table = Table(
            title="📊 Collection Statistics", show_header=True, header_style="bold magenta"
        )
        collections_table.add_column("Collection", style="cyan", width=25)
        collections_table.add_column("Documents", style="yellow", width=15)
        collections_table.add_column("Description", style="white", width=40)

        total_documents = 0

        # Count documents in each collection
        for collection in collections:
            graphql_query = f"""
            {{
              Aggregate {{
                {collection} {{
                  meta {{
                    count
                  }}
                }}
              }}
            }}
            """

            count_response = requests.post(
                f"{weaviate_url}/v1/graphql",
                json={"query": graphql_query},
                headers=headers,
                timeout=10,
            )

            if count_response.status_code == 200:
                data = count_response.json()
                count = (
                    data.get("data", {})
                    .get("Aggregate", {})
                    .get(collection, [{}])[0]
                    .get("meta", {})
                    .get("count", 0)
                )
                total_documents += count

                # Add emoji and description based on collection name
                if "Polish" in collection:
                    emoji = "🇵🇱"
                    description = "Polish court documents"
                elif "Chunk" in collection:
                    emoji = "📄"
                    description = "Text chunks for vector search"
                elif "Legal" in collection:
                    emoji = "⚖️"
                    description = "Legal documents"
                else:
                    emoji = "📋"
                    description = "General documents"

                collections_table.add_row(
                    f"{emoji} {collection}", f"[bold]{count:,}[/bold]", description
                )
            else:
                collections_table.add_row(
                    f"❌ {collection}", "[red]Error[/red]", "Could not retrieve count"
                )

        console.print(collections_table)

        # Summary
        summary_text = Text()
        summary_text.append("📊 Total Documents: ", style="bold cyan")
        summary_text.append(f"{total_documents:,}", style="bold yellow")
        summary_text.append("\n🏗️ Collections: ", style="bold cyan")
        summary_text.append(f"{len(collections)}", style="bold yellow")

        console.print(Panel(summary_text, title="📈 Database Summary", border_style="green"))

        # Show recent activity or sample data
        if total_documents > 0:
            console.print("\n🔍 [bold green]Sample Recent Data:[/bold green]")

            # Try to get a sample document from the first collection
            sample_collection = collections[0]
            sample_query = f"""
            {{
              Get {{
                {sample_collection}(limit: 1) {{
                  _additional {{
                    id
                    creationTimeUnix
                  }}
                }}
              }}
            }}
            """

            sample_response = requests.post(
                f"{weaviate_url}/v1/graphql",
                json={"query": sample_query},
                headers=headers,
                timeout=10,
            )

            if sample_response.status_code == 200:
                sample_data = sample_response.json()
                documents = sample_data.get("data", {}).get("Get", {}).get(sample_collection, [])
                if documents:
                    doc = documents[0]
                    console.print(
                        f"📄 Latest document ID: [cyan]{doc['_additional']['id'][:8]}...[/cyan]"
                    )

                    # Convert Unix timestamp to readable format
                    import datetime

                    timestamp = doc["_additional"]["creationTimeUnix"]
                    if isinstance(timestamp, str):
                        timestamp = int(timestamp)
                    created_time = datetime.datetime.fromtimestamp(timestamp / 1000)
                    console.print(
                        f"⏰ Created: [yellow]{created_time.strftime('%Y-%m-%d %H:%M:%S')}[/yellow]"
                    )

        # Usage suggestions
        console.print("\n💡 [bold green]Quick Actions:[/bold green]")
        actions_text = Text()
        actions_text.append("• Query documents: ", style="cyan")
        actions_text.append("python scripts/demos/demo_weaviate_ingestion.py\n", style="white")
        actions_text.append("• Add more data: ", style="cyan")
        actions_text.append("python scripts/demos/demo_local_dataset_ingestion.py\n", style="white")
        actions_text.append("• View available datasets: ", style="cyan")
        actions_text.append("python scripts/demos/show_available_datasets.py\n", style="white")

        console.print(Panel(actions_text, border_style="bright_cyan"))

    except requests.RequestException as e:
        console.print(f"❌ [red]Connection error:[/red] {e}")
        console.print("\n💡 [yellow]To start Weaviate:[/yellow]")
        console.print("cd weaviate/ && docker-compose up -d")
    except Exception as e:
        console.print(f"❌ [red]Unexpected error:[/red] {e}")


if __name__ == "__main__":
    check_weaviate_status()
