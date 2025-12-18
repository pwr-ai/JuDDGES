#!/usr/bin/env python3
"""
Quick test script to verify Weaviate setup and connectivity.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import juddges modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import weaviate
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()

# Global variables - will be set from command line arguments or environment
WEAVIATE_URL = None
WEAVIATE_API_KEY = None


def test_connection():
    """Test basic connection to Weaviate."""
    console.print("\n[bold cyan]1. Testing Weaviate Connection[/bold cyan]")

    try:
        headers = {"Authorization": f"Bearer {WEAVIATE_API_KEY}"} if WEAVIATE_API_KEY else {}
        response = requests.get(f"{WEAVIATE_URL}/v1/meta", headers=headers, timeout=5)

        if response.status_code == 200:
            meta = response.json()
            console.print("[green]✓[/green] Connected successfully")
            console.print(f"  Version: {meta.get('version', 'unknown')}")
            console.print(f"  Hostname: {meta.get('hostname', 'unknown')}")
            return True
        else:
            console.print(f"[red]✗[/red] Connection failed: {response.status_code}")
            console.print(f"  Response: {response.text}")
            return False
    except Exception as e:
        console.print(f"[red]✗[/red] Connection error: {e}")
        return False


def test_schema():
    """Test schema and collections."""
    console.print("\n[bold cyan]2. Checking Schema and Collections[/bold cyan]")

    try:
        headers = {"Authorization": f"Bearer {WEAVIATE_API_KEY}"} if WEAVIATE_API_KEY else {}
        response = requests.get(f"{WEAVIATE_URL}/v1/schema", headers=headers, timeout=5)

        if response.status_code == 200:
            schema = response.json()
            classes = schema.get("classes", [])

            if not classes:
                console.print("[yellow]⚠[/yellow] No collections found")
                console.print("  Run ingestion script to create collections")
                return False

            console.print(f"[green]✓[/green] Found {len(classes)} collection(s)")

            table = Table(title="Collections")
            table.add_column("Class Name", style="cyan")
            table.add_column("Properties", style="green")
            table.add_column("Vectorizer", style="yellow")

            for cls in classes:
                name = cls.get("class", "unknown")
                prop_count = len(cls.get("properties", []))
                vectorizer = cls.get("vectorizer", "none")
                table.add_row(name, str(prop_count), vectorizer)

            console.print(table)
            return True
        else:
            console.print(f"[red]✗[/red] Schema request failed: {response.status_code}")
            return False
    except Exception as e:
        console.print(f"[red]✗[/red] Schema error: {e}")
        return False


def test_collection_counts():
    """Test collection counts using GraphQL."""
    console.print("\n[bold cyan]3. Checking Collection Counts[/bold cyan]")

    try:
        headers = {"Authorization": f"Bearer {WEAVIATE_API_KEY}"} if WEAVIATE_API_KEY else {}

        # Get schema first to find actual collection names
        response = requests.get(f"{WEAVIATE_URL}/v1/schema", headers=headers, timeout=5)
        if response.status_code != 200:
            console.print("[yellow]⚠[/yellow] Could not fetch schema")
            return False

        schema = response.json()
        classes = [cls.get("class") for cls in schema.get("classes", [])]

        if not classes:
            console.print("[yellow]⚠[/yellow] No collections to count")
            return False

        table = Table(title="Collection Statistics")
        table.add_column("Collection", style="cyan")
        table.add_column("Object Count", style="green")

        for class_name in classes:
            query = f"""
            {{
              Aggregate {{
                {class_name} {{
                  meta {{
                    count
                  }}
                }}
              }}
            }}
            """

            response = requests.post(
                f"{WEAVIATE_URL}/v1/graphql",
                json={"query": query},
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                count = (
                    data.get("data", {})
                    .get("Aggregate", {})
                    .get(class_name, [{}])[0]
                    .get("meta", {})
                    .get("count", 0)
                )
                table.add_row(class_name, str(count))
            else:
                table.add_row(class_name, "[red]error[/red]")

        console.print(table)
        console.print("[green]✓[/green] Count queries successful")
        return True

    except Exception as e:
        console.print(f"[red]✗[/red] Count error: {e}")
        return False


def test_transformers_service():
    """Test transformer service connectivity."""
    console.print("\n[bold cyan]4. Testing Transformer Service[/bold cyan]")

    try:
        # Try to reach the transformer service through Weaviate
        # This tests if the vectorization module is working
        headers = {"Authorization": f"Bearer {WEAVIATE_API_KEY}"} if WEAVIATE_API_KEY else {}
        response = requests.get(f"{WEAVIATE_URL}/v1/meta", headers=headers, timeout=5)

        if response.status_code == 200:
            meta = response.json()
            modules = meta.get("modules", {})

            if "text2vec-transformers" in modules:
                console.print("[green]✓[/green] text2vec-transformers module loaded")
                t2v_config = modules["text2vec-transformers"]
                model_info = t2v_config.get("model", {})
                model_name = model_info.get("_name_or_path", "unknown")
                hidden_size = model_info.get("hidden_size", "unknown")
                console.print(f"  Model path: {model_name}")
                console.print(f"  Hidden size: {hidden_size}")
                return True
            else:
                console.print("[yellow]⚠[/yellow] text2vec-transformers module not found")
                console.print(f"  Available modules: {list(modules.keys())}")
                return False
        else:
            console.print(f"[red]✗[/red] Could not check modules: {response.status_code}")
            return False

    except Exception as e:
        console.print(f"[red]✗[/red] Transformer test error: {e}")
        return False


def test_grpc_connection():
    """Test gRPC connection using Weaviate client."""
    console.print("\n[bold cyan]5. Testing gRPC Connection[/bold cyan]")

    try:
        # Parse URL to extract host and determine if HTTPS
        from urllib.parse import urlparse
        parsed_url = urlparse(WEAVIATE_URL)
        host = parsed_url.hostname or "localhost"
        is_https = parsed_url.scheme == "https"

        # Determine gRPC port
        # For DNS with HTTPS (no explicit port), try 443 first (nginx with gRPC), then 8085
        # For localhost:8084, use 8085
        if parsed_url.port:
            rest_port = parsed_url.port
            if rest_port == 8084:
                grpc_port = 8085
            else:
                grpc_port = 50051
        elif is_https:
            # HTTPS without explicit port (DNS address)
            rest_port = 443
            # Try 443 first (nginx with gRPC support), fallback to 8085
            grpc_port = 443
        else:
            rest_port = 80
            grpc_port = 50051

        console.print(f"  Connecting to gRPC: {host}:{grpc_port}")

        # Create Weaviate client with gRPC
        if WEAVIATE_API_KEY:
            auth_config = weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY)
        else:
            auth_config = None

        # Connect with explicit gRPC configuration
        client = weaviate.connect_to_custom(
            http_host=host,
            http_port=rest_port,
            http_secure=is_https,
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=is_https,
            auth_credentials=auth_config,
            skip_init_checks=False
        )

        # Test connection
        if client.is_ready():
            console.print("[green]✓[/green] gRPC connection successful")

            # Get metadata via gRPC
            meta = client.get_meta()
            console.print(f"  Version: {meta.get('version', 'unknown')}")

            # Test a simple query via gRPC
            collections = client.collections.list_all()
            console.print(f"  Collections accessible via gRPC: {len(collections)}")

            # Try a sample aggregate query to verify gRPC performance
            if collections:
                sample_collection = list(collections.keys())[0]
                collection = client.collections.get(sample_collection)
                result = collection.aggregate.over_all(total_count=True)
                console.print(f"  Sample query on '{sample_collection}': {result.total_count} objects")

            client.close()
            return True
        else:
            console.print("[red]✗[/red] gRPC connection not ready")
            client.close()
            return False

    except Exception as e:
        console.print(f"[red]✗[/red] gRPC connection error: {e}")
        console.print(f"  Error type: {type(e).__name__}")

        # Check if this is a DNS address issue
        if is_https and not parsed_url.port:
            console.print("\n  [yellow]Note:[/yellow] gRPC may not be accessible through the nginx proxy.")
            console.print("  gRPC typically works only via localhost or direct IP connections.")
            console.print("  For production use, consider exposing gRPC on a separate port/subdomain")
            console.print("  or use REST API for external connections.")

        return False


def test_docker_services():
    """Check if Docker services are running."""
    console.print("\n[bold cyan]6. Checking Docker Services[/bold cyan]")

    try:
        import subprocess

        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )

        if result.returncode == 0:
            import json
            services = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    services.append(json.loads(line))

            if not services:
                console.print("[yellow]⚠[/yellow] No services running")
                console.print("  Run: docker compose up -d")
                return False

            table = Table(title="Docker Services")
            table.add_column("Service", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Health", style="yellow")

            for svc in services:
                name = svc.get("Service", "unknown")
                state = svc.get("State", "unknown")
                health = svc.get("Health", "N/A")

                # Color code status
                if state == "running":
                    state_str = f"[green]{state}[/green]"
                else:
                    state_str = f"[red]{state}[/red]"

                table.add_row(name, state_str, health)

            console.print(table)
            console.print(f"[green]✓[/green] Found {len(services)} service(s)")
            return True
        else:
            console.print("[yellow]⚠[/yellow] Could not check Docker services")
            console.print(f"  Error: {result.stderr}")
            return False

    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Docker check skipped: {e}")
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test Weaviate connection and setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test DNS address (default)
  python test_weaviate_connection.py

  # Test localhost
  python test_weaviate_connection.py --url http://localhost:8084

  # Test with custom URL
  python test_weaviate_connection.py --url https://legal-ai-weaviate.augustyniak.ai
        """
    )
    parser.add_argument(
        "--url",
        default=os.getenv("WEAVIATE_URL", "https://legal-ai-weaviate.augustyniak.ai"),
        help="Weaviate URL (default: DNS from docker-compose.yaml or WEAVIATE_URL env var)"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("WEAVIATE_API_KEY", ""),
        help="Weaviate API key (default: WEAVIATE_API_KEY env var)"
    )
    return parser.parse_args()


def main():
    """Run all tests."""
    args = parse_args()

    # Set global variables from arguments
    global WEAVIATE_URL, WEAVIATE_API_KEY
    WEAVIATE_URL = args.url
    WEAVIATE_API_KEY = args.api_key

    console.print("[bold magenta]═══════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]  Weaviate Setup Test Suite[/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════[/bold magenta]")

    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  URL: {WEAVIATE_URL}")
    if WEAVIATE_API_KEY:
        console.print(f"  API Key: {'*' * (len(WEAVIATE_API_KEY) - 4) + WEAVIATE_API_KEY[-4:]}")
    else:
        console.print(f"  API Key: [yellow]Not provided[/yellow]")

    results = {
        "REST Connection": test_connection(),
        "Schema": test_schema(),
        "Collection Counts": test_collection_counts(),
        "Transformers": test_transformers_service(),
        "gRPC Connection": test_grpc_connection(),
        "Docker Services": test_docker_services(),
    }

    # Summary
    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  Test Summary[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "[green]✓ PASS[/green]" if result else "[red]✗ FAIL[/red]"
        console.print(f"  {status} - {test_name}")

    console.print(f"\n[bold]Results: {passed}/{total} tests passed[/bold]")

    if passed == total:
        console.print("\n[bold green]🎉 All tests passed! Weaviate is ready to use.[/bold green]")
    elif passed > 0:
        console.print("\n[bold yellow]⚠ Some tests failed. Check the details above.[/bold yellow]")
    else:
        console.print("\n[bold red]❌ All tests failed. Check your setup.[/bold red]")
        console.print("\n[bold]Troubleshooting steps:[/bold]")
        console.print("  1. Check if Docker services are running: docker compose ps")
        console.print("  2. Check logs: docker compose logs weaviate")
        console.print("  3. Restart services: docker compose restart")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
