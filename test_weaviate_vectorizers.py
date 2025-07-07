#!/usr/bin/env python3
"""
Test script to verify Weaviate and multiple vectorizer services are working.
"""

import os

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def test_transformer_services():
    """Test that all transformer services are responding."""
    services = {
        "base": "http://localhost:8080",
        "dev": "http://localhost:8081",
        "fast": "http://localhost:8082",
    }

    console.print("\n[bold cyan]🔧 Testing Transformer Services[/bold cyan]")

    results = {}
    for name, url in services.items():
        try:
            # Test the readiness endpoint
            ready_url = f"{url}/.well-known/ready"
            response = requests.get(ready_url, timeout=5)

            if response.status_code == 200:
                console.print(f"✅ {name} service ({url}): [green]Ready[/green]")
                results[name] = "ready"
            else:
                console.print(
                    f"⚠️  {name} service ({url}): [yellow]Not ready (status: {response.status_code})[/yellow]"
                )
                results[name] = "not_ready"

        except requests.exceptions.ConnectionError:
            console.print(f"❌ {name} service ({url}): [red]Connection refused[/red]")
            results[name] = "connection_error"
        except requests.exceptions.Timeout:
            console.print(f"⏱️  {name} service ({url}): [yellow]Timeout[/yellow]")
            results[name] = "timeout"
        except Exception as e:
            console.print(f"❌ {name} service ({url}): [red]Error: {e}[/red]")
            results[name] = "error"

    return results


def test_weaviate_connection():
    """Test Weaviate connection."""
    console.print("\n[bold cyan]🌐 Testing Weaviate Connection[/bold cyan]")

    try:
        import weaviate
        import weaviate.auth as wv_auth

        # Get API key
        api_key = os.getenv("WEAVIATE_API_KEY") or os.getenv("WV_API_KEY")

        if api_key:
            client = weaviate.connect_to_local(
                host="localhost", port=8084, auth_credentials=wv_auth.AuthApiKey(api_key)
            )
        else:
            client = weaviate.connect_to_local(host="localhost", port=8084)

        # Test connection
        meta = client.get_meta()
        console.print(
            f"✅ Weaviate: [green]Connected[/green] (version: {meta.get('version', 'unknown')})"
        )

        # List collections
        collections = client.collections.list_all()
        if isinstance(collections, dict):
            collection_names = list(collections.keys())
        else:
            collection_names = (
                [c.name for c in collections] if hasattr(collections, "__iter__") else []
            )

        console.print(f"📊 Collections: {len(collection_names)} found")
        for name in collection_names:
            console.print(f"  - {name}")

        client.close()
        return True, collection_names

    except Exception as e:
        console.print(f"❌ Weaviate: [red]Failed to connect: {e}[/red]")
        return False, []


def test_vectorizer_endpoints():
    """Test vectorizer endpoints with sample text."""
    console.print("\n[bold cyan]🎯 Testing Vectorizer Endpoints[/bold cyan]")

    sample_text = "This is a test legal document about court proceedings."
    services = {
        "base": "http://localhost:8080",
        "dev": "http://localhost:8081",
        "fast": "http://localhost:8082",
    }

    results = {}
    for name, url in services.items():
        try:
            # Test vectorization endpoint
            vector_url = f"{url}/vectors"
            payload = {"text": sample_text}

            response = requests.post(vector_url, json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if "vector" in data and isinstance(data["vector"], list):
                    vector_length = len(data["vector"])
                    console.print(
                        f"✅ {name} vectorizer: [green]Working[/green] (dim: {vector_length})"
                    )
                    results[name] = {"status": "working", "dimensions": vector_length}
                else:
                    console.print(
                        f"⚠️  {name} vectorizer: [yellow]Unexpected response format[/yellow]"
                    )
                    results[name] = {"status": "unexpected_format", "dimensions": 0}
            else:
                console.print(f"❌ {name} vectorizer: [red]HTTP {response.status_code}[/red]")
                results[name] = {"status": "http_error", "dimensions": 0}

        except requests.exceptions.ConnectionError:
            console.print(f"❌ {name} vectorizer: [red]Connection refused[/red]")
            results[name] = {"status": "connection_error", "dimensions": 0}
        except requests.exceptions.Timeout:
            console.print(f"⏱️  {name} vectorizer: [yellow]Timeout (>30s)[/yellow]")
            results[name] = {"status": "timeout", "dimensions": 0}
        except Exception as e:
            console.print(f"❌ {name} vectorizer: [red]Error: {e}[/red]")
            results[name] = {"status": "error", "dimensions": 0}

    return results


def test_collection_creation():
    """Test creating a collection with multiple named vectors."""
    console.print("\n[bold cyan]🏗️  Testing Collection Creation with Named Vectors[/bold cyan]")

    try:
        import weaviate
        import weaviate.auth as wv_auth
        import weaviate.classes.config as wvc

        # Get API key
        api_key = os.getenv("WEAVIATE_API_KEY") or os.getenv("WV_API_KEY")

        if api_key:
            client = weaviate.connect_to_local(
                host="localhost", port=8084, auth_credentials=wv_auth.AuthApiKey(api_key)
            )
        else:
            client = weaviate.connect_to_local(host="localhost", port=8084)

        # Test collection name
        test_collection = "TestMultiVector"

        # Delete if exists
        try:
            client.collections.delete(test_collection)
            console.print(f"🗑️  Deleted existing {test_collection} collection")
        except:
            pass

        # Create test collection
        client.collections.create(
            name=test_collection,
            properties=[
                wvc.Property(name="text", data_type=wvc.DataType.TEXT),
                wvc.Property(name="title", data_type=wvc.DataType.TEXT),
            ],
            vectorizer_config=[
                wvc.Configure.NamedVectors.text2vec_transformers(
                    name="base",
                    source_properties=["text"],
                    inference_url="http://t2v-transformers-base:8080",
                ),
                wvc.Configure.NamedVectors.text2vec_transformers(
                    name="dev",
                    source_properties=["text"],
                    inference_url="http://t2v-transformers-dev:8080",
                ),
                wvc.Configure.NamedVectors.text2vec_transformers(
                    name="fast",
                    source_properties=["text"],
                    inference_url="http://t2v-transformers-fast:8080",
                ),
            ],
        )

        console.print(f"✅ Created test collection: [green]{test_collection}[/green]")

        # Test adding a document
        collection = client.collections.get(test_collection)

        # Insert test document
        uuid = collection.data.insert(
            properties={
                "text": "This is a test legal document for vectorization testing.",
                "title": "Test Document",
            }
        )

        console.print(f"✅ Inserted test document: [green]{uuid}[/green]")

        # Query the document to verify vectors were created
        response = collection.query.get(include_vector=["base", "dev", "fast"], limit=1)

        if response.objects:
            obj = response.objects[0]
            vectors_info = {}

            if hasattr(obj, "vector") and obj.vector:
                for vector_name, vector_data in obj.vector.items():
                    if vector_data:
                        vectors_info[vector_name] = len(vector_data)

            console.print(f"✅ Vectors generated: [green]{list(vectors_info.keys())}[/green]")
            for name, dim in vectors_info.items():
                console.print(f"  - {name}: {dim} dimensions")

        # Clean up
        client.collections.delete(test_collection)
        console.print("🗑️  Cleaned up test collection")

        client.close()
        return True

    except Exception as e:
        console.print(f"❌ Collection test failed: [red]{e}[/red]")
        return False


def create_status_table(transformer_results, vectorizer_results):
    """Create a status summary table."""
    table = Table(title="🔍 Service Status Summary", show_header=True, header_style="bold magenta")
    table.add_column("Service", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")

    # Transformer services
    for name, status in transformer_results.items():
        status_color = "green" if status == "ready" else "red"
        table.add_row(
            f"t2v-transformers-{name}",
            "Transformer",
            f"[{status_color}]{status}[/{status_color}]",
            "Service readiness",
        )

    # Vectorizer endpoints
    for name, info in vectorizer_results.items():
        status = info["status"]
        dims = info["dimensions"]
        status_color = "green" if status == "working" else "red"
        details = f"{dims} dimensions" if dims > 0 else "No vector generated"

        table.add_row(
            f"{name}-vectorizer",
            "Vectorizer",
            f"[{status_color}]{status}[/{status_color}]",
            details,
        )

    return table


def main():
    """Main test function."""
    console.print(
        Panel.fit(
            "[bold blue]🧪 Weaviate Multi-Vectorizer Test Suite[/bold blue]\n\n"
            "This script tests:\n"
            "• Transformer service connectivity\n"
            "• Weaviate connection\n"
            "• Vectorizer endpoint functionality\n"
            "• Multi-vector collection creation",
            title="Test Suite",
            border_style="blue",
        )
    )

    # Test transformer services
    transformer_results = test_transformer_services()

    # Test Weaviate connection
    weaviate_connected, collections = test_weaviate_connection()

    # Test vectorizer endpoints
    vectorizer_results = test_vectorizer_endpoints()

    # Test collection creation (only if Weaviate is connected)
    collection_test_passed = False
    if weaviate_connected:
        collection_test_passed = test_collection_creation()

    # Create summary table
    console.print("\n")
    console.print(create_status_table(transformer_results, vectorizer_results))

    # Final summary
    all_transformers_ready = all(status == "ready" for status in transformer_results.values())
    all_vectorizers_working = all(
        info["status"] == "working" for info in vectorizer_results.values()
    )

    if (
        all_transformers_ready
        and weaviate_connected
        and all_vectorizers_working
        and collection_test_passed
    ):
        console.print(
            Panel.fit(
                "[bold green]🎉 All tests passed![/bold green]\n\n"
                "Your multi-vectorizer setup is working correctly:\n"
                "✅ All transformer services are ready\n"
                "✅ Weaviate is connected and responsive\n"
                "✅ All vectorizer endpoints are working\n"
                "✅ Multi-vector collection creation successful\n\n"
                "You can now run your ingestion script!",
                title="Success",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel.fit(
                "[bold red]❌ Some tests failed[/bold red]\n\n"
                "Issues detected:\n"
                f"• Transformers ready: {all_transformers_ready}\n"
                f"• Weaviate connected: {weaviate_connected}\n"
                f"• Vectorizers working: {all_vectorizers_working}\n"
                f"• Collection test passed: {collection_test_passed}\n\n"
                "Please check the logs above for specific error details.",
                title="Issues Found",
                border_style="red",
            )
        )


if __name__ == "__main__":
    main()
