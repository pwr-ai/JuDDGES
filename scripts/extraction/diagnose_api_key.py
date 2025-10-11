"""Diagnose Google API key and check available models."""

import json
import os
import sys

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def main():
    console.print(
        Panel.fit(
            "[bold cyan]Google API Key Diagnostic[/bold cyan]\n\n"
            "Testing your API key and checking available models",
            border_style="cyan",
        )
    )

    # Load from .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        console.print("[red]✗ GOOGLE_API_KEY not set[/red]")
        sys.exit(1)

    console.print(f"\n[bold]API Key:[/bold] {api_key[:20]}...")

    # Test 1: List models
    console.print("\n[bold cyan]Test 1: Checking Available Models[/bold cyan]")

    try:
        response = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])

            console.print(f"[green]✓ Found {len(models)} models[/green]\n")

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Model Name", style="cyan")
            table.add_column("Display Name", style="white")
            table.add_column("Supports Generate", style="green")

            for model in models[:15]:  # Show first 15
                name = model.get("name", "").replace("models/", "")
                display = model.get("displayName", "N/A")
                methods = model.get("supportedGenerationMethods", [])
                supports_generate = "✓" if "generateContent" in methods else "✗"

                table.add_row(name, display, supports_generate)

            console.print(table)

            # Check for gemini-2.5-flash
            gemini_flash_found = any("gemini-2" in m.get("name", "").lower() or "gemini-1.5" in m.get("name", "").lower() for m in models)

            if gemini_flash_found:
                console.print("\n[green]✓ Gemini models available![/green]")
            else:
                console.print("\n[yellow]⚠ Gemini 2.5 might not be available with this key[/yellow]")

        elif response.status_code == 403:
            console.print(f"[red]✗ 403 Permission Denied[/red]")
            console.print("\n[yellow]Your API key doesn't have access to Gemini API[/yellow]")
            console.print("\n🔧 [bold]Solution:[/bold]")
            console.print("  1. Go to: https://aistudio.google.com/apikey")
            console.print("  2. Create a new API key")
            console.print("  3. Update your .env file")
            sys.exit(1)

        elif response.status_code == 400:
            console.print(f"[red]✗ 400 Invalid API Key[/red]")
            console.print("\n[yellow]The API key format is invalid[/yellow]")
            sys.exit(1)

        else:
            console.print(f"[red]✗ Error {response.status_code}[/red]")
            console.print(response.text)
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]✗ Request failed: {e}[/red]")
        sys.exit(1)

    # Test 2: Try a simple generation
    console.print("\n[bold cyan]Test 2: Testing Generation[/bold cyan]")

    # Find a working model
    working_model = None
    for model in models:
        if "generateContent" in model.get("supportedGenerationMethods", []):
            working_model = model.get("name")
            break

    if not working_model:
        console.print("[red]✗ No models support generateContent[/red]")
        sys.exit(1)

    console.print(f"[yellow]Using model: {working_model}[/yellow]")

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/{working_model}:generateContent?key={api_key}",
            json={
                "contents": [{
                    "parts": [{
                        "text": "Say hello in one word"
                    }]
                }]
            },
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            console.print(f"\n[green]✓ Generation successful![/green]")
            console.print(f"[dim]Response: {text}[/dim]")

            # Success message
            console.print("\n" + "=" * 60)
            console.print("[bold green]✅ YOUR API KEY WORKS![/bold green]")
            console.print("=" * 60)

            console.print("\n[bold]Now you can run:[/bold]")
            console.print("  python scripts/extraction/test_langfuse_simple.py")
            console.print("  python scripts/extraction/run_10_examples.py")

        elif response.status_code == 403:
            console.print(f"\n[red]✗ 403 Permission Denied during generation[/red]")
            error_data = response.json()
            console.print(f"[dim]{json.dumps(error_data, indent=2)}[/dim]")

            console.print("\n[yellow]This key can list models but can't generate content[/yellow]")
            console.print("\n🔧 [bold]Get a proper Gemini API key:[/bold]")
            console.print("  1. https://aistudio.google.com/apikey")
            console.print("  2. Click 'Create API Key'")
            console.print("  3. Update .env with new key")

        else:
            console.print(f"\n[red]✗ Generation failed: {response.status_code}[/red]")
            console.print(response.text)

    except Exception as e:
        console.print(f"[red]✗ Generation test failed: {e}[/red]")


if __name__ == "__main__":
    main()
