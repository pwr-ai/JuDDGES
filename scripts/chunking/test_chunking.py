#!/usr/bin/env python3
"""
Test chunking with sdadas model and example judgment text.
Usage: python test_chunking.py [dataset_name] [--chunk-size SIZE] [--overlap SIZE] [--output-prefix PREFIX]
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from rich.console import Console

from juddges.preprocessing.text_chunker import TextChunker

console = Console()


def extract_tokenizer_from_model(model):
    """Extract tokenizer from SentenceTransformer model."""
    # Try different ways to access the tokenizer
    if hasattr(model, "tokenizer"):
        return model.tokenizer

    # Check if it's a list/sequence of modules
    if hasattr(model, "__getitem__") and len(model) > 0:
        first_module = model[0]
        if hasattr(first_module, "tokenizer"):
            return first_module.tokenizer

    # Check _modules attribute
    if hasattr(model, "_modules"):
        for module in model._modules.values():
            if hasattr(module, "tokenizer"):
                return module.tokenizer

    return None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test chunking with sdadas model and dataset examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_chunking.py                                    # Use default dataset
  python test_chunking.py AI-TAX/pl-eureka-raw-sample       # Specify dataset
  python test_chunking.py --chunk-size 256 --overlap 64     # Custom chunk settings
  python test_chunking.py my-dataset --output-prefix my_test # Custom output files
""",
    )

    parser.add_argument(
        "dataset_name",
        nargs="?",
        default="juddges/pl-court-raw-sample",
        help="HuggingFace dataset name (default: juddges/pl-court-raw-sample)",
    )

    parser.add_argument(
        "--chunk-size", type=int, default=512, help="Chunk size in tokens (default: 512)"
    )

    parser.add_argument(
        "--overlap", type=int, default=128, help="Chunk overlap in tokens (default: 128)"
    )

    parser.add_argument(
        "--min-chars", type=int, default=50, help="Minimum chunk size in characters (default: 50)"
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="chunking_test_results",
        help="Output file prefix (default: chunking_test_results)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="sdadas/mmlw-roberta-large",
        help="Embedding model name (default: sdadas/mmlw-roberta-large)",
    )

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    console.print("[bold blue]Configuration:[/bold blue]")
    console.print(f"  Dataset: {args.dataset_name}")
    console.print(f"  Model: {args.model}")
    console.print(f"  Chunk size: {args.chunk_size} tokens")
    console.print(f"  Overlap: {args.overlap} tokens")
    console.print(f"  Output prefix: {args.output_prefix}")
    console.print()
    # Use the model name for tokenizer loading in TextChunker
    console.print(f"[bold blue]Using {args.model} model for tokenizer...[/bold blue]")
    console.print(f"[green]✓ Model name configured: {args.model}[/green]")

    # Load example judgment text
    console.print("\n[bold blue]Loading example judgment data...[/bold blue]")
    try:
        # Try to load a sample dataset to get example text
        dataset = load_dataset(args.dataset_name, split="train", streaming=True)
        console.print(f"[green]✓ Dataset loaded successfully {args.dataset_name}![/green]")

        # Get first document
        first_doc = next(iter(dataset))
        judgment_text = first_doc.get("full_text", "")
        document_id = first_doc.get("document_id", "example_doc")

        if not judgment_text:
            judgment_text = first_doc.get("html_content", "")

        if not judgment_text:
            console.print("[red]No text content found in first document![/red]")
            return

        console.print(f"[green]✓ Loaded judgment text ({len(judgment_text)} characters)[/green]")
        console.print(f"[dim]Document ID: {document_id}[/dim]")

    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{args.dataset_name}': {e}")

    # Initialize TextChunker with tokenizer
    console.print(
        f"\n[bold blue]Initializing TextChunker with {args.chunk_size} token window...[/bold blue]"
    )
    chunker = TextChunker(
        id_col="document_id",
        text_col="text",
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        min_split_chars=args.min_chars,
        tokenizer=args.model,
    )

    # Apply chunking
    console.print("[bold blue]Applying chunking...[/bold blue]")
    chunk_data = chunker({"document_id": [document_id], "text": [judgment_text]})

    # Prepare results for review
    results = {
        "model_info": {
            "model_name": args.model,
            "tokenizer_class": tokenizer.__class__.__name__,
            "max_token_length": getattr(tokenizer, "model_max_length", 512),
            "vocab_size": getattr(tokenizer, "vocab_size", "unknown"),
        },
        "chunking_config": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.overlap,
            "min_split_chars": args.min_chars,
        },
        "dataset_info": {
            "dataset_name": args.dataset_name,
        },
        "original_text": {
            "document_id": document_id,
            "text_length": len(judgment_text),
            "text_preview": judgment_text[:500] + "..."
            if len(judgment_text) > 500
            else judgment_text,
        },
        "chunks": [],
    }

    # Process chunks
    console.print(f"[green]✓ Generated {len(chunk_data['chunk_text'])} chunks[/green]")

    for i, (chunk_id, chunk_text, chunk_len) in enumerate(
        zip(chunk_data["chunk_id"], chunk_data["chunk_text"], chunk_data["chunk_len"])
    ):
        # Count tokens in chunk
        tokens = tokenizer.encode(chunk_text, add_special_tokens=True)
        token_count = len(tokens)

        chunk_info = {
            "chunk_id": chunk_id,
            "position": i,
            "text": chunk_text,
            "char_length": chunk_len,
            "token_count": token_count,
            "tokens_preview": tokens[:20] if len(tokens) > 20 else tokens,
        }
        results["chunks"].append(chunk_info)

        console.print(f"[dim]Chunk {i}: {chunk_len} chars, {token_count} tokens[/dim]")

    # Save results to file
    output_file = Path(f"{args.output_prefix}.json")
    console.print(f"\n[bold blue]Saving results to {output_file}...[/bold blue]")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Also create a readable text version
    text_output = Path(f"{args.output_prefix}.txt")
    with open(text_output, "w", encoding="utf-8") as f:
        f.write("CHUNKING TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("MODEL INFO:\n")
        f.write(f"- Model: {results['model_info']['model_name']}\n")
        f.write(f"- Tokenizer: {results['model_info']['tokenizer_class']}\n")
        f.write(f"- Max tokens: {results['model_info']['max_token_length']}\n")
        f.write(f"- Vocab size: {results['model_info']['vocab_size']}\n\n")

        f.write("CHUNKING CONFIG:\n")
        f.write(f"- Chunk size: {results['chunking_config']['chunk_size']} tokens\n")
        f.write(f"- Overlap: {results['chunking_config']['chunk_overlap']} tokens\n")
        f.write(f"- Min chars: {results['chunking_config']['min_split_chars']}\n\n")

        f.write("ORIGINAL TEXT:\n")
        f.write(f"- Document ID: {results['original_text']['document_id']}\n")
        f.write(f"- Length: {results['original_text']['text_length']} characters\n\n")
        f.write("Text preview:\n")
        f.write(results["original_text"]["text_preview"])
        f.write("\n\n" + "=" * 50 + "\n\n")

        f.write("CHUNKS:\n\n")
        for chunk in results["chunks"]:
            f.write(f"CHUNK {chunk['position']} (ID: {chunk['chunk_id']})\n")
            f.write(f"Characters: {chunk['char_length']}, Tokens: {chunk['token_count']}\n")
            f.write("-" * 40 + "\n")
            f.write(chunk["text"])
            f.write("\n\n" + "=" * 40 + "\n\n")

    console.print("[bold green]✓ Results saved to:[/bold green]")
    console.print(f"  - [cyan]{output_file}[/cyan] (JSON format)")
    console.print(f"  - [cyan]{text_output}[/cyan] (Human readable)")

    # Print summary
    console.print("\n[bold yellow]SUMMARY:[/bold yellow]")
    console.print(f"- Original text: {len(judgment_text)} characters")
    console.print(f"- Generated chunks: {len(results['chunks'])}")
    console.print(
        f"- Average chunk size: {sum(c['char_length'] for c in results['chunks']) / len(results['chunks']):.1f} chars"
    )
    console.print(
        f"- Average tokens per chunk: {sum(c['token_count'] for c in results['chunks']) / len(results['chunks']):.1f}"
    )
    console.print(
        f"- Token range: {min(c['token_count'] for c in results['chunks'])}-{max(c['token_count'] for c in results['chunks'])} tokens"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        raise
