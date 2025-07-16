#!/usr/bin/env python3
"""
Analyze chunking statistics across 100 examples from the dataset.
Usage: python analyze_dataset_chunking.py [dataset_name] [--num-examples N] [--output-prefix PREFIX]
"""

import argparse
import json
import statistics
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from juddges.preprocessing.text_chunker import TextChunker
from rich.console import Console
from rich.progress import track
from rich.table import Table
import numpy as np

console = Console()

def extract_tokenizer_from_model(model):
    """Extract tokenizer from SentenceTransformer model."""
    # Try different ways to access the tokenizer
    if hasattr(model, 'tokenizer'):
        return model.tokenizer
    
    # Check if it's a list/sequence of modules
    if hasattr(model, '__getitem__') and len(model) > 0:
        first_module = model[0]
        if hasattr(first_module, 'tokenizer'):
            return first_module.tokenizer
    
    # Check _modules attribute
    if hasattr(model, '_modules'):
        for module in model._modules.values():
            if hasattr(module, 'tokenizer'):
                return module.tokenizer
    
    return None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze chunking statistics across dataset examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_dataset_chunking.py                                    # Use default dataset
  python analyze_dataset_chunking.py AI-TAX/pl-eureka-raw-sample       # Specify dataset
  python analyze_dataset_chunking.py --num-examples 50                 # Process 50 examples
  python analyze_dataset_chunking.py juddges/pl-court-raw-sample --output-prefix court_analysis
"""
    )
    
    parser.add_argument(
        "dataset_name",
        nargs="?",
        default="juddges/pl-court-raw-sample",
        help="HuggingFace dataset name (default: juddges/pl-court-raw-sample)"
    )
    
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of examples to process (default: 100)"
    )
    
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="dataset_chunking_analysis",
        help="Output file prefix (default: dataset_chunking_analysis)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in tokens (default: 512)"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Chunk overlap in tokens (default: 128)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="sdadas/mmlw-roberta-large",
        help="Embedding model name (default: sdadas/mmlw-roberta-large)"
    )
    
    return parser.parse_args()

def analyze_chunking_statistics(dataset_name, num_examples=100, output_prefix="dataset_chunking_analysis", chunk_size=512, overlap=128, model_name="sdadas/mmlw-roberta-large"):
    """Analyze chunking statistics across dataset examples."""
    
    console.print(f"[bold blue]Configuration:[/bold blue]")
    console.print(f"  Dataset: {dataset_name}")
    console.print(f"  Model: {model_name}")
    console.print(f"  Examples: {num_examples}")
    console.print(f"  Chunk size: {chunk_size} tokens")
    console.print(f"  Overlap: {overlap} tokens")
    console.print(f"  Output prefix: {output_prefix}")
    console.print()
    
    # Use the model name for tokenizer loading in TextChunker
    console.print(f"[bold blue]Using {model_name} model for tokenizer...[/bold blue]")
    console.print(f"[green]✓ Model name configured: {model_name}[/green]")
    
    # Load dataset
    console.print(f"\n[bold blue]Loading dataset (first {num_examples} examples)...[/bold blue]")
    try:
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        console.print(f"[green]✓ Dataset loaded: {dataset_name}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        return
    
    # Initialize TextChunker with tokenizer
    console.print(f"\n[bold blue]Initializing TextChunker with {chunk_size} token window...[/bold blue]")
    chunker = TextChunker(
        id_col="document_id",
        text_col="text",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        min_split_chars=50,
        tokenizer=model_name
    )
    
    # Statistics collection
    all_stats = []
    document_stats = []
    chunk_lengths = []
    chunk_token_counts = []
    chunks_per_document = []
    
    console.print(f"\n[bold blue]Processing {num_examples} documents...[/bold blue]")
    
    processed_count = 0
    for doc in track(dataset, total=num_examples, description="Processing documents"):
        if processed_count >= num_examples:
            break
            
        # Extract document info
        doc_id = doc.get("document_id", f"doc_{processed_count}")
        judgment_text = doc.get("full_text", "") or doc.get("html_content", "")
        
        if not judgment_text or len(judgment_text) < 100:  # Skip very short documents
            continue
        
        try:
            # Apply chunking
            chunk_data = chunker({
                "document_id": [doc_id],
                "text": [judgment_text]
            })
            
            # Calculate statistics for this document
            doc_chunk_count = len(chunk_data["chunk_text"])
            doc_char_length = len(judgment_text)
            
            # Calculate token counts for each chunk
            doc_chunk_tokens = []
            doc_chunk_chars = []
            
            for chunk_text in chunk_data["chunk_text"]:
                tokens = tokenizer.encode(chunk_text, add_special_tokens=True)
                token_count = len(tokens)
                char_count = len(chunk_text)
                
                doc_chunk_tokens.append(token_count)
                doc_chunk_chars.append(char_count)
                chunk_token_counts.append(token_count)
                chunk_lengths.append(char_count)
            
            chunks_per_document.append(doc_chunk_count)
            
            # Store document-level statistics
            doc_stats = {
                "document_id": doc_id,
                "original_length": doc_char_length,
                "num_chunks": doc_chunk_count,
                "chunk_token_counts": doc_chunk_tokens,
                "chunk_char_lengths": doc_chunk_chars,
                "avg_chunk_tokens": statistics.mean(doc_chunk_tokens),
                "avg_chunk_chars": statistics.mean(doc_chunk_chars),
                "min_chunk_tokens": min(doc_chunk_tokens),
                "max_chunk_tokens": max(doc_chunk_tokens),
                "min_chunk_chars": min(doc_chunk_chars),
                "max_chunk_chars": max(doc_chunk_chars)
            }
            
            document_stats.append(doc_stats)
            all_stats.append(doc_stats)
            
            processed_count += 1
            
        except Exception as e:
            console.print(f"[yellow]Error processing document {doc_id}: {e}[/yellow]")
            continue
    
    console.print(f"[green]✓ Successfully processed {len(all_stats)} documents[/green]")
    
    # Calculate aggregated statistics
    console.print("\n[bold blue]Calculating aggregated statistics...[/bold blue]")
    
    if not all_stats:
        console.print("[red]No documents were successfully processed![/red]")
        return
    
    # Overall statistics
    total_chunks = sum(stat["num_chunks"] for stat in all_stats)
    total_documents = len(all_stats)
    
    # Document-level aggregations
    doc_lengths = [stat["original_length"] for stat in all_stats]
    chunks_per_doc = [stat["num_chunks"] for stat in all_stats]
    
    # Chunk-level aggregations
    all_chunk_tokens = chunk_token_counts
    all_chunk_chars = chunk_lengths
    
    aggregated_stats = {
        "dataset_info": {
            "dataset_name": dataset_name,
            "processed_documents": total_documents,
            "total_chunks": total_chunks
        },
        "model_info": {
            "model_name": model_name,
            "tokenizer_class": tokenizer.__class__.__name__,
            "max_token_length": getattr(tokenizer, 'model_max_length', 512),
            "vocab_size": getattr(tokenizer, 'vocab_size', 'unknown')
        },
        "chunking_config": {
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "min_split_chars": 50
        },
        "document_statistics": {
            "count": total_documents,
            "original_length": {
                "mean": statistics.mean(doc_lengths),
                "median": statistics.median(doc_lengths),
                "min": min(doc_lengths),
                "max": max(doc_lengths),
                "std": statistics.stdev(doc_lengths) if len(doc_lengths) > 1 else 0
            },
            "chunks_per_document": {
                "mean": statistics.mean(chunks_per_doc),
                "median": statistics.median(chunks_per_doc),
                "min": min(chunks_per_doc),
                "max": max(chunks_per_doc),
                "std": statistics.stdev(chunks_per_doc) if len(chunks_per_doc) > 1 else 0
            }
        },
        "chunk_statistics": {
            "total_count": total_chunks,
            "character_length": {
                "mean": statistics.mean(all_chunk_chars),
                "median": statistics.median(all_chunk_chars),
                "min": min(all_chunk_chars),
                "max": max(all_chunk_chars),
                "std": statistics.stdev(all_chunk_chars) if len(all_chunk_chars) > 1 else 0
            },
            "token_count": {
                "mean": statistics.mean(all_chunk_tokens),
                "median": statistics.median(all_chunk_tokens),
                "min": min(all_chunk_tokens),
                "max": max(all_chunk_tokens),
                "std": statistics.stdev(all_chunk_tokens) if len(all_chunk_tokens) > 1 else 0
            }
        },
        "detailed_documents": all_stats[:10]  # Store first 10 for detailed review
    }
    
    # Save detailed results
    output_file = Path(f"{output_prefix}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(aggregated_stats, f, indent=2, ensure_ascii=False)
    
    # Display results in console
    display_results(aggregated_stats)
    
    console.print(f"\n[bold green]✓ Detailed results saved to {output_file}[/bold green]")
    
    return aggregated_stats

def display_results(stats):
    """Display results in a formatted table."""
    console.print("\n[bold yellow]DATASET CHUNKING ANALYSIS RESULTS[/bold yellow]")
    console.print("=" * 60)
    
    # Dataset overview
    console.print(f"\n[bold cyan]Dataset Overview:[/bold cyan]")
    console.print(f"Dataset: {stats['dataset_info']['dataset_name']}")
    console.print(f"Processed documents: {stats['dataset_info']['processed_documents']:,}")
    console.print(f"Total chunks generated: {stats['dataset_info']['total_chunks']:,}")
    console.print(f"Average chunks per document: {stats['dataset_info']['total_chunks'] / stats['dataset_info']['processed_documents']:.1f}")
    
    # Model info
    console.print(f"\n[bold cyan]Model Configuration:[/bold cyan]")
    console.print(f"Model: {stats['model_info']['model_name']}")
    console.print(f"Tokenizer: {stats['model_info']['tokenizer_class']}")
    console.print(f"Chunk size: {stats['chunking_config']['chunk_size']} tokens")
    console.print(f"Overlap: {stats['chunking_config']['chunk_overlap']} tokens")
    
    # Document statistics table
    doc_table = Table(title="Document Statistics", show_header=True, header_style="bold magenta")
    doc_table.add_column("Metric", style="cyan")
    doc_table.add_column("Mean", style="green")
    doc_table.add_column("Median", style="green")
    doc_table.add_column("Min", style="yellow")
    doc_table.add_column("Max", style="yellow")
    doc_table.add_column("Std Dev", style="blue")
    
    # Original length stats
    doc_stats = stats["document_statistics"]
    doc_table.add_row(
        "Original Length (chars)",
        f"{doc_stats['original_length']['mean']:,.0f}",
        f"{doc_stats['original_length']['median']:,.0f}",
        f"{doc_stats['original_length']['min']:,}",
        f"{doc_stats['original_length']['max']:,}",
        f"{doc_stats['original_length']['std']:,.0f}"
    )
    
    # Chunks per document stats
    doc_table.add_row(
        "Chunks per Document",
        f"{doc_stats['chunks_per_document']['mean']:.1f}",
        f"{doc_stats['chunks_per_document']['median']:.1f}",
        f"{doc_stats['chunks_per_document']['min']}",
        f"{doc_stats['chunks_per_document']['max']}",
        f"{doc_stats['chunks_per_document']['std']:.1f}"
    )
    
    console.print()
    console.print(doc_table)
    
    # Chunk statistics table
    chunk_table = Table(title="Chunk Statistics", show_header=True, header_style="bold magenta")
    chunk_table.add_column("Metric", style="cyan")
    chunk_table.add_column("Mean", style="green")
    chunk_table.add_column("Median", style="green")
    chunk_table.add_column("Min", style="yellow")
    chunk_table.add_column("Max", style="yellow")
    chunk_table.add_column("Std Dev", style="blue")
    
    # Character length stats
    chunk_stats = stats["chunk_statistics"]
    chunk_table.add_row(
        "Character Length",
        f"{chunk_stats['character_length']['mean']:.0f}",
        f"{chunk_stats['character_length']['median']:.0f}",
        f"{chunk_stats['character_length']['min']}",
        f"{chunk_stats['character_length']['max']}",
        f"{chunk_stats['character_length']['std']:.0f}"
    )
    
    # Token count stats
    chunk_table.add_row(
        "Token Count",
        f"{chunk_stats['token_count']['mean']:.0f}",
        f"{chunk_stats['token_count']['median']:.0f}",
        f"{chunk_stats['token_count']['min']}",
        f"{chunk_stats['token_count']['max']}",
        f"{chunk_stats['token_count']['std']:.0f}"
    )
    
    console.print()
    console.print(chunk_table)
    
    # Key insights
    console.print(f"\n[bold yellow]Key Insights:[/bold yellow]")
    
    avg_tokens = chunk_stats['token_count']['mean']
    max_tokens = 512  # Our chunk size
    
    console.print(f"• Average chunk uses {avg_tokens:.0f} tokens ({avg_tokens/max_tokens*100:.1f}% of max capacity)")
    
    if chunk_stats['token_count']['max'] > max_tokens:
        console.print(f"• [red]Warning: Some chunks exceed token limit! Max: {chunk_stats['token_count']['max']} tokens[/red]")
    else:
        console.print(f"• [green]✓ All chunks within token limit (max: {chunk_stats['token_count']['max']} tokens)[/green]")
    
    efficiency = (chunk_stats['token_count']['mean'] / max_tokens) * 100
    if efficiency < 70:
        console.print(f"• [yellow]Low token efficiency ({efficiency:.1f}%) - consider reducing chunk size[/yellow]")
    elif efficiency > 95:
        console.print(f"• [yellow]High token efficiency ({efficiency:.1f}%) - chunks are near capacity[/yellow]")
    else:
        console.print(f"• [green]Good token efficiency ({efficiency:.1f}%)[/green]")

def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        stats = analyze_chunking_statistics(
            dataset_name=args.dataset_name,
            num_examples=args.num_examples,
            output_prefix=args.output_prefix,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            model_name=args.model
        )
        if stats:
            console.print("\n[bold green]Analysis completed successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error during analysis: {e}[/bold red]")
        raise

if __name__ == "__main__":
    main()