#!/usr/bin/env python3
"""
Example usage of the simplified streaming ingester.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from juddges.data.stream_ingester import StreamingIngester
from rich.console import Console


def main():
    """Example usage of StreamingIngester."""
    console = Console()
    
    console.print("[bold blue]Simple Streaming Ingestion Example[/bold blue]")
    
    # Example 1: Basic usage with local dataset
    console.print("\n[bold green]Example 1: Basic Usage[/bold green]")
    
    try:
        # Initialize ingester
        ingester = StreamingIngester(
            weaviate_url="http://localhost:8080",
            embedding_model="sdadas/mmlw-roberta-large",
            chunk_size=512,
            overlap=128,
            batch_size=32,
            tracker_db="example_processed.db"
        )
        
        # Process a small sample dataset
        console.print("Processing dataset...")
        stats = ingester.process_dataset(
            dataset_path="JuDDGES/pl-court-raw",
            streaming=True
        )
        
        console.print(f"✅ Processed {stats.processed_documents} documents")
        
    except Exception as e:
        console.print(f"❌ Error: {e}")
    
    # Example 2: Processing with custom settings
    console.print("\n[bold green]Example 2: Custom Settings[/bold green]")
    
    try:
        # Initialize with custom settings
        custom_ingester = StreamingIngester(
            weaviate_url="http://localhost:8080",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model
            chunk_size=256,  # Smaller chunks
            overlap=64,
            batch_size=16,
            tracker_db="custom_processed.db"
        )
        
        # Reset tracker for fresh start
        custom_ingester.reset_tracker()
        
        console.print("Processing with custom settings...")
        # This would process a different dataset or same dataset with different settings
        
    except Exception as e:
        console.print(f"❌ Error: {e}")
    
    # Example 3: Resume processing after interruption
    console.print("\n[bold green]Example 3: Resume Processing[/bold green]")
    
    try:
        # Create new ingester with same tracker DB
        resume_ingester = StreamingIngester(
            weaviate_url="http://localhost:8080",
            embedding_model="sdadas/mmlw-roberta-large",
            tracker_db="example_processed.db"  # Same DB as example 1
        )
        
        # This will skip already processed documents
        console.print("Resuming processing (will skip already processed documents)...")
        stats = resume_ingester.process_dataset(
            dataset_path="JuDDGES/pl-court-raw",
            streaming=True
        )
        
        console.print(f"✅ Skipped {stats.skipped_documents} already processed documents")
        
    except Exception as e:
        console.print(f"❌ Error: {e}")
    
    console.print("\n[bold blue]Examples completed![/bold blue]")


if __name__ == "__main__":
    main()