#!/usr/bin/env python3
"""
Show all datasets available for Weaviate ingestion with the Universal System.
"""

from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def show_available_datasets():
    """Display comprehensive list of available datasets using rich formatting."""

    console = Console()

    # Main header
    console.print(
        Panel.fit(
            "🗂️  COMPLETE DATASET INVENTORY FOR WEAVIATE INGESTION",
            style="bold blue",
            border_style="bright_blue",
        )
    )

    # Legal datasets section
    console.print("\n")
    console.print("📚 LEGAL DATASETS (JuDDGES Project)", style="bold green")

    legal_table = Table(show_header=True, header_style="bold magenta")
    legal_table.add_column("Dataset", style="cyan", width=25)
    legal_table.add_column("Description", style="white", width=30)
    legal_table.add_column("Size", style="yellow", width=15)
    legal_table.add_column("Status", style="green", width=20)

    datasets = [
        {
            "name": "🏛️ JuDDGES/pl-court-raw",
            "description": "Polish court judgments from all court levels",
            "size": "📊 1,200,000+ docs",
            "status": "✅ Ready for ingestion",
        },
        {
            "name": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 JuDDGES/en-court-raw",
            "description": "English & Welsh court judgments",
            "size": "📊 500,000+ docs",
            "status": "✅ Universal mapping",
        },
        {
            "name": "⚖️ JuDDGES/pl-nsa",
            "description": "Polish Supreme Administrative Court",
            "size": "📊 50,000+ docs",
            "status": "✅ Pre-computed embeddings",
        },
        {
            "name": "🏦 JuDDGES/pl-swiss-franc-loans",
            "description": "Swiss franc loan court decisions",
            "size": "📊 10,000+ docs",
            "status": "✅ Banking law focus",
        },
    ]

    for dataset in datasets:
        legal_table.add_row(
            dataset["name"], dataset["description"], dataset["size"], dataset["status"]
        )

    console.print(legal_table)

    # HuggingFace datasets section
    console.print("\n")
    console.print("🌍 UNIVERSAL HUGGINGFACE DATASETS", style="bold green")
    console.print(
        Text("The Universal Ingestion System can adapt to ANY HuggingFace dataset!", style="italic")
    )

    hf_table = Table(show_header=True, header_style="bold magenta")
    hf_table.add_column("Dataset", style="cyan", width=25)
    hf_table.add_column("Description", style="white", width=35)
    hf_table.add_column("Use Case", style="yellow", width=30)

    popular_datasets = [
        {
            "name": "🔍 microsoft/ms-marco",
            "description": "Microsoft Question Answering dataset",
            "use_case": "FAQ systems, document Q&A",
        },
        {
            "name": "📖 squad",
            "description": "Stanford Question Answering Dataset",
            "use_case": "Reading comprehension, Q&A",
        },
        {
            "name": "🔎 natural_questions",
            "description": "Real questions from Google Search",
            "use_case": "Search engines, knowledge bases",
        },
        {
            "name": "🌐 wiki40b/en",
            "description": "Wikipedia articles in 40+ languages",
            "use_case": "Knowledge graphs, encyclopedic search",
        },
        {
            "name": "📰 cnn_dailymail",
            "description": "News articles with summaries",
            "use_case": "News search, summarization",
        },
        {
            "name": "🔬 scientific_papers",
            "description": "Academic papers from arXiv",
            "use_case": "Research databases, literature review",
        },
    ]

    for dataset in popular_datasets:
        hf_table.add_row(dataset["name"], dataset["description"], dataset["use_case"])

    console.print(hf_table)

    # File formats section
    console.print("\n")
    console.print("🔧 LOCAL FILE FORMATS SUPPORTED", style="bold green")

    formats_table = Table(show_header=True, header_style="bold magenta")
    formats_table.add_column("Format", style="cyan", width=20)
    formats_table.add_column("Description", style="white", width=30)
    formats_table.add_column("Example", style="yellow", width=35)

    formats = [
        {
            "format": "📁 Parquet files",
            "description": "High-performance columnar format",
            "example": "data/datasets/pl/raw/*.parquet",
        },
        {
            "format": "📝 JSON Lines",
            "description": "Streaming JSON format",
            "example": "*.jsonl files",
        },
        {
            "format": "📊 CSV files",
            "description": "Comma-separated values",
            "example": "data/datasets/en/csv/judgments.csv",
        },
        {
            "format": "🗂️ Arrow files",
            "description": "Apache Arrow format",
            "example": "data/datasets/en/en_judgements_dataset/",
        },
    ]

    for fmt in formats:
        formats_table.add_row(fmt["format"], fmt["description"], fmt["example"])

    console.print(formats_table)

    # Getting started section
    console.print("\n")
    getting_started_panel = Panel(
        Markdown("""
## 🚀 GETTING STARTED

1. **Preview any dataset:**
   ```bash
   python scripts/dataset_manager.py preview 'dataset-name'
   ```

2. **Auto-register dataset (creates smart mapping):**
   ```bash
   python scripts/dataset_manager.py add 'dataset-name' --auto
   ```

3. **Validate dataset quality:**
   ```bash
   python scripts/dataset_manager.py validate 'dataset-name'
   ```

4. **Ingest with progress tracking:**
   ```bash
   python scripts/dataset_manager.py ingest 'dataset-name' --max-docs 1000
   ```

5. **Use enhanced ingestion script:**
   ```bash
   python scripts/embed/universal_ingest_to_weaviate.py \\
       dataset_name='dataset-name' max_documents=1000
   ```
        """),
        title="Getting Started Guide",
        border_style="bright_green",
    )
    console.print(getting_started_panel)

    # Key features section
    console.print("\n")
    features_text = Text()
    features_text.append("✨ KEY FEATURES\n", style="bold green")
    features = [
        "✅ Zero configuration required - automatic field mapping",
        "✅ Smart data type detection (text, arrays, dates, JSON)",
        "✅ Intelligent chunking for large documents",
        "✅ Batch processing with progress tracking",
        "✅ Error recovery and validation",
        "✅ Memory-efficient streaming",
        "✅ Multi-language support (Polish, English, etc.)",
        "✅ Rich metadata preservation",
    ]

    for feature in features:
        features_text.append(f"{feature}\n", style="green")

    console.print(Panel(features_text, border_style="bright_green"))

    # Recommendations section
    console.print("\n")
    legal_panel = Panel(
        Text(
            "• Start with: JuDDGES/pl-court-raw (1000 docs)\n"
            "• Add: JuDDGES/en-court-raw (1000 docs)\n"
            "• Expand: JuDDGES/pl-nsa (500 docs)",
            style="cyan",
        ),
        title="⚖️ For Legal Document Search",
        border_style="cyan",
    )

    general_panel = Panel(
        Text(
            "• Start with: microsoft/ms-marco (1000 docs)\n"
            "• Add: squad (1000 docs)\n"
            "• Expand: wiki40b/en (5000 docs)",
            style="yellow",
        ),
        title="🌐 For General Knowledge Search",
        border_style="yellow",
    )

    console.print("🎯 RECOMMENDED STARTING DATASETS", style="bold green")
    console.print(Columns([legal_panel, general_panel]))

    # Final note
    console.print("\n")
    console.print(
        Panel.fit(
            "📊 Total ingestion capacity: Limited only by Weaviate storage!",
            style="bold blue",
            border_style="bright_blue",
        )
    )


if __name__ == "__main__":
    show_available_datasets()
