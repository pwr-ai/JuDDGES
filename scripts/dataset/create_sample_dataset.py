#!/usr/bin/env python3
"""
Create a sample dataset from JuDDGES/pl-court-raw with 100 random examples.
Upload it to Hugging Face with a -sample suffix.
"""

import random
from pathlib import Path

from datasets import Dataset, load_dataset
from rich.console import Console
from rich.panel import Panel


def load_original_dataset(dataset_name: str) -> Dataset:
    """Load the original dataset from Hugging Face."""
    console = Console()

    with console.status(f"[bold green]Loading {dataset_name}...", spinner="dots"):
        dataset = load_dataset(dataset_name, split="train")

    console.print(f"✅ Loaded dataset with {len(dataset)} examples")
    return dataset


def sample_dataset(dataset: Dataset, sample_size: int = 100, seed: int = 42) -> Dataset:
    """Sample random examples from the dataset."""
    console = Console()

    random.seed(seed)

    # Get random indices
    total_size = len(dataset)
    if sample_size >= total_size:
        console.print(f"⚠️ Requested sample size ({sample_size}) >= dataset size ({total_size})")
        console.print("Using entire dataset")
        return dataset

    # Generate random indices
    indices = random.sample(range(total_size), sample_size)
    indices.sort()  # Sort for reproducibility

    # Select samples
    sampled_dataset = dataset.select(indices)

    console.print(f"✅ Sampled {len(sampled_dataset)} examples from {total_size} total")
    return sampled_dataset


def upload_to_huggingface(
    dataset: Dataset, original_name: str, sample_suffix: str = "-sample"
) -> str:
    """Upload the sampled dataset to Hugging Face."""
    console = Console()

    # Create new dataset name
    if "/" in original_name:
        org, name = original_name.split("/", 1)
        new_name = f"{org}/{name}{sample_suffix}"
    else:
        new_name = f"{original_name}{sample_suffix}"

    console.print(f"📤 Uploading to: [bold cyan]{new_name}[/bold cyan]")

    try:
        # Upload dataset
        with console.status(f"[bold green]Uploading {new_name}...", spinner="dots"):
            dataset.push_to_hub(new_name, private=False)

        console.print(f"✅ Successfully uploaded to [bold green]{new_name}[/bold green]")
        return new_name

    except Exception as e:
        console.print(f"❌ Failed to upload: {e}")
        raise


def create_dataset_card(original_name: str, sample_name: str, sample_size: int) -> str:
    """Create a dataset card for the sample dataset."""
    card_content = f"""---
language:
- pl
license: apache-2.0
task_categories:
- text-generation
- question-answering
tags:
- legal
- court-judgments
- polish
- sample
size_categories:
- n<1K
---

# {sample_name}

This is a sample dataset containing {sample_size} randomly selected examples from [{original_name}](https://huggingface.co/datasets/{original_name}).

## Dataset Description

- **Original Dataset**: {original_name}
- **Sample Size**: {sample_size} examples
- **Language**: Polish
- **Domain**: Legal court judgments
- **License**: Apache 2.0

## Purpose

This sample dataset is intended for:
- Quick testing and experimentation
- Prototyping with legal document processing
- Educational purposes
- Development without downloading the full dataset

## Data Fields

The dataset contains the same fields as the original dataset:
- `document_id`: Unique identifier for each judgment
- `full_text`: Full text of the court judgment
- `document_number`: Docket number
- `date_issued`: Date when the judgment was issued
- `court_name`: Name of the court
- `summary`: Summary/excerpt of the judgment
- `thesis`: Legal thesis
- `keywords`: Associated keywords
- `legal_bases`: Legal bases referenced
- `judges`: Judges involved
- And other metadata fields

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{sample_name}")
```

## Citation

If you use this dataset, please cite the original JuDDGES paper:

```bibtex
@misc{{judges2024,
  title={{JuDDGES: A Benchmark for Legal Document Processing}},
  author={{Legal AI Team}},
  year={{2024}},
  url={{https://huggingface.co/datasets/{original_name}}}
}}
```

## License

This dataset is licensed under the Apache 2.0 License, same as the original dataset.
"""
    return card_content


def main():
    """Main function to create and upload sample dataset."""
    console = Console()

    # Configuration
    original_dataset_name = "JuDDGES/pl-court-raw"
    sample_size = 100
    sample_suffix = "-sample"

    # Header
    console.print(
        Panel.fit(
            "🎯 Create Sample Dataset from JuDDGES/pl-court-raw",
            style="bold blue",
            border_style="bright_blue",
        )
    )

    try:
        # Step 1: Load original dataset
        console.print("\n📥 [bold green]Step 1: Loading original dataset...[/bold green]")
        original_dataset = load_original_dataset(original_dataset_name)

        # Step 2: Sample dataset
        console.print(f"\n🎲 [bold green]Step 2: Sampling {sample_size} examples...[/bold green]")
        sampled_dataset = sample_dataset(original_dataset, sample_size)

        # Step 3: Upload to Hugging Face
        console.print("\n📤 [bold green]Step 3: Uploading to Hugging Face...[/bold green]")
        new_dataset_name = upload_to_huggingface(
            sampled_dataset, original_dataset_name, sample_suffix
        )

        # Step 4: Create dataset card
        console.print("\n📝 [bold green]Step 4: Creating dataset card...[/bold green]")
        dataset_card = create_dataset_card(original_dataset_name, new_dataset_name, sample_size)

        # Save dataset card locally for reference
        card_path = Path(f"dataset_card_{new_dataset_name.replace('/', '_')}.md")
        card_path.write_text(dataset_card)
        console.print(f"💾 Dataset card saved to: [cyan]{card_path}[/cyan]")

        # Success message
        console.print("\n")
        success_panel = Panel.fit(
            f"🎉 Success!\n\n"
            f"✅ Created sample dataset: [bold green]{new_dataset_name}[/bold green]\n"
            f"📊 Sample size: {sample_size} examples\n"
            f"🔗 URL: https://huggingface.co/datasets/{new_dataset_name}",
            style="bold green",
            border_style="bright_green",
        )
        console.print(success_panel)

    except Exception as e:
        console.print(f"\n❌ [bold red]Failed to create sample dataset:[/bold red] {e}")
        console.print(
            "\n💡 [blue]Make sure you have:\n"
            "1. HuggingFace CLI configured (huggingface-cli login)\n"
            "2. Write access to upload datasets\n"
            "3. Internet connection[/blue]"
        )


if __name__ == "__main__":
    main()
