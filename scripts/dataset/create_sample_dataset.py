#!/usr/bin/env python3
"""
Create a sample dataset from any HuggingFace dataset with 100 random examples.
Upload it to Hugging Face with a -sample suffix as a private dataset.
"""

import random
import sys
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from rich.console import Console
from rich.panel import Panel


def check_sample_exists(sample_name: str) -> bool:
    """Check if the sample dataset already exists on Hugging Face."""
    console = Console()

    try:
        api = HfApi()
        # Check if dataset exists
        api.dataset_info(sample_name)
        console.print(f"⚠️ Sample dataset [bold yellow]{sample_name}[/bold yellow] already exists")
        return True
    except Exception:
        # Dataset doesn't exist
        return False


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
    """Upload the sampled dataset to Hugging Face as a private dataset."""
    console = Console()

    # Create new dataset name
    if "/" in original_name:
        org, name = original_name.split("/", 1)
        new_name = f"{org}/{name}{sample_suffix}"
    else:
        new_name = f"{original_name}{sample_suffix}"

    console.print(f"📤 Uploading to: [bold cyan]{new_name}[/bold cyan] (private)")

    try:
        # Upload dataset as private
        with console.status(f"[bold green]Uploading {new_name}...", spinner="dots"):
            dataset.push_to_hub(new_name, private=True)

        console.print(f"✅ Successfully uploaded to [bold green]{new_name}[/bold green] (private)")
        return new_name

    except Exception as e:
        console.print(f"❌ Failed to upload: {e}")
        raise


def create_dataset_card(original_name: str, sample_name: str, sample_size: int) -> str:
    """Create a dataset card for the sample dataset."""
    card_content = f"""---
tags:
- sample
- dataset
size_categories:
- n<1K
---

# {sample_name}

This is a sample dataset containing {sample_size} randomly selected examples from [{original_name}](https://huggingface.co/datasets/{original_name}).

## Dataset Description

- **Original Dataset**: {original_name}
- **Sample Size**: {sample_size} examples
- **Type**: Sample dataset for testing and experimentation

## Purpose

This sample dataset is intended for:
- Quick testing and experimentation
- Prototyping without downloading the full dataset
- Educational purposes
- Development and debugging

## Data Fields

The dataset contains the same fields as the original dataset. Please refer to the original dataset's documentation for detailed field descriptions.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{sample_name}")
```

## Original Dataset

This sample is derived from [{original_name}](https://huggingface.co/datasets/{original_name}). 
Please refer to the original dataset for complete documentation, licensing information, and citation details.

## License

This dataset follows the same license as the original dataset: {original_name}
"""
    return card_content


def main():
    """Main function to create and upload sample dataset."""
    console = Console()

    # Configuration
    sample_size = 100
    sample_suffix = "-sample"

    # Get dataset name from command line or use default
    if len(sys.argv) > 1:
        original_dataset_name = sys.argv[1]
    else:
        original_dataset_name = "AI-TAX/pl-eureka-raw"

    console.print(
        Panel.fit(
            "🎯 Create Sample Dataset from Any HuggingFace Dataset",
            style="bold blue",
            border_style="bright_blue",
        )
    )

    console.print("\n📋 [bold blue]Configuration:[/bold blue]")
    console.print(f"  Original Dataset: [cyan]{original_dataset_name}[/cyan]")
    console.print(f"  Sample Size: [cyan]{sample_size}[/cyan]")
    console.print(f"  Sample Suffix: [cyan]{sample_suffix}[/cyan]")
    console.print("  Private Dataset: [cyan]Yes[/cyan]")

    try:
        # Create sample dataset name
        if "/" in original_dataset_name:
            org, name = original_dataset_name.split("/", 1)
            sample_dataset_name = f"{org}/{name}{sample_suffix}"
        else:
            sample_dataset_name = f"{original_dataset_name}{sample_suffix}"

        # Step 0: Check if sample already exists
        console.print("\n🔍 [bold green]Step 0: Checking if sample already exists...[/bold green]")
        if check_sample_exists(sample_dataset_name):
            console.print(
                f"⚠️ Sample dataset [bold yellow]{sample_dataset_name}[/bold yellow] already exists. Skipping creation."
            )
            return

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
            f"🔒 Private dataset created\n"
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
            "3. Internet connection\n"
            "4. Valid dataset name[/blue]"
        )


if __name__ == "__main__":
    main()
