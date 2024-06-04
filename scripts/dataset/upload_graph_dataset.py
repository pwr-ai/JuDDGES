import json
from pathlib import Path
from typing import Any
import networkx as nx
from omegaconf import OmegaConf
import typer
from huggingface_hub import DatasetCard, DatasetCardData, HfApi

DATASET_CARD_TEMPLATE = "data/datasets/pl/graph/README.md"


def main(
    root_dir: Path = typer.Option(...),
    repo_id: str = typer.Option(...),
    commit_message: str = typer.Option("Add dataset", "--commit-message", "-m"),
    dry: bool = typer.Option(False, "--dry-run", "-d"),
):
    card_data = DatasetCardData(
        language="pl",
        pretty_name="Polish Court Judgments Graph",
        size_categories="100K<n<1M",
        source_datasets=["pl-court-raw"],
        viewer=False,
        tags=["graph", "bipartite", "polish court"],
    )
    card = DatasetCard.from_template(
        card_data,
        template_path=DATASET_CARD_TEMPLATE,
    )
    card.save(root_dir / "README.md")

    if not dry:
        api = HfApi()
        api.upload_folder(
            folder_path=str(root_dir),
            repo_id=repo_id,
            repo_type="dataset",
            ignore_patterns=["metadata.yaml"],
            commit_message=commit_message,
        )

if __name__ == "__main__":
    typer.run(main)
