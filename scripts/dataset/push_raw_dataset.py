"""
Push raw dataset to Hugging Face.
Partially based on https://github.com/huggingface/datasets/blob/14fb15ade27dd8a2cdc4e5992e8d1b9fd1347f1c/src/datasets/arrow_dataset.py#L5401
"""

from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger
from tabulate import tabulate

from juddges.data.pl_court_repo import (
    commit_hf_operations_to_repo,
    prepare_dataset_card,
    prepare_hf_repo_commit_operations,
)
from juddges.settings import PL_JUDGEMENTS_PATH_RAW

typer.rich_utils.STYLE_ERRORS = False

load_dotenv()

DATASET_CARD_TEMPLATE = Path("nbs/Dataset Cards/01_Dataset_Description_Raw.ipynb")
DATASET_CARD_PATH = Path("data/datasets/pl/pl-court-raw/README.md")
DATASET_CARD_ASSETS = Path("data/datasets/pl/pl-court-raw/README_files")


def main(
    repo_id: str = typer.Option(
        ...,
        help="Repository ID",
    ),
    data_files_dir: Path = typer.Option(
        PL_JUDGEMENTS_PATH_RAW,
        help="Path to the dataset directory",
    ),
    dataset_card_template: Path = typer.Option(
        DATASET_CARD_TEMPLATE,
        help="Path to the dataset card template",
    ),
    dataset_card_path: Path = typer.Option(
        DATASET_CARD_PATH,
        help="Path to the dataset card",
    ),
    dataset_card_assets: Path = typer.Option(
        DATASET_CARD_ASSETS,
        help="Path to the dataset card assets",
    ),
    commit_message: str = typer.Option(
        ...,
        help="Commit message",
    ),
) -> None:
    dataset_card_path = prepare_dataset_card(
        data_files_dir=data_files_dir,
        dataset_card_template=dataset_card_template,
        dataset_card_path=dataset_card_path,
    )

    logger.info(f"Created dataset card at: {dataset_card_path}")
    # Ask for confirmation before pushing to HF
    if not typer.confirm("Are you sure you want to push the dataset to Hugging Face?"):
        typer.echo("Operation cancelled.")
        raise typer.Abort()

    operations = prepare_hf_repo_commit_operations(
        repo_id=repo_id,
        commit_message=commit_message,
        data_files_dir=data_files_dir,
        dataset_card_path=dataset_card_path,
        dataset_card_assets=dataset_card_assets,
    )

    operations_table = [(op.path_in_repo, type(op).__name__) for op in operations]
    logger.info(
        "Repository operations:\n"
        f"{tabulate(operations_table, headers=['Path', 'Operation'], tablefmt='grid')}"
    )

    if not typer.confirm("Are you sure you want to push the dataset to Hugging Face?"):
        typer.echo("Operation cancelled.")
        raise typer.Abort()

    commit_hf_operations_to_repo(
        repo_id=repo_id,
        commit_message=commit_message,
        operations=operations,
    )


if __name__ == "__main__":
    typer.run(main)
