"""
Push raw dataset to Hugging Face.
Partially based on https://github.com/huggingface/datasets/blob/14fb15ade27dd8a2cdc4e5992e8d1b9fd1347f1c/src/datasets/arrow_dataset.py#L5401
"""

from pathlib import Path

import typer
from dotenv import load_dotenv

from juddges.data.pl_court_repo import prepare_dataset_card_and_push_data_to_hf_repo
from juddges.settings import PL_JUDGEMENTS_PATH_RAW

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
    prepare_dataset_card_and_push_data_to_hf_repo(
        repo_id=repo_id,
        commit_message=commit_message,
        data_files_dir=data_files_dir,
        dataset_card_template=dataset_card_template,
        dataset_card_path=dataset_card_path,
        dataset_card_assets=dataset_card_assets,
    )


if __name__ == "__main__":
    typer.run(main)
