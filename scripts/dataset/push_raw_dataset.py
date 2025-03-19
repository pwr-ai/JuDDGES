"""
Push raw dataset to Hugging Face.
Partially based on https://github.com/huggingface/datasets/blob/14fb15ade27dd8a2cdc4e5992e8d1b9fd1347f1c/src/datasets/arrow_dataset.py#L5401
"""

import subprocess
from pathlib import Path

import typer
from datasets import disable_caching, enable_caching, load_dataset
from dotenv import load_dotenv
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    DatasetCardData,
    HfApi,
)
from huggingface_hub.errors import RepositoryNotFoundError
from loguru import logger
from tabulate import tabulate

from juddges.settings import PL_JUDGEMENTS_PATH_RAW

load_dotenv()

DATASET_CARD_TEMPLATE = Path("nbs/Dataset Cards/01_Dataset_Description_Raw.ipynb")
DATASET_CARD_PATH = Path("data/datasets/pl/pl-court-raw/README.md")
DATASET_CARD_ASSETS = Path("data/datasets/pl/pl-court-raw/README_files")

DEFAULT_DATA_DIR_IN_REPO = PL_JUDGEMENTS_PATH_RAW.name
DEFAULT_ASSETS_DIR_IN_REPO = DATASET_CARD_ASSETS.name


def main(
    repo_id: str = typer.Option(
        ...,
        help="Repository ID",
    ),
    data_files_dir: Path = typer.Option(
        PL_JUDGEMENTS_PATH_RAW,
        help="Path to the dataset directory",
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
    assert data_files_dir.exists()
    assert list(data_files_dir.glob("*.parquet"))

    api = HfApi()

    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        logger.error(f"Repository {repo_id} does not exist")
        raise typer.Abort()

    prepare_dataset_card(data_files_dir)

    # Replace old data files with new ones
    deletions = []
    for f_name in api.list_repo_files(
        repo_id,
        repo_type="dataset",
    ):
        if f_name.startswith(f"{DEFAULT_DATA_DIR_IN_REPO}/") or f_name.startswith(
            f"{DEFAULT_ASSETS_DIR_IN_REPO}/"
        ):
            deletions.append(CommitOperationDelete(path_in_repo=f_name))

    additions = []
    for file_path in data_files_dir.glob("*.parquet"):
        additions.append(
            CommitOperationAdd(
                path_in_repo=f"{DEFAULT_DATA_DIR_IN_REPO}/{file_path.name}",
                path_or_fileobj=file_path,
            )
        )

    # Replace readme and readme assets with new ones
    deletions.append(CommitOperationDelete(path_in_repo="README.md"))

    additions.append(
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=dataset_card_path,
        )
    )

    for f_name in dataset_card_assets.glob("*"):
        additions.append(
            CommitOperationAdd(
                path_in_repo=f"README_files/{f_name.name}",
                path_or_fileobj=f_name,
            )
        )

    operations = deletions + additions

    operations_table = [(op.path_in_repo, type(op).__name__) for op in operations]
    print("\nProposed changes:")
    print(tabulate(operations_table, headers=["Path", "Operation"], tablefmt="grid"))

    if not typer.confirm("Do you want to proceed with these changes?"):
        raise typer.Abort()

    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=commit_message,
    )


def prepare_dataset_card(data_files_dir: Path) -> Path:
    disable_caching()
    dataset_info = load_dataset("parquet", data_dir=data_files_dir)["train"].info
    enable_caching()

    generate_dataset_card()

    card_data = DatasetCardData(
        language="pl",
        multilinguality="monolingual",
        size_categories="100K<n<1M",
        source_datasets=["original"],
        pretty_name="Polish Court Judgments Raw",
        tags=["polish court"],
        configs=[
            {
                "config_name": "default",
                "data_files": [{"split": "train", "path": f"{DEFAULT_DATA_DIR_IN_REPO}/train_*"}],
            }
        ],
        dataset_info=dataset_info._to_yaml_dict(),
    )

    with DATASET_CARD_PATH.open("r") as f:
        card_content = f.read()

    card_content = f"---\n{card_data}\n---\n\n{card_content}"

    with DATASET_CARD_PATH.open("w") as f:
        f.write(card_content)


def generate_dataset_card() -> Path:
    logger.info("Generating dataset card...")
    cmd = [
        "jupyter",
        "nbconvert",
        "--no-input",
        "--to",
        "markdown",
        "--execute",
        str(DATASET_CARD_TEMPLATE),
        "--output-dir",
        str(DATASET_CARD_PATH.parent),
        "--output",
        DATASET_CARD_PATH.stem,
    ]
    subprocess.run(cmd, check=True)
    return DATASET_CARD_PATH


if __name__ == "__main__":
    typer.run(main)
