from asyncio import subprocess
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, DatasetCardData, HfApi
from huggingface_hub.errors import RepositoryNotFoundError
from loguru import logger
from tabulate import tabulate

from juddges.utils.hf import disable_caching_ctx

DEFAULT_DATA_DIR_IN_REPO = "data"
DATA_SHARD_FILE_PATTERN = "train_*.parquet"
DEFAULT_ASSETS_DIR_IN_REPO = "README_files"


def prepare_dataset_card_and_push_data_to_hf_repo(
    repo_id: str,
    commit_message: str,
    data_files_dir: Path,
    dataset_card_template: str | Path,
    dataset_card_path: Path,
    dataset_card_assets: Path,
) -> None:
    assert data_files_dir.exists()
    assert list(data_files_dir.glob(DATA_SHARD_FILE_PATTERN))

    api = HfApi()

    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        logger.error(f"Repository {repo_id} does not exist")
        raise

    prepare_dataset_card(
        data_files_dir=data_files_dir,
        dataset_card_template=dataset_card_template,
        dataset_card_path=dataset_card_path,
    )

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
    logger.info(
        "Repository operations:\n"
        f"{tabulate(operations_table, headers=['Path', 'Operation'], tablefmt='grid')}"
    )

    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=commit_message,
    )


def prepare_dataset_card(
    data_files_dir: Path,
    dataset_card_template: str | Path,
    dataset_card_path: Path,
) -> Path:
    with disable_caching_ctx():
        dataset_info = load_dataset("parquet", data_dir=data_files_dir)["train"].info

    generate_dataset_card(
        dataset_card_template=dataset_card_template,
        dataset_card_path=dataset_card_path,
    )

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
                "data_files": [
                    {
                        "split": "train",
                        "path": f"{DEFAULT_DATA_DIR_IN_REPO}/{DATA_SHARD_FILE_PATTERN}",
                    }
                ],
            }
        ],
        dataset_info=dataset_info._to_yaml_dict(),
    )

    with dataset_card_path.open("r") as f:
        card_content = f.read()

    card_content = f"---\n{card_data}\n---\n\n{card_content}"

    with dataset_card_path.open("w") as f:
        f.write(card_content)


def generate_dataset_card(
    dataset_card_template: str | Path,
    dataset_card_path: str | Path,
) -> Path:
    logger.info("Generating dataset card...")
    cmd = [
        "jupyter",
        "nbconvert",
        "--no-input",
        "--to",
        "markdown",
        "--execute",
        str(dataset_card_template),
        "--output-dir",
        str(dataset_card_path.parent),
        "--output",
        dataset_card_path.stem,
    ]
    subprocess.run(cmd, check=True)

    assert dataset_card_path.exists()
    assert dataset_card_path.is_file()

    assets_dir = dataset_card_path.parent / DEFAULT_ASSETS_DIR_IN_REPO
    assert assets_dir.exists()
    assert assets_dir.is_dir()

    return dataset_card_path
