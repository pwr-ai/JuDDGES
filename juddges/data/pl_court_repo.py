import subprocess
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, DatasetCardData, HfApi
from huggingface_hub.errors import RepositoryNotFoundError
from loguru import logger

from juddges.utils.hf import disable_hf_dataset_cache

DEFAULT_DATA_DIR_IN_REPO = "data"
DATA_SHARD_FILE_PATTERN = "train_*.parquet"
DEFAULT_ASSETS_DIR_IN_REPO = "README_files"

T_operations = list[CommitOperationAdd | CommitOperationDelete]


def prepare_hf_repo_commit_operations(
    repo_id: str,
    data_files_dir: Path,
    dataset_card_path: Path,
    dataset_card_assets: Path,
) -> T_operations:
    assert data_files_dir.exists()
    assert list(data_files_dir.glob(DATA_SHARD_FILE_PATTERN))

    _check_repo_exists(repo_id)
    api = HfApi()

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

    return operations


def commit_hf_operations_to_repo(
    repo_id: str,
    commit_message: str,
    operations: T_operations,
) -> None:
    _check_repo_exists(repo_id)
    api = HfApi()
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
    with disable_hf_dataset_cache():
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

    return dataset_card_path


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


def _check_repo_exists(repo_id: str):
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        logger.error(f"Repository {repo_id} does not exist")
        raise
