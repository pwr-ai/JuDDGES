import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow.dataset as ds
from huggingface_hub import DatasetCard, upload_folder


def push_dataset_dir_to_hub(
    dataset_path: Path | str,
    card: DatasetCard,
    card_assets: Path | str | None,
    repo_id: str,
    commit_message: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        dataset_path = Path(dataset_path)
        assert all(ds_file.name.endswith(".parquet") for ds_file in dataset_path.glob("*.parquet"))
        shutil.copytree(
            dataset_path,
            tmp_path / "data",
            dirs_exist_ok=True,
        )

        readme_path = tmp_path / "README.md"
        card.save(readme_path)

        if card_assets is not None:
            card_assets = Path(card_assets)
            assets_dir = tmp_path / card_assets.name
            shutil.copytree(
                card_assets,
                assets_dir,
                dirs_exist_ok=True,
            )

        print(f"Pushing dataset to {repo_id} with commit message: {commit_message}")
        res = subprocess.run(["tree", tmp_path], capture_output=True, text=True, check=True)
        print(res.stdout)

        upload_folder(
            folder_path=tmp_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )


def get_parquet_num_rows(dataset_path: Path | str) -> int:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise ValueError("Dataset path does not exist")

    if dataset_path.is_dir() and not list(dataset_path.glob("*.parquet")):
        raise ValueError("Dataset path does not contain any parquet files")

    dataset = ds.dataset(dataset_path)
    total_rows = sum(
        rg.num_rows for fragment in dataset.get_fragments() for rg in fragment.row_groups
    )

    return total_rows
