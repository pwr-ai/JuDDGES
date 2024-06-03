from pathlib import Path
from typing import Optional

import typer
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import DatasetCardData, DatasetCard, HfApi
from loguru import logger

from juddges.settings import PL_JUDGEMENTS_PATH_RAW

load_dotenv()

MAX_SHARD_SIZE = "4GB"

DATASET_CARD_TEMPLATE = Path("data/datasets/pl/readme/raw/README.md")
DATASET_CARD_TEMPLATE_FILES = Path("data/datasets/pl/readme/raw/README_files")


def main(
    dataset_dir: Path = typer.Option(PL_JUDGEMENTS_PATH_RAW, help="Path to the dataset directory"),
    repo_id: Optional[str] = typer.Option(...),
    branch: Optional[str] = typer.Option(None, help="Branch to push the dataset to"),
) -> None:
    logger.info("Loading dataset...")
    ds = load_dataset("parquet", name="pl_judgements", data_dir=dataset_dir)

    num_rows = ds["train"].num_rows
    logger.info(f"Loaded dataset size: {num_rows}")

    ds.push_to_hub(repo_id, max_shard_size=MAX_SHARD_SIZE, revision=branch)

    # Card creation

    assert 100_000 < num_rows < 1_000_000
    card_data = DatasetCardData(
        language="pl",
        multilinguality="monolingual",
        language_creators=["found"],
        size_categories="100K<n<1M",
        source_datasets=["original"],
        pretty_name="Polish Court Judgments Raw",
        tags=["polish court"],
    )
    card = DatasetCard.from_template(
        card_data,
        template_path=DATASET_CARD_TEMPLATE,
    )
    card.push_to_hub(repo_id, revision=branch)

    api = HfApi()
    api.upload_folder(
        folder_path=DATASET_CARD_TEMPLATE_FILES,
        path_in_repo=DATASET_CARD_TEMPLATE_FILES.name,
        repo_id=repo_id,
        repo_type="dataset",
        revision=branch,
    )


if __name__ == "__main__":
    typer.run(main)
