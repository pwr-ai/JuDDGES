from pathlib import Path
import typer

from juddges.settings import NSA_DATA_PATH
from loguru import logger
from juddges.utils.logging import setup_loguru
from huggingface_hub import HfApi, DatasetCardData, DatasetCard

setup_loguru(extra={"script": __file__})

OUTPUT_PATH = NSA_DATA_PATH / "dataset"
DATASET_CARD_DIR = Path("data/datasets/nsa/readme")


def main(
    repo_name: str = typer.Option("JuDDGES/nsa"),
    log_file: Path = typer.Option(None, help="Log file to save the logs to."),
) -> None:
    setup_loguru(extra={"script": __file__}, log_file=log_file)
    logger.info("Running push_to_repo.py with args:\n" + str(locals()))

    api = HfApi()
    api.upload_folder(
        repo_id=repo_name,
        path_in_repo="data",
        folder_path=str(OUTPUT_PATH),
        repo_type="dataset",
        delete_patterns="*.parquet",
    )

    card_data = DatasetCardData(
        language="pl",
        multilinguality="monolingual",
        size_categories="1M<n<10M",
        source_datasets=["original"],
        pretty_name="Supreme Administrative Court of Poland Judgements",
        tags=["polish court"],
    )
    card = DatasetCard.from_template(
        card_data,
        template_path=DATASET_CARD_DIR / "README.md",
    )
    card.push_to_hub(repo_name)
    api.upload_folder(
        folder_path=DATASET_CARD_DIR / "README_files",
        path_in_repo="README_files",
        repo_id=repo_name,
        repo_type="dataset",
    )


if __name__ == "__main__":
    typer.run(main)
