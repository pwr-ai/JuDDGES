from pathlib import Path
import typer

from juddges.settings import NSA_DATA_PATH
from loguru import logger
from juddges.utils.logging import setup_loguru
from huggingface_hub import HfApi

setup_loguru(extra={"script": __file__})

OUTPUT_PATH = NSA_DATA_PATH / "dataset"
N_JOBS = 10


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


if __name__ == "__main__":
    typer.run(main)
