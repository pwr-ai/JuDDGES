import subprocess
import sys
from pathlib import Path
from typing import Optional
from juddges.settings import ROOT_PATH
from juddges.utils.logging import setup_loguru

from loguru import logger
from rich import print

import typer

setup_loguru(extra={"script": __file__})


NSA_SCRIPTS_PATH = ROOT_PATH / "scripts" / "nsa"


def main(
    proxy_address: str = typer.Option(
        ..., help="Proxy address. eg. http://user:password@address:port"
    ),
    db_uri: str = typer.Option(..., help="MongoDB URI"),
    start_date: str = typer.Option(
        "1981-01-01", help="Start date for scraping (YYYY-MM-DD). Defaults to 1981-01-01."
    ),
    end_date: Optional[str] = typer.Option(
        None, help="End date for scraping (YYYY-MM-DD). Defaults to yesterday's date in Poland."
    ),
    n_jobs: int = typer.Option(25, help="Number of parallel workers"),
    scrap_dates_iterations: int = typer.Option(1, help="Number of iterations to scrap dates."),
    cleanup_iterations: int = typer.Option(
        1, help="Number of cleanup iterations to perform. Defaults to 1."
    ),
) -> None:
    logger.info("Running full procedure with args:\n" + str(locals()))

    base_args = [
        "--proxy-address",
        proxy_address,
        "--db-uri",
        db_uri,
    ]

    if n_jobs:
        base_args.extend(["--n-jobs", str(n_jobs)])

    scrap_documents_list_args = base_args.copy()
    if start_date:
        scrap_documents_list_args.extend(["--start-date", start_date])

    if end_date:
        scrap_documents_list_args.extend(["--end-date", end_date])

    # Define the pipeline steps
    pipeline = [
        ("scrap_documents_list.py", scrap_documents_list_args)
        for _ in range(scrap_dates_iterations)
    ]
    for _ in range(cleanup_iterations):
        pipeline.extend(
            [
                ("drop_dates_with_duplicated_documents.py", ["--db-uri", db_uri]),
                ("scrap_documents_list.py", scrap_documents_list_args),
            ]
        )

    pipeline.append(("download_document_pages.py", base_args))

    for _ in range(cleanup_iterations):
        pipeline.extend(
            [
                ("drop_duplicated_document_pages.py", ["--db-uri", db_uri]),
                ("download_document_pages.py", base_args),
            ]
        )

    pipeline.extend(
        [
            ("save_pages_from_db_to_file.py", ["--db-uri", db_uri]),
            ("extract_data_from_pages.py", ["--n-jobs", str(n_jobs)]),
        ]
    )

    print("Pipeline steps:")
    for i, step in enumerate(pipeline):
        print(f"{i}. {step[0]}")

    confirm = typer.confirm("Are you sure you want to run the pipeline?")
    if not confirm:
        logger.error("Pipeline cancelled.")
        raise typer.Abort()

    for step in pipeline:
        script_name, script_args = step
        script_path = NSA_SCRIPTS_PATH / script_name
        assert script_path.exists()

        if not run_script(script_path, script_args):
            logger.error(f"Pipeline failed at step: {script_name}")
            raise typer.Exit(code=1)

    logger.info("Pipeline completed successfully!")


def run_script(script_path: Path, args: list[str]) -> bool:
    """
    Run a Python script and return True if successful, False otherwise.
    """
    cmd = [sys.executable, str(script_path)] + args
    try:
        logger.info(f"Running {script_path} with args: {' '.join(args)}")
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully completed {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}: {e}")
        return False


if __name__ == "__main__":
    typer.run(main)
