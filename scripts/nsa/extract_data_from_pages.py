from pathlib import Path

import pyarrow.parquet as pq
import typer
from loguru import logger

from juddges.data.nsa.extractor import NSADataExtractor
from juddges.data.nsa.state import NSAState
from juddges.settings import NSA_DATA_PATH
from juddges.utils.logging import setup_loguru

OUTPUT_PATH = NSA_DATA_PATH / "dataset"
N_JOBS = 10


def main(
    n_jobs: int = typer.Option(N_JOBS),
    state_dir: Path = typer.Option(None, help="State directory (default: NSA_DATA_PATH/state)."),
    output_path: Path = typer.Option(OUTPUT_PATH, help="Directory for data_{i}.parquet files."),
    log_file: Path = typer.Option(None, help="Log file to save the logs to."),
) -> None:
    setup_loguru(extra={"script": __file__}, log_file=log_file)
    logger.info("Running extract_data_from_pages.py with args:\n" + str(locals()))

    output_path.mkdir(exist_ok=True, parents=True)
    if any(output_path.glob("data_*.parquet")):
        if not typer.confirm(f"Output directory {output_path} has data files. Remove them?"):
            raise typer.Abort()
        for file in output_path.glob("data_*.parquet"):
            file.unlink()

    extractor = NSADataExtractor()
    for i, df in enumerate(NSAState(state_dir).iter_latest_pages()):
        data = extractor.extract_data_from_pages_to_pyarrow(df["page"], df["doc_id"], n_jobs=n_jobs)
        pq.write_table(data, output_path / f"data_{i}.parquet")
        logger.info(f"Saved data_{i}.parquet with {len(df)} documents")


if __name__ == "__main__":
    typer.run(main)
