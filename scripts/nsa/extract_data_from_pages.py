from pathlib import Path

import pyarrow.parquet as pq
import typer
from loguru import logger

from juddges.data.nsa.extractor import NSADataExtractor
from juddges.data.nsa.state import NSAState, StateLocked
from juddges.settings import NSA_DATA_PATH
from juddges.utils.logging import setup_loguru

OUTPUT_PATH = NSA_DATA_PATH / "dataset"
N_JOBS = 10


def main(
    n_jobs: int = typer.Option(N_JOBS),
    state_dir: Path = typer.Option(None, help="State directory (default: NSA_DATA_PATH/state)."),
    output_path: Path = typer.Option(OUTPUT_PATH, help="Directory for data_{i}.parquet files."),
    rows_per_file: int = typer.Option(
        50_000, help="Documents per data_{i}.parquet (all of them are held in memory at once)."
    ),
    log_file: Path = typer.Option(None, help="Log file to save the logs to."),
) -> None:
    setup_loguru(extra={"script": __file__}, log_file=log_file)
    logger.info("Running extract_data_from_pages.py with args:\n" + str(locals()))

    state = NSAState(state_dir)
    try:
        state.acquire_lock()
    except StateLocked as e:
        logger.error(str(e))
        raise typer.Exit(1)

    output_path.mkdir(exist_ok=True, parents=True)
    if any(output_path.glob("data_*.parquet")):
        if not typer.confirm(f"Output directory {output_path} has data files. Remove them?"):
            raise typer.Abort()
        for file in output_path.glob("data_*.parquet"):
            file.unlink()

    extractor = NSADataExtractor()
    for i, df in enumerate(state.iter_latest_pages(min_rows=rows_per_file)):
        data = extractor.extract_data_from_pages_to_pyarrow(df["page"], df["doc_id"], n_jobs=n_jobs)
        pq.write_table(data, output_path / f"data_{i}.parquet")
        logger.info(f"Saved data_{i}.parquet with {len(df)} documents")


if __name__ == "__main__":
    typer.run(main)
