import pandas as pd
import typer
from tqdm import tqdm

from juddges.settings import NSA_DATA_PATH
from juddges.data.nsa.extractor import NSADataExtractor
from loguru import logger
from juddges.utils.logging import setup_loguru

setup_loguru(extra={"script": __file__})

OUTPUT_PATH = NSA_DATA_PATH / "dataset"
N_JOBS = 10


def main(
    n_jobs: int = typer.Option(N_JOBS),
) -> None:
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
    files = list(sorted(NSA_DATA_PATH.glob("pages/pages_chunk_*.parquet")))

    logger.info(f"Extracting data from {len(files)} files to {OUTPUT_PATH}...")

    extractor = NSADataExtractor()
    for path in tqdm(files):
        df = pd.read_parquet(path)
        data = extractor.extract_data_from_pages_to_df(df["page"], df["doc_id"], n_jobs=n_jobs)
        data.to_parquet(OUTPUT_PATH / f"data_{path.stem.split('_')[-1]}.parquet")
        del df
        del data


if __name__ == "__main__":
    typer.run(main)
