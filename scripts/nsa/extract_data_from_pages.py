import pandas as pd
from tqdm import tqdm

from juddges.settings import NSA_DATA_PATH
from juddges.data.nsa.extractor import NSADataExtractor

extractor = NSADataExtractor()


OUTPUT_PATH = NSA_DATA_PATH / "dataset"
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

N_JOBS = 10

files = list(sorted(NSA_DATA_PATH.glob("pages/pages_chunk_*.parquet")))

for path in tqdm(files):
    print(f"Extracting data from {path}")
    df = pd.read_parquet(path)
    data = extractor.extract_data_from_pages_to_df(df["page"], df["doc_id"], n_jobs=N_JOBS)
    data.to_parquet(OUTPUT_PATH / f"data_{path.stem.split('_')[-1]}.parquet")
    del df
    del data

# # Define the batch size
# batch_size = 10000
#
# # Open the Parquet file
# parquet_file = pq.ParquetFile(NSA_DATA_PATH / "pages" / "pages_chunk_0.parquet")
#
# # Iterate over the file in batches
# for batch in parquet_file.iter_batches(batch_size=batch_size):
#     df = batch.to_pandas()
#     # Process each batch of data as a Pandas DataFrame
#     print(df.head())  # Example: just print the first few rows
