import math
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from tqdm.auto import tqdm

from juddges.preprocessing.text_chunker import TextSplitter


def mean_average_embeddings_and_save(
    df: pl.LazyFrame,
    num_shards: int,
    output_dir: Path,
    id_col: str,
    embedding_col: str,
) -> None:
    """Aggregates embeddings by taking the mean of document's chunks weighted by their sizes.

    Args:
        df (pl.LazyFrame): dataframe with columns ["judgment_id", "chunk_len", "embedding"]

    Returns:
        tuple[list[str], Tensor]: tuple of list of ids and aggregated embeddings
    """
    schema = df.collect_schema()
    required_cols = {id_col, TextSplitter.CHUNK_LEN_COL, embedding_col}
    assert required_cols.issubset(
        schema.names()
    ), f"Missing required columns: {required_cols - set(schema.names())}"

    num_unique_docs = df.select(pl.col(id_col).n_unique()).collect().item()
    logger.info(f"Number of unique documents with aggregated embeddings: {num_unique_docs}")

    # Create a normalized weight column
    df = df.with_columns(
        pl.col(TextSplitter.CHUNK_LEN_COL).over(id_col).alias("weight")
        / pl.col(TextSplitter.CHUNK_LEN_COL).sum().over(id_col)
    )

    # Multiply embeddings by weights and sum
    df = df.with_columns((pl.col(embedding_col) * pl.col("weight")).alias("weighted_embedding"))

    # Group by and aggregate using a custom function that can handle numpy arrays
    def weighted_sum(series: pl.Series) -> list[float]:
        # Convert to numpy arrays and sum
        arrays = [np.array(x) for x in series]
        return np.sum(arrays, axis=0).tolist()

    agg_df = df.group_by(id_col).agg(
        pl.col("weighted_embedding")
        .map_elements(weighted_sum, return_dtype=pl.List(pl.Float64))
        .alias(embedding_col)
    )

    shard_size = math.ceil(num_unique_docs / num_shards)
    for i in tqdm(range(num_shards), desc="Aggregating embeddings"):
        shard = agg_df.slice(i * shard_size, i * shard_size + shard_size).collect()
        shard.write_parquet(output_dir / f"mean_embeddings_{i}.parquet")
