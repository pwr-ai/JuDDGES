from pathlib import Path

import numpy as np
import polars as pl
import torch
import typer
from datasets import load_from_disk
from loguru import logger
from torch import Tensor
from tqdm.auto import tqdm


def main(
    embeddings_dir: Path = typer.Option(...),
) -> None:
    target_file = embeddings_dir.parent / "agg_embeddings.pt"

    if target_file.exists():
        logger.info(f"File {target_file} already exists. Loading sample row...")
        data = torch.load(target_file)
        sample_id = next(iter(data.keys()))
        logger.info(f"Sample judgment_id: {sample_id}")
        logger.info(f"Sample embedding shape: {data[sample_id].shape}")
        return

    embs = load_from_disk(dataset_path=str(embeddings_dir))
    embs.set_format(type="torch", columns=["embedding"])

    logger.info("Converting embeddings to polars...")
    df = embs.to_polars()

    logger.info("Aggregating embeddings...")
    ids, embs = mean_average_embeddings(df)

    logger.info(f"Saving aggregated embeddings at {target_file}")
    torch.save({id_: emb for id_, emb in zip(ids, embs)}, target_file)


def mean_average_embeddings(df: pl.DataFrame) -> tuple[list[str], Tensor]:
    """Aggregates embeddings by taking the mean of document's chunks weighted by their sizes.

    Args:
        df (pl.DataFrame): dataframe with columns ["judgment_id", "chunk_len", "embeddings"]

    Returns:
        tuple[list[str], Tensor]: tuple of list of ids and aggregated embeddings
    """
    ids = []
    emb_dim = df["embedding"][0].shape[0]
    num_unique_docs = df.select(pl.col("judgment_id")).n_unique()
    embs = torch.empty(num_unique_docs, emb_dim, dtype=torch.float)

    for i, (id_, x) in tqdm(
        enumerate(df.group_by(["judgment_id"])), total=num_unique_docs
    ):
        ids.append(id_[0])
        chunk_lengths = torch.tensor(np.stack(x["chunk_len"].to_numpy()))  # type: ignore
        chunk_lengths = chunk_lengths / chunk_lengths.sum()
        chunk_embs = torch.tensor(np.stack(x["embedding"].to_numpy()))  # type: ignore
        chunk_embs = chunk_embs * chunk_lengths[:, None]
        embs[i] = chunk_embs.sum(dim=0)

    return ids, embs


if __name__ == "__main__":
    typer.run(main)
