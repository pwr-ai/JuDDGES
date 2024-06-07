from pathlib import Path
from loguru import logger
import numpy as np
from tqdm.auto import tqdm
import typer
from datasets import load_from_disk
import polars as pl
import torch
from torch import Tensor


def main(
    embeddings_dir: Path = typer.Option(...),
):
    embs = load_from_disk(dataset_path=str(embeddings_dir))["train"]
    embs.set_format(type="torch", columns=["embedding"])

    logger.info("Converting embeddings to polars...")
    df = embs.to_polars()

    logger.info("Aggregating embeddings...")
    ids, embs = mean_average_embeddings(df)

    target_file = embeddings_dir.parent / "agg_embeddings.pt"
    logger.info(f"Saving aggregated embeddings at {target_file}")
    torch.save({id_: emb for id_, emb in zip(ids, embs)}, target_file)


def mean_average_embeddings(df: pl.DataFrame) -> tuple[list[str], Tensor]:
    """Aggregates embeddings by taking the mean of document's chunks weighted by their sizes.

    Args:
        df (pl.DataFrame): dataframe with columns ["_id", "chunk_len", "embeddings"]

    Returns:
        tuple[list[str], Tensor]: tuple of list of ids and aggregated embeddings
    """
    ids = []
    emb_dim = df["embedding"][0].shape[0]
    num_unique_docs = df.select(pl.col("_id")).n_unique()
    embs = torch.empty(num_unique_docs, emb_dim, dtype=torch.float)

    for i, (id_, x) in tqdm(enumerate(df.group_by(["_id"])), total=num_unique_docs):
        ids.append(id_[0])
        chunk_lengths = torch.tensor(np.stack(x["chunk_len"].to_numpy()))
        chunk_lengths = chunk_lengths / chunk_lengths.sum()
        chunk_embs = torch.tensor(np.stack(x["embedding"].to_numpy()))
        chunk_embs = chunk_embs * chunk_lengths[:, None]
        embs[i] = chunk_embs.sum(axis=0)

    return ids, embs


if __name__ == "__main__":
    typer.run(main)
