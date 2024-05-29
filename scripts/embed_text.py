import json
import os
from pathlib import Path
from datasets import Dataset
from typing import Any, Literal
import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from openai import BaseModel
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from juddges.config import EmbeddingModelConfig, RawDatasetConfig
from juddges.defaults import CONFIG_PATH
from juddges.preprocessing.text_chunker import TextSplitter

NUM_PROC = int(os.getenv("NUM_PROC", 1))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingConfig(BaseModel, extra="forbid"):
    dataset: RawDatasetConfig
    embedding_model: EmbeddingModelConfig
    length_adjust_mode: Literal["truncate", "chunk"]
    truncation_tokens: int | None = None
    chunk_config: dict[str, Any] | None = None
    batch_size: int
    output_dir: Path


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="embedding.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    logger.info(f"config:\n{cfg}")
    config = EmbeddingConfig(**cfg)

    assert (config.chunk_config is not None) ^ (config.truncation_tokens is not None)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        config.dataset.format,
        data_dir=str(config.dataset.root_dir),
        columns=["_id", "text"],
    )
    ds = ds.filter(lambda item: item["text"] is not None)

    if config.chunk_config is not None:
        ds = chunk_dataset(ds, config)
        text_column = "text_chunk"
    else:
        text_column = "text"

    model = SentenceTransformer(config.embedding_model.name).to(DEVICE)

    if config.truncation_tokens is not None:
        assert config.truncation_tokens <= config.embedding_model.max_seq_length
        model.max_seq_length = config.truncation_tokens

    embedder = Embedder(model, text_column)
    ds = ds.map(
        embedder,
        batched=True,
        batch_size=config.batch_size,
        num_proc=None,
        remove_columns=[text_column],
    )
    ds.save_to_disk(config.dataset.output_dir)

    with open(config.output_dir / "config.json", "w") as f:
        json.dump(config.model_dump(), f)


def chunk_dataset(dataset: Dataset, config: EmbeddingConfig) -> Dataset:
    # todo: To be verified
    split_worker = TextSplitter(**config.chunk_config)
    ds = dataset.select_columns(["_id", "text"]).map(
        split_worker,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=["_id", "text"],
    )
    logger.info(f"Dataset split into {ds.num_rows} chunks")


class Embedder:
    def __init__(self, model: SentenceTransformer, text_column: str) -> None:
        self.model = model
        self.text_column = text_column

    def __call__(self, items: dict[str, Any]) -> dict[str, Any]:
        return {"embeddings": self.model.encode(items[self.text_column])}


if __name__ == "__main__":
    main()
