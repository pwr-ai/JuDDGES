import os
import pickle
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Literal

import hydra
import torch
import yaml
from datasets import Dataset, load_dataset
from loguru import logger
from omegaconf import DictConfig
from openai import BaseModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.utils import is_flash_attn_2_available

from juddges.config import EmbeddingModelConfig
from juddges.preprocessing.text_chunker import TextSplitter
from juddges.settings import CONFIG_PATH
from juddges.utils.config import resolve_config

assert is_flash_attn_2_available(), "FlashAttention2 is required for this script"

NUM_PROC = int(os.getenv("NUM_PROC", cpu_count() - 2))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false" if (NUM_PROC > 1) else "true"


class EmbeddingConfig(BaseModel, extra="forbid"):
    dataset_name: str
    embedding_model: EmbeddingModelConfig
    length_adjust_mode: Literal["truncate", "chunk"]
    truncation_tokens: int | None = None
    chunk_config: dict[str, Any] | None = None
    batch_size: int
    output_dir: Path


@torch.inference_mode()
@hydra.main(
    version_base="1.3", config_path=str(CONFIG_PATH), config_name="embedding.yaml"
)
def main(cfg: DictConfig) -> None:
    cfg_dict = resolve_config(cfg)
    logger.info(f"config:\n{cfg_dict}")
    config = EmbeddingConfig(**cfg_dict)

    assert (config.chunk_config is not None) ^ (config.truncation_tokens is not None)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        config.dataset_name,
        columns=["judgement_id", "full_text"],
    )["train"]

    ds = ds.rename_column("full_text", "text")

    # Add logging to inspect dataset items
    logger.info(f"Dataset columns: {ds.column_names}")
    sample_item = ds[0]
    logger.info(f"Sample item keys: {list(sample_item.keys())}")
    logger.info(f"Sample item content: {sample_item}")

    ds = ds.filter(lambda item: item["text"] is not None, num_proc=NUM_PROC)

    model = SentenceTransformer(
        config.embedding_model.name,
        device=DEVICE,
        model_kwargs=dict(torch_dtype=torch.bfloat16),
    )
    model.compile()

    if config.chunk_config is not None:
        logger.info("Chunking dataset")

        ds = chunk_dataset(dataset=ds, config=config, tokenizer=model.tokenizer)
        text_column = "chunk_text"
    else:
        text_column = "text"

    if config.truncation_tokens is not None:
        assert config.truncation_tokens <= config.embedding_model.max_seq_length
        model.max_seq_length = config.truncation_tokens

    # Start multi-GPU processing pool
    pool = model.start_multi_process_pool()

    logger.info(
        f"Embedding dataset with {config.batch_size} batch size using multiple GPUs"
    )
    texts = ds[text_column]

    # Calculate total number of batches
    total_batches = (len(texts) + config.batch_size - 1) // config.batch_size

    # Process embeddings with progress bar
    embeddings = []
    for i in tqdm(
        range(0, len(texts), config.batch_size),
        total=total_batches,
        desc="Calculating embeddings",
    ):
        batch_texts = texts[i : i + config.batch_size]
        batch_embeddings = model.encode_multi_process(
            batch_texts, pool, batch_size=config.batch_size
        )
        embeddings.extend(batch_embeddings)

    # save embeddings to disk
    with open(config.output_dir / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    ds = ds.add_column("embedding", embeddings)

    # Stop the multi-GPU pool
    model.stop_multi_process_pool(pool)

    logger.info(f"Saving dataset to {config.output_dir}")
    ds.save_to_disk(str(config.output_dir))
    logger.info(f"Dataset saved to {config.output_dir}")

    with open(config.output_dir / "config.yaml", "w") as f:
        yaml.dump(config.model_dump(), f)


def chunk_dataset(
    dataset: Dataset,
    config: EmbeddingConfig,
    tokenizer: PreTrainedTokenizer | None = None,
) -> Dataset:
    # todo: To be verified
    assert config.chunk_config is not None
    split_worker = TextSplitter(**config.chunk_config, tokenizer=tokenizer)
    logger.info(f"Chunking dataset with {config.chunk_config}")
    logger.info(f"Dataset columns: {dataset.column_names}")
    ds = dataset.select_columns(["judgement_id", "text"]).map(
        split_worker,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=["judgement_id", "text"],
        desc="Chunking documents",
    )
    ds.save_to_disk(str(config.output_dir / "chunked"))
    logger.info(f"Dataset split into {ds.num_rows} chunks")
    return ds


if __name__ == "__main__":
    main()
