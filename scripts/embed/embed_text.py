import os
from multiprocessing import cpu_count
from pathlib import Path
from pprint import pformat
from typing import Any

import hydra
import psutil
import torch
from datasets import Dataset, load_dataset
from loguru import logger
from omegaconf import DictConfig
from openai import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizer
from transformers.utils import is_flash_attn_2_available

from juddges.config import EmbeddingModelConfig
from juddges.preprocessing.text_chunker import TextSplitter
from juddges.settings import CONFIG_PATH
from juddges.utils.config import resolve_config
from juddges.utils.misc import save_dataset_as_parquet_shards

ID_COL: str = "judgment_id"
TEXT_COL: str = "full_text"

NUM_PROC = int(os.getenv("NUM_PROC", cpu_count() - 2))
NUM_GPUS = min(torch.cuda.device_count(), NUM_PROC)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false" if (NUM_PROC > 1) else "true"

if DEVICE == "cuda":
    assert is_flash_attn_2_available(), "FlashAttention2 is required for this script"


class EmbeddingConfig(BaseModel, extra="forbid"):
    CHUNK_EMBEDDINGS_DIR: str = "chunk_embeddings"
    AGG_EMBEDDINGS_DIR: str = "agg_embeddings"

    embeddings_root_dir: Path
    dataset_name: str
    embedding_model: EmbeddingModelConfig
    chunk_config: dict[str, Any] = None
    batch_size: int
    num_output_shards: int

    @property
    def output_dir(self) -> str:
        return self.embeddings_root_dir / self.dataset_name.replace("/", "__")

    @property
    def chunk_embeddings_dir(self) -> Path:
        return self.output_dir / self.CHUNK_EMBEDDINGS_DIR

    @property
    def agg_embeddings_dir(self) -> Path:
        return self.output_dir / self.AGG_EMBEDDINGS_DIR


@torch.inference_mode()
@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="embedding.yaml")
def main(cfg: DictConfig) -> None:
    cfg_dict = resolve_config(cfg)
    logger.info(f"config:\n{pformat(cfg_dict)}")
    config = EmbeddingConfig(**cfg_dict)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        config.dataset_name,
        columns=[ID_COL, TEXT_COL],
    )["train"]

    logger.info(f"Dataset columns: {ds.column_names}")
    sample_item = ds[0]
    logger.info(f"Sample item keys: {list(sample_item.keys())}")
    logger.info(f"Sample item content: {sample_item}")

    ds = ds.select(range(1_000))
    ds = ds.filter(lambda item: item[TEXT_COL] is not None, num_proc=NUM_PROC)
    model = SentenceTransformer(
        config.embedding_model.name,
        device=DEVICE,
        model_kwargs=dict(torch_dtype=torch.bfloat16),
    )
    model.compile()

    logger.info("Chunking dataset")
    chunk_ds = chunk_dataset(dataset=ds, config=config, tokenizer=model.tokenizer)

    embedder = Embedder(model=model, column_to_embed=TextSplitter.CHUNK_TEXT_COL)
    chunk_ds = chunk_ds.map(
        embedder,
        batched=True,
        batch_size=config.batch_size,
        num_proc=1,
    )
    save_dataset_as_parquet_shards(chunk_ds, config.num_output_shards, config.chunk_embeddings_dir)


def chunk_dataset(
    dataset: Dataset,
    config: EmbeddingConfig,
    tokenizer: PreTrainedTokenizer | None = None,
) -> Dataset:
    assert config.chunk_config is not None
    split_worker = TextSplitter(
        id_col=ID_COL,
        text_col=TEXT_COL,
        **config.chunk_config,
        tokenizer=tokenizer,
    )
    logger.info(f"Chunking dataset with {config.chunk_config}")
    logger.info(f"Dataset columns: {dataset.column_names}")
    chunk_ds = dataset.select_columns([ID_COL, TEXT_COL]).map(
        split_worker,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=[ID_COL, TEXT_COL],
        desc="Chunking documents",
    )
    logger.info(
        f"Dataset with {dataset.num_rows} judgments split into {chunk_ds.num_rows} chunks with columns: {chunk_ds.column_names}"
    )
    return chunk_ds


class Embedder:
    def __init__(self, model: SentenceTransformer, column_to_embed: str) -> None:
        self.model = model
        self.column_to_embed = column_to_embed

    def __call__(self, items: dict[str, list[Any]]) -> dict[str, Any]:
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
        logger.info(f"Initial memory usage: {initial_memory:.2f} GB")

        pool: dict[str, Any] | None = None
        try:
            # It'll take processes equal to NUM_GPUS or 4 in case of CPU
            pool = self.model.start_multi_process_pool()
            embeddings = self.model.encode_multi_process(
                sentences=items[self.column_to_embed],
                pool=pool,
                batch_size=len(items[self.column_to_embed]),
            )
            final_memory = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
            logger.info(f"Final memory usage: {final_memory:.2f} GB")
            logger.info(f"Memory difference: {final_memory - initial_memory:.2f} GB")
        except Exception as e:
            logger.error(f"Error embedding dataset: {e}")
            raise e
        else:
            return {
                "embedding": embeddings,
            }
        finally:
            if pool is not None:
                self.model.stop_multi_process_pool(pool)


if __name__ == "__main__":
    main()
