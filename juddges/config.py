from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel, extra="forbid"):
    """Configuration class for autoregressive LLM."""

    name: str
    tokenizer_name: str
    adapter_path: Path | None
    padding_side: str = "left"
    padding: str | bool
    batch_size: int
    use_4bit: bool
    use_unsloth: bool = False


class EmbeddingModelConfig(BaseModel, extra="forbid"):
    """Configuration class for embedding model."""

    name: str
    max_seq_length: int


class DatasetConfig(BaseModel, extra="forbid"):
    """Configuration class for instructions dataset."""

    name: str
    prompt_field: str
    context_field: str
    output_field: str
    max_output_tokens: int


class RawDatasetConfig(BaseModel, extra="forbid"):
    """Configuration class for raw dataset (parquet files)."""

    name: str
    format: str
    root_dir: Path


class FineTuningConfig(BaseModel, extra="forbid"):
    model: LLMConfig
    dataset: DatasetConfig
    max_context_size: int
    training_args: dict[str, Any]
    peft_args: dict[str, Any] | None
    truncate_context: bool
    wandb_entity: str
    wandb_project: str
    output_dir: Path
    run_name: str

    @property
    def use_peft(self) -> bool:
        return self.peft_args is not None


class PredictConfig(BaseModel, extra="forbid"):
    model: LLMConfig
    dataset: DatasetConfig
    device_map: str
    output_file: Path
    truncate_context: bool
    generate_kwargs: dict[str, Any] = Field(default_factory=dict)
    random_seed: int

    def get_max_input_length(self, max_position_embeddings: int) -> int:
        return max_position_embeddings - self.dataset.max_output_tokens


class EmbeddingConfig(BaseModel, extra="forbid"):
    CHUNK_EMBEDDINGS_DIR: str = "chunk_embeddings"
    AGG_EMBEDDINGS_DIR: str = "agg_embeddings"

    output_dir: Path
    dataset_name: str
    embedding_model: EmbeddingModelConfig
    chunk_config: dict[str, Any] = None
    batch_size: int
    num_output_shards: int
    max_documents: Optional[int] = None
    ingest_batch_size: int = 32
    upsert: bool = True
    default_column_values: Optional[dict[str, Any]] = None

    @property
    def chunk_embeddings_dir(self) -> Path:
        return self.output_dir / self.CHUNK_EMBEDDINGS_DIR

    @property
    def agg_embeddings_dir(self) -> Path:
        return self.output_dir / self.AGG_EMBEDDINGS_DIR
