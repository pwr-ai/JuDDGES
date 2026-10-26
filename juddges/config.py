import json
import re
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field


class PromptInfoExtractionConfig(BaseModel, extra="forbid"):
    """Configuration class for prompt."""

    language: Literal["pl", "en"]
    ie_schema: dict[str, dict[str, Any]]
    content: str

    def render(self, context: str) -> str:
        schema_str = json.dumps(self.ie_schema, indent=2)
        return self.content.format(
            language=self.language,
            schema=schema_str,
            context=context,
        ).strip()


class LLMConfig(BaseModel, extra="forbid"):
    """Configuration class for autoregressive LLM."""

    name: str
    tokenizer_name: str
    adapter_path: Path | None
    padding_side: str = "left"
    padding: str | bool
    batch_size: int
    use_4bit: bool
    # None keeps the bitsandbytes default (fp4); the 24gb profiles set "nf4" (QLoRA, vLLM in-flight)
    bnb_4bit_quant_type: str | None = None
    use_unsloth: bool = False
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def should_load_adapter(self) -> bool:
        return self.adapter_path is not None

    @cached_property
    def adapter_path_or_first_ckpt_path(self) -> Path:
        assert self.should_load_adapter, "Adapter path is not set"

        if (self.adapter_path / "adapter_model.safetensors").exists():
            return self.adapter_path

        checkpoints = {
            int(match.group(1)): path
            for path in self.adapter_path.glob("checkpoint-*")
            if (match := re.fullmatch(r"checkpoint-(\d+)", path.name))
        }
        if not checkpoints:
            raise ValueError(
                f"No adapter_model.safetensors or checkpoint dir found in {self.adapter_path}"
            )

        last_checkpoint_adapter_path = checkpoints[max(checkpoints)]
        logger.warning(
            "adapter_path was set to checkpoints dir, using LAST checkpoint "
            f"(set specific checkpoint path as adapter_path to use other checkpoint): {last_checkpoint_adapter_path}"
        )
        return last_checkpoint_adapter_path

    @property
    def bnb_4bit_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"load_in_4bit": True}
        if self.bnb_4bit_quant_type is not None:
            kwargs["bnb_4bit_quant_type"] = self.bnb_4bit_quant_type
        return kwargs


class EmbeddingModelConfig(BaseModel, extra="forbid"):
    """Configuration class for embedding model."""

    name: str
    max_seq_length: int


class DatasetInfoExtractionConfig(BaseModel, extra="forbid"):
    """Configuration class for instructions dataset for information extraction."""

    name: str
    language: Literal["pl", "en"]
    prompt_field: str | None = Field(
        default=None,
        deprecated=True,
        desc="Legacy, prompt now is defined outside",
    )
    language: Literal["pl", "en"]
    prompt_field: str | None = Field(
        default=None,
        deprecated=True,
        desc="Legacy, prompt now is defined outside",
    )
    context_field: str
    output_field: str
    max_output_tokens: int


class RawDatasetConfig(BaseModel, extra="forbid"):
    """Configuration class for raw dataset (parquet files)."""

    name: str
    format: str
    root_dir: Path


class FineTuningConfig(BaseModel, extra="forbid"):
    llm: LLMConfig
    dataset: DatasetInfoExtractionConfig
    prompt: PromptInfoExtractionConfig
    ie_schema: dict[str, dict[str, Any]]
    llm: LLMConfig
    dataset: DatasetInfoExtractionConfig
    prompt: PromptInfoExtractionConfig
    ie_schema: dict[str, dict[str, Any]]
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


class PredictInfoExtractionConfig(BaseModel, extra="forbid"):
    llm: LLMConfig
    dataset: DatasetInfoExtractionConfig
    split: str
    prompt: PromptInfoExtractionConfig
    ie_schema: dict[str, dict[str, Any]]
    device_map: str
    output_dir: Path
    max_model_len: int | None = None
    truncate_context: bool
    generate_kwargs: dict[str, Any] = Field(default_factory=dict)
    random_seed: int
    parallel: Literal["tensor", "pipeline"] | None = None
    guided_json: bool = False  # constrain decoding to the JSON schema built from ie_schema
    quantization: str | None = None  # vLLM in-flight quantization, e.g. "bitsandbytes"
    gpu_memory_utilization: float = 0.9  # vLLM default
    max_lora_rank: int = 16  # vLLM default is 16; must be >= peft r of the adapter
    max_docs: int | None = None  # use only the first max_docs documents (smoke tests)

    @property
    def predictions_file(self) -> Path:
        return self.output_dir / "predictions.json"

    @property
    def dataset_file(self) -> Path:
        """Path to the file with final inputs."""
        return self.output_dir / "dataset.json"

    @property
    def config_file(self) -> Path:
        """Path to the file with config."""
        return self.output_dir / "config.yaml"

    def get_max_input_length_accounting_for_output(
        self,
        max_position_embeddings: int | None,
    ) -> int:
        if max_position_embeddings is None:
            assert self.max_model_len is not None
            max_position_embeddings = self.max_model_len

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
    ingest_batch_size: int = 32
    upsert: bool = True
    default_column_values: dict[str, Any] | None = None

    @property
    def chunk_embeddings_dir(self) -> Path:
        return self.output_dir / self.CHUNK_EMBEDDINGS_DIR

    @property
    def agg_embeddings_dir(self) -> Path:
        return self.output_dir / self.AGG_EMBEDDINGS_DIR
