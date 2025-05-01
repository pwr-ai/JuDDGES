import json
from pathlib import Path
from typing import Any, Literal

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
    use_unsloth: bool = False


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
    prompt: PromptInfoExtractionConfig
    ie_schema: dict[str, dict[str, Any]]
    device_map: str
    output_dir: Path
    truncate_context: bool
    generate_kwargs: dict[str, Any] = Field(default_factory=dict)
    random_seed: int

    @property
    def predictions_file(self) -> Path:
        return self.output_dir / "predictions.jsonl"

    @property
    def dataset_file(self) -> Path:
        """Path to the file with final inputs."""
        return self.output_dir / "dataset.json"

    @property
    def config_file(self) -> Path:
        """Path to the file with config."""
        return self.output_dir / "config.yaml"

    def get_max_input_length_accounting_for_output(self, max_position_embeddings: int) -> int:
        return max_position_embeddings - self.dataset.max_output_tokens
