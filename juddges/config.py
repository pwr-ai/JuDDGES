from pathlib import Path
from pydantic import BaseModel


class LLMConfig(BaseModel, extra="forbid"):
    _org_name: str
    _model_name: str
    model_name: str
    tokenizer_name: str
    adapter_path: Path | None
    max_seq_length: int
    padding: str | bool
    batch_size: int


class DatasetConfig(BaseModel, extra="forbid"):
    _org_name: str
    _dataset_name: str
    dataset_name: str
    prompt_field: str
    context_field: str
    output_field: str
