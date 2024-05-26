from pathlib import Path
from pydantic import BaseModel


class LLMConfig(BaseModel, extra="forbid"):
    org_name: str
    name: str
    llm: str
    tokenizer_name: str
    adapter_path: Path | None
    max_seq_length: int
    padding: str | bool
    batch_size: int


class DatasetConfig(BaseModel, extra="forbid"):
    org_name: str
    name: str
    dataset_name: str
    prompt_field: str
    context_field: str
    output_field: str
