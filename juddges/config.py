from pathlib import Path
from pydantic import BaseModel


class LLMConfig(BaseModel, extra="forbid"):
    name: str
    tokenizer_name: str
    adapter_path: Path | None
    max_seq_length: int
    padding: str | bool
    batch_size: int
    use_unsloth: bool = False


class DatasetConfig(BaseModel, extra="forbid"):
    name: str
    prompt_field: str
    context_field: str
    output_field: str
