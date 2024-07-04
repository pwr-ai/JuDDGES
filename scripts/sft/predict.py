import json
import os
from pathlib import Path
from pprint import pformat
from typing import Any

import hydra
import torch
from datasets import Dataset, load_dataset
from lightning_fabric import seed_everything
from loguru import logger
from omegaconf import DictConfig
from openai import BaseModel
from pydantic import Field

from juddges.config import DatasetConfig, LLMConfig
from juddges.models.factory import get_model
from juddges.models.predict import predict_with_llm
from juddges.preprocessing.text_encoder import TextEncoderForEval
from juddges.settings import CONFIG_PATH
from juddges.utils.config import resolve_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PROC = int(os.getenv("NUM_PROC", 1))

if NUM_PROC > 1:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PredictConfig(BaseModel, extra="forbid"):
    model: LLMConfig
    dataset: DatasetConfig
    device_map: str
    output_file: Path
    truncate_context: bool
    generate_kwargs: dict[str, Any] = Field(default_factory=dict)
    random_seed: int


@torch.inference_mode()
@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    cfg_dict = resolve_config(cfg)
    logger.info(f"config:\n{pformat(cfg_dict)}")
    config = PredictConfig(**cfg_dict)

    output_file = Path(config.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(config.dataset.name, split="test")
    ds, _ = sort_dataset_by_text_length(ds, config.dataset.context_field)
    logger.info("Loading model...")

    model_pack = get_model(config.model, device_map=config.device_map)

    assert not any(key in model_pack.generate_kwargs for key in config.generate_kwargs.keys())
    model_pack.generate_kwargs |= config.generate_kwargs

    model, tokenizer = model_pack.model, model_pack.tokenizer
    model.eval()
    if config.model.batch_size > 1 and config.model.padding is False:
        raise ValueError("Padding has to be enabled if batch size > 1.")

    gold_outputs = [item["output"] for item in ds]

    encoder = TextEncoderForEval(
        tokenizer=tokenizer,
        max_length=config.model.max_seq_length,
        padding=config.model.padding,
    )
    ds.set_transform(encoder, columns=["prompt", "context"])

    seed_everything(config.random_seed)
    model_outputs = predict_with_llm(
        model_pack=model_pack,
        dataset=ds,
        batch_size=config.model.batch_size,
        num_proc=NUM_PROC,
        verbose=True,
    )
    results = [
        {"answer": ans, "gold": gold_ans} for ans, gold_ans in zip(model_outputs, gold_outputs)
    ]

    with open(output_file, "w") as f:
        json.dump(results, f, indent="\t", ensure_ascii=False)


def sort_dataset_by_text_length(ds: Dataset, field: str) -> tuple[Dataset, torch.Tensor]:
    lenghts = torch.tensor(
        ds.map(lambda x: {"length": len(x[field])}, desc="Computing length")["length"]
    )
    sort_idx = torch.argsort(lenghts, descending=True, stable=True)
    ds = ds.select(sort_idx)
    return ds, sort_idx


if __name__ == "__main__":
    main()
