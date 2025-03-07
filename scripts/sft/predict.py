import json
import os
from pathlib import Path
from pprint import pformat
from typing import Any

import hydra
import torch
from datasets import load_dataset
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
from juddges.utils.misc import sort_dataset_by_input_length

torch.set_float32_matmul_precision("high")
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

    @property
    def corrected_max_seq_length(self) -> int:
        return self.model.max_seq_length - self.dataset.max_output_tokens


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="predict.yaml")
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    """Performs inference on a given dataset using given model.
    The outputs are saved to a file in the following JSONL format:
    [
        {
            "answer": str,
            "gold": str
        },
        ...
    ]
    """
    config = PredictConfig(**resolve_config(cfg))
    logger.info(f"config:\n{pformat(config.model_dump())}")

    output_file = Path(config.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(config.dataset.name, split="test")
    ds, reverse_sort_idx = sort_dataset_by_input_length(ds, config.dataset.context_field)
    logger.info("Loading model...")

    model_pack = get_model(config.model, device_map=config.device_map)

    assert not any(key in model_pack.generate_kwargs for key in config.generate_kwargs.keys())
    model_pack.generate_kwargs |= config.generate_kwargs

    if config.model.use_unsloth:
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model_pack.model)
    else:
        model_pack.model.eval()
        # model_pack.model.compile() # might cause libcuda.so not found error

    if config.model.batch_size > 1 and config.model.padding is False:
        raise ValueError("Padding has to be enabled if batch size > 1.")

    gold_outputs = [item["output"] for item in ds]

    encoder = TextEncoderForEval(
        tokenizer=model_pack.tokenizer,
        max_length=config.corrected_max_seq_length,
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
    results = [results[i] for i in reverse_sort_idx]

    with open(output_file, "w") as f:
        json.dump(results, f, indent="\t", ensure_ascii=False)


if __name__ == "__main__":
    main()
