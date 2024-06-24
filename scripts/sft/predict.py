import json
import os
import time
from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from loguru import logger
from omegaconf import DictConfig
from openai import BaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from juddges.config import DatasetConfig, LLMConfig
from juddges.models.factory import get_model
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
    metrics_file: Path
    max_new_tokens: int
    truncate_context: bool


@torch.inference_mode()
@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    cfg_dict = resolve_config(cfg)
    logger.info(f"config:\n{cfg_dict}")
    config = PredictConfig(**cfg_dict)

    output_file = Path(config.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(config.dataset.name, split="test")
    logger.info("Loading model...")

    model_pack = get_model(config.model, device_map=config.device_map)
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
    dataloader = DataLoader(
        ds,
        batch_size=config.model.batch_size,
        num_workers=NUM_PROC,
        pin_memory=(NUM_PROC > 1),
        shuffle=False,
    )

    model_outputs = []

    with tqdm(dataloader) as pbar:
        for batch in pbar:
            model_inputs = batch["input_ids"].view(config.model.batch_size, -1)
            model_inputs = model_inputs.to(DEVICE, non_blocking=True)
            input_length = model_inputs.size(1)

            start_time = time.time()
            generated_ids = model.generate(
                model_inputs,
                max_new_tokens=config.max_new_tokens,
                **model_pack.generate_kwargs,
            )
            duration = time.time() - start_time

            decoded = tokenizer.batch_decode(
                generated_ids[:, input_length:],
                skip_special_tokens=True,
            )
            model_outputs.extend(decoded)

            pbar.set_postfix_str(f"{generated_ids.numel() / duration: 0.2f} tok/sec")

    results = [
        {"answer": ans, "gold": gold_ans} for ans, gold_ans in zip(model_outputs, gold_outputs)
    ]

    with open(output_file, "w") as f:
        json.dump(results, f, indent="\t")


if __name__ == "__main__":
    main()
