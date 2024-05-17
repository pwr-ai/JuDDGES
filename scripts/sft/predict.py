import json
import math
from pathlib import Path
import time

import hydra
from datasets import load_dataset

from loguru import logger
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from juddges.defaults import ROOT_PATH
from juddges.models.factory import get_model
from juddges.preprocessing.text_encoder import EvalEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# LLM = "meta-llama/Meta-Llama-3-8B-Instruct"
# BATCH_SIZE = 1  # for now llama-3 doesn't work with padding, hence with batch_size > 1 also
# MAX_NEW_TOKENS = 250
# MAX_LENGTH = 2_048


@torch.no_grad()
@hydra.main(
    version_base="1.3", config_path=str(ROOT_PATH / "configs/predict"), config_name="config.yaml"
)
def main(cfg: DictConfig) -> None:
    logger.info(f"config:\n{cfg}")
    device = cfg.device if cfg.device else DEVICE

    output_file = Path(cfg.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("JuDDGES/pl-court-instruct")
    logger.info("Loading model...")

    model_pack = get_model(cfg.model.model_name, device_map=cfg.device_map)
    model, tokenizer = model_pack.model, model_pack.tokenizer
    if cfg.model.batch_size > 1 and cfg.model.padding is False:
        raise ValueError("Padding has to be enabled if batch size > 1.")
    encoder = EvalEncoder(
        tokenizer=tokenizer,
        max_length=cfg.model.max_seq_length,
        padding=cfg.model.padding,
    )
    encoded_ds = ds["test"].map(
        encoder,
        batched=False,
        num_proc=cfg.num_proc,
    )
    encoded_ds.set_format("torch")
    results = []
    num_batches = math.ceil(encoded_ds.num_rows / cfg.model.batch_size)

    with tqdm(encoded_ds.iter(batch_size=cfg.model.batch_size), total=num_batches) as pbar:
        for item in pbar:
            model_inputs = item["tokens"].view(cfg.model.batch_size, -1).to(device)
            input_length = model_inputs.size(1)

            start_time = time.time()
            generated_ids = model.generate(
                model_inputs,
                max_new_tokens=cfg.max_new_tokens,
                **model_pack.generate_kwargs,
            )
            duration = time.time() - start_time

            decoded = tokenizer.batch_decode(
                generated_ids[:, input_length:], skip_special_tokens=True
            )
            results.extend(
                [{"answer": ans, "gold": gold} for ans, gold in zip(decoded, item["output"])]
            )

            pbar.set_postfix_str(f"{generated_ids.numel() / duration: 0.2f} tok/sec")

    with open(output_file, "w") as f:
        json.dump(results, f, indent="\t")

    # res = evaluate_extraction(results)
    # with open(cfg.metrics_file, "w") as file:
    #     json.dump(res, file, indent="\t")


if __name__ == "__main__":
    main()
