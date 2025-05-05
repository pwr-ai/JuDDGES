import json
import os
from pathlib import Path
from pprint import pformat

import hydra
import torch
from datasets import load_dataset
from loguru import logger
from omegaconf import DictConfig
from vllm import LLM, SamplingParams

from juddges.config import PredictInfoExtractionConfig
from juddges.preprocessing.context_truncator import ContextTruncator
from juddges.preprocessing.text_encoder import TextEncoderForEvalPlainTextFormat
from juddges.settings import CONFIG_PATH
from juddges.utils.config import resolve_config

torch.set_float32_matmul_precision("high")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PROC = int(os.getenv("NUM_PROC", 1))


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="predict.yaml")
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    config = PredictInfoExtractionConfig(**resolve_config(cfg))
    logger.info(f"config:\n{pformat(config.model_dump())}")

    output_file = Path(config.output_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(config.dataset.name, split="test")

    llm = LLM(
        model=config.llm.name,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_lora=True,
        qlora_adapter_name_or_path=config.llm.adapter_path_or_last_ckpt_path,
        max_model_len=config.llm.max_seq_length,
        max_num_seqs=config.llm.batch_size,
    )

    truncator = ContextTruncator(
        tokenizer=llm.llm_engine.tokenizer.get_lora_tokenizer(),
        max_length=config.corrected_max_seq_length,
    )
    encoder = TextEncoderForEvalPlainTextFormat(truncator=truncator)
    ds = ds.map(encoder, num_proc=NUM_PROC)

    params = SamplingParams(
        max_tokens=config.generate_kwargs.get("max_new_tokens", 100),
        temperature=config.generate_kwargs.get("temperature", 0.0),
    )

    outputs = llm.generate(
        prompts=ds["final_input"],
        sampling_params=params,
    )
    results = [{"answer": ans, "gold": gold} for ans, gold in zip(outputs, ds["output"])]

    with open(output_file, "w") as f:
        json.dump(results, f, indent="\t", ensure_ascii=False)


if __name__ == "__main__":
    main()
