import json
import os
from pathlib import Path
from pprint import pformat
from typing import Any

import hydra
import torch
from accelerate import PartialState
from datasets import load_dataset
from loguru import logger
from omegaconf import DictConfig
from pydantic import BaseModel, Field
from torch import Tensor
from transformers import PreTrainedTokenizer

from juddges.config import LLMConfig
from juddges.models.factory import get_model
from juddges.models.predict import predict_with_llm
from juddges.settings import CONFIG_PATH
from juddges.utils.config import resolve_config

NUM_PROC = int(os.getenv("NUM_PROC", 1))
if NUM_PROC > 1:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

JUDGE_PROMPT = """
You are evaluating information extraction system by comparing a submitted answer to an expert answer on a given question.
Data is in Polish. Here is the data:
[BEGIN DATA]
************
[Expert]: {gold}
************
[Submission]: {answer}
************
[END DATA]

Submitted answer should be formatted as YAML. If the submitted answer cannot be parsed as YAML, return incorrect.
When comparing consecutive fields, ignore order of fields, capitalization and don't be sensitive to abbreviations which preserves the meaning of the answer.
In comparison, ignore legal_bases field.
Format you answer as follows: The answer is <correct/incorrect>. Don't provide any additional explanation.
"""


class LLMJudgeConfig(BaseModel, extra="forbid"):
    model: LLMConfig
    answers_file: Path
    out_metric_file: Path
    out_predictions_file: Path
    generate_kwargs: dict[str, Any] = Field(default_factory=dict)


@torch.inference_mode()
@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="llm_judge.yaml")
def main(cfg: DictConfig) -> None:
    cfg_dict = resolve_config(cfg)
    logger.info(f"config:\n{pformat(cfg_dict)}")
    config = LLMJudgeConfig(**cfg_dict)

    config.out_metric_file.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("json", data_files=str(config.answers_file), split="train")
    ds = ds.map(
        lambda x: {"input_text": JUDGE_PROMPT.format(answer=x["answer"], gold=x["gold"])},
    )
    ds.cleanup_cache_files()

    model_pack = get_model(
        llm_config=config.model,
        device_map={"": PartialState().process_index},
    )
    model_pack.generate_kwargs |= config.generate_kwargs

    encoder = SimpleEncoder(tokenizer=model_pack.tokenizer)
    ds.set_transform(encoder, columns=["input_text"])

    predictions = predict_with_llm(
        model_pack=model_pack,
        dataset=ds,
        batch_size=config.model.batch_size,
        num_proc=NUM_PROC,
        verbose=True,
    )

    with open(config.out_predictions_file, "w") as f:
        json.dump(predictions, f, indent="\t")


class SimpleEncoder:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: dict[str, list[str]]) -> dict[str, Tensor]:
        # NOTE: truncation is disabled and padding is set to "longest"
        input_texts = []
        for text in batch["input_text"]:
            input_chat = [{"role": "user", "content": text}]
            final_input = self.tokenizer.apply_chat_template(
                input_chat,
                add_generation_prompt=True,
                tokenize=False,
            )
            input_texts.append(final_input)

        return self.tokenizer(
            input_texts,
            padding="longest",
            truncation=False,
            return_tensors="pt",
            return_attention_mask=False,
            return_special_tokens_mask=False,
        )


if __name__ == "__main__":
    main()
