import json
import os
from pathlib import Path
from pprint import pformat
from typing import Any, Literal

import hydra
import requests
import torch
from deepdiff import DeepDiff
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from loguru import logger
from omegaconf import DictConfig
from openai import OpenAI
from pydantic import BaseModel, Field

from juddges.evaluation.llm_evaluator import StructuredLLMJudgeEvaluator
from juddges.settings import CONFIG_PATH
from juddges.utils.config import resolve_config

LLAMA_CPP_ENDOPOINT = os.getenv("LLAMA_CPP_ENDPOINT", "http://localhost:8000")

load_dotenv()


class ApiModel(BaseModel, extra="forbid"):
    name: str
    api: Literal["llama.cpp", "openai"]
    endpoint: str = Field(default=LLAMA_CPP_ENDOPOINT)
    use_langsmith: bool
    config: dict[str, Any] | None = None


class LLMJudgeConfig(BaseModel, extra="forbid"):
    api_model: ApiModel
    answers_file: Path
    out_metric_file: Path


@torch.inference_mode()
@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="llm_judge.yaml")
def main(cfg: DictConfig) -> None:
    cfg_dict = resolve_config(cfg)
    logger.info(f"config:\n{pformat(cfg_dict)}")
    config = LLMJudgeConfig(**cfg_dict)

    if config.api_model.api == "llama.cpp":
        results = evaluate_with_llama_cpp(config)
    elif config.api_model.api == "openai":
        raise NotImplementedError("OpenAI API is not implemented yet")
    else:
        raise ValueError(f"Unknown API: {config.api_model.api}")

    with open(config.out_metric_file, "w") as f:
        json.dump(results, f, indent="\t")


def evaluate_with_llama_cpp(config: LLMJudgeConfig) -> dict[str, Any]:
    _check_llama_cpp_integrity(config)

    oai_client = OpenAI(
        api_key="not-required",
        base_url=config.api_model.endpoint,
    )
    if config.api_model.use_langsmith:
        oai_client = wrap_openai(oai_client)

    evaluator = StructuredLLMJudgeEvaluator(
        oai_client=oai_client,
        model_name=config.api_model.name,
    )
    with open(config.answers_file) as f:
        answers = json.load(f)

    return evaluator.evaluate(answers)


def _check_llama_cpp_integrity(config: LLMJudgeConfig) -> None:
    props = _get_llama_props(config.api_model.endpoint)
    diff = DeepDiff(config.api_model.config, props, ignore_order=True)
    if diff:
        raise ValueError(f"Config mismatch:\n{pformat(diff)}")


def _get_llama_props(endpoint: str) -> dict[str, Any]:
    return requests.get(f"{endpoint}/props").json()


if __name__ == "__main__":
    main()
