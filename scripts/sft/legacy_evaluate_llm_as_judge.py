import json
import logging
import os
from pathlib import Path
from pprint import pformat
from typing import Any, Literal

import hydra
import torch
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_openai import ChatOpenAI
from loguru import logger
from omegaconf import DictConfig
from pydantic import BaseModel

from juddges.evaluation.eval_structured_llm_judge import PROMPTS, StructuredLLMJudgeEvaluator
from juddges.evaluation.parse import parse_results
from juddges.settings import CONFIG_PATH
from juddges.utils.config import resolve_config

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "mocked_key"
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")


class ApiModel(BaseModel, extra="forbid"):
    name: str
    endpoint: str | None = OPENAI_ENDPOINT
    request_cache_db: Path | None


class LLMJudgeConfig(BaseModel, extra="forbid"):
    api_llm: ApiModel
    answers_file: Path
    out_metric_file: Path
    prompt: Literal["pl", "en"]


@torch.inference_mode()
@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="llm_judge.yaml")
def main(cfg: DictConfig) -> None:
    cfg_dict = resolve_config(cfg)
    logger.info(f"config:\n{pformat(cfg_dict)}")
    config = LLMJudgeConfig(**cfg_dict)

    results = evaluate_with_api_llm(config)

    with config.out_metric_file.open("w") as f:
        json.dump(results, f, indent="\t")


def evaluate_with_api_llm(config: LLMJudgeConfig) -> dict[str, Any]:
    client = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=config.api_llm.endpoint,
        model_name=config.api_llm.name,
        temperature=0.0,
    )

    if config.api_llm.request_cache_db is not None:
        set_llm_cache(SQLiteCache(str(config.api_llm.request_cache_db)))

    evaluator = StructuredLLMJudgeEvaluator(client=client, prompt=PROMPTS[config.prompt])

    with open(config.answers_file) as f:
        answers = json.load(f)
    preds, golds = parse_results(answers)

    return evaluator.evaluate(preds=preds, golds=golds)


if __name__ == "__main__":
    main()
