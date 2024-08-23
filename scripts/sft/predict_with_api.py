import asyncio
import json
import os
from pathlib import Path
from pprint import pformat

import hydra
import torch
from datasets import Dataset, load_dataset
from dotenv import dotenv_values
from loguru import logger
from omegaconf import DictConfig
from openai import APIConnectionError, AsyncOpenAI, BaseModel, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from juddges.config import DatasetConfig
from juddges.preprocessing.context_truncator import ContextTruncatorTiktoken
from juddges.preprocessing.text_encoder import TextEncoderForOpenAIEval
from juddges.settings import CONFIG_PATH
from juddges.utils.config import resolve_config

OPENAI_API_KEY = dotenv_values()["OPENAI_API_KEY"]
NUM_PROC = int(os.getenv("NUM_PROC", 1))
MAX_REQUEST_ATTEMPTS = 3
MIN_RETRY_WAIT = 1
MAX_RETRY_WAIT = 10


class PredictWithAPIConfig(BaseModel, extra="forbid"):
    dataset: DatasetConfig
    model_version: str
    max_seq_len: int | None
    temperature: float
    seed: int
    batch_size: int | None  # use batch size for systems with continuous batch optimizations
    output_file: Path


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="predict_with_api.yaml")
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    config = PredictWithAPIConfig(**resolve_config(cfg))
    logger.info(f"config:\n{pformat(config.model_dump())}")

    output_file = Path(config.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(config.dataset.name, split="test")

    gold_outputs = [item["output"] for item in ds]

    truncator = ContextTruncatorTiktoken(model=config.model_version, max_length=config.max_seq_len)
    encoder = TextEncoderForOpenAIEval(truncator=truncator)
    ds = ds.map(encoder, num_proc=NUM_PROC)

    predictor = OpenAIPredictor(config=config)
    model_outputs = predictor.run(ds)

    results = [
        {"answer": ans, "gold": gold_ans} for ans, gold_ans in zip(model_outputs, gold_outputs)
    ]
    with open(output_file, "w") as f:
        json.dump(results, f, indent="\t", ensure_ascii=False)


class OpenAIPredictor:
    def __init__(self, config: PredictWithAPIConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
        )

    def run(self, dataset: Dataset) -> list[str]:
        return asyncio.run(self._predict_dataset(dataset))

    async def _predict_dataset(self, dataset: Dataset) -> list[str]:
        if self.config.batch_size is None or (self.config.batch_size > dataset.num_rows):
            pred_tasks = [self._predict_item(item["final_input"]) for item in dataset]
            predictions = await tqdm_asyncio.gather(*pred_tasks)
            return predictions
        else:
            predictions = []
            for batch in tqdm(
                dataset.iter(batch_size=self.config.batch_size),
                total=len(dataset) // self.config.batch_size,
                desc=f"Predicting with {self.config.model_version}",
            ):
                llm_requests = [self._predict_item(item) for item in batch["final_input"]]
                predictions.extend(await asyncio.gather(*llm_requests))

        return predictions

    @retry(
        wait=wait_random_exponential(min=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
        stop=stop_after_attempt(MAX_REQUEST_ATTEMPTS),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    )
    async def _predict_item(self, item: str) -> str:
        """Requests OpenAI API LLM."""
        completion = await self.client.chat.completions.create(
            model=self.config.model_version,
            messages=[{"role": "user", "content": item}],
            temperature=self.config.temperature,
            seed=self.config.seed,
        )
        return completion.choices[0].message.content  # type: ignore[return-value]


if __name__ == "__main__":
    main()
