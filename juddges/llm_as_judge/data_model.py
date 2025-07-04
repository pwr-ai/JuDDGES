import json
from asyncio.log import logger
from functools import cached_property
from pathlib import Path
from statistics import mean
from typing import Any

from langchain_core.utils.json import parse_json_markdown
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from juddges.config import PredictInfoExtractionConfig
from juddges.utils.config import load_and_resolve_config


class ParsedPredictions(BaseModel):
    predictions: dict[int, dict[str, Any]]
    gold: dict[int, dict[str, Any]]
    errors: dict[int, str]
    missing_keys: dict[int, list[str]] = Field(default_factory=dict)
    extra_keys: dict[int, list[str]] = Field(default_factory=dict)
    num_items: int

    def get_stats(self) -> dict[str, Any]:
        num_missing_keys = [len(keys) for keys in self.missing_keys.values() if keys]
        num_extra_keys = [len(keys) for keys in self.extra_keys.values() if keys]

        if num_missing_keys:
            avg_missing_keys_when_missing_any = round(mean(num_missing_keys), 3)
        else:
            avg_missing_keys_when_missing_any = 0
        if num_extra_keys:
            avg_extra_keys_when_missing_any = round(mean(num_extra_keys), 3)
        else:
            avg_extra_keys_when_missing_any = 0

        return {
            "num_items": self.num_items,
            "total_num_parsing_errors": len(self.errors),
            "total_num_missing_keys": sum(num_missing_keys),
            "total_num_extra_keys": sum(num_extra_keys),
            "avg_missing_keys_when_missing_any": avg_missing_keys_when_missing_any,
            "avg_extra_keys_when_missing_any": avg_extra_keys_when_missing_any,
        }


class PredictionLoader:
    def __init__(self, root_dir: Path | str, judge_name: str | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.judge_name = judge_name.replace("/", "__") if judge_name else None

    @cached_property
    def schema(self) -> dict[str, Any]:
        return load_and_resolve_config(self.config_file)["ie_schema"]

    @property
    def predictions_file(self) -> Path:
        return self.root_dir / "predictions.json"

    @property
    def config_file(self) -> Path:
        return self.root_dir / "config.yaml"

    @property
    def judge_dir(self) -> Path:
        if self.judge_name is None:
            raise ValueError("judge_name must be set before accessing judge_dir")
        return self.root_dir / f"judge_{self.judge_name}"

    @property
    def batch_api_requests_file(self) -> Path:
        return self.judge_dir / "batch_api_requests.jsonl"

    @property
    def batch_api_metadata_file(self) -> Path:
        return self.judge_dir / "batch_api_metadata.json"

    @property
    def batch_request_info_file(self) -> Path:
        return self.judge_dir / "batch_request_info.json"

    @property
    def batch_api_results_file(self) -> Path:
        return self.judge_dir / "batch_api_results.jsonl"

    @property
    def batch_api_errors_file(self) -> Path:
        return self.judge_dir / "batch_api_errors.jsonl"

    @property
    def llm_judge_scores_file(self) -> Path:
        if self.judge_name is None:
            raise ValueError("judge_name must be set before accessing output_file")
        return self.judge_dir / f"scores_llm_as_judge_{self.judge_name}.json"

    @property
    def ngram_scores_file(self) -> Path:
        return self.root_dir / "scores_ngram.json"

    @cached_property
    def config(self) -> PredictInfoExtractionConfig:
        return PredictInfoExtractionConfig(**load_and_resolve_config(self.config_file))

    def setup_judge_dir(self) -> None:
        if self.judge_name is None:
            raise ValueError("judge_name must be set before setting up judge directory")
        if self.judge_dir.exists():
            logger.warning(f"Judge directory {self.judge_dir} already exists")
        self.judge_dir.mkdir(parents=True, exist_ok=True)

    def load_predictions(self, verbose: bool = False) -> ParsedPredictions:
        # todo: extract path management and pred loading to injected class
        with open(self.predictions_file) as f:
            preds = json.load(f)
        if not all("answer" in pred_item and "gold" in pred_item for pred_item in preds):
            raise ValueError("Predictions must contain 'answer' and 'gold' keys")

        parsed_preds = {}
        parsed_gold = {}
        parsing_errors = {}
        missing_keys = {}
        extra_keys = {}
        for idx, pred_item in enumerate(
            tqdm(preds, disable=not verbose, desc="Loading predictions")
        ):
            # we assume gold should always parse without errors
            gold_pred = parse_json_markdown(pred_item["gold"])
            try:
                parsed_preds[idx] = parse_json_markdown(pred_item["answer"])
                parsed_gold[idx] = gold_pred
                missing_keys[idx] = list(set(self.schema.keys()) - set(parsed_preds[idx].keys()))
                extra_keys[idx] = list(set(parsed_preds[idx].keys()) - set(self.schema.keys()))
            except json.JSONDecodeError as e:
                parsing_errors[idx] = str(e)
                missing_keys[idx] = []
                extra_keys[idx] = []

        return ParsedPredictions(
            predictions=parsed_preds,
            gold=parsed_gold,
            errors=parsing_errors,
            missing_keys=missing_keys,
            extra_keys=extra_keys,
            num_items=len(preds),
        )
