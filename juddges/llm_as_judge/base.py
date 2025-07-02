import itertools
import json
from functools import cached_property
from json import JSONDecodeError
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Counter, Literal, defaultdict

from langchain_core.utils.json import parse_json_markdown
from loguru import logger
from pydantic import BaseModel, Field

from juddges.llm_as_judge.prompts import SYSTEM_PROMPT, USER_PROMPT
from juddges.utils.config import load_and_resolve_config


class ParsedPredictions(BaseModel):
    predictions: dict[int, dict[str, Any]]
    gold: dict[int, dict[str, Any]]
    errors: dict[int, str]
    missing_keys: dict[int, list[str]] = Field(default_factory=dict)
    extra_keys: dict[int, list[str]] = Field(default_factory=dict)
    num_items: int


class ItemEvalResult(BaseModel):
    status: Literal["success", "judge_error", "parsing_error"]
    error: str | None
    result: dict[str, Any]
    missing_keys: list[str] = Field(default_factory=list)
    extra_keys: list[str] = Field(default_factory=list)

    @classmethod
    def from_success(
        cls,
        result: dict[str, Any],
        missing_keys: list[str] | None = None,
        extra_keys: list[str] | None = None,
    ) -> "ItemEvalResult":
        return cls(
            status="success",
            error=None,
            result=result,
            missing_keys=missing_keys or [],
            extra_keys=extra_keys or [],
        )


class EvalResults(BaseModel):
    ie_schema: dict[str, Any]
    results: list[ItemEvalResult]

    def model_dump(self) -> dict[str, Any]:
        return {
            "stats": self.get_statistics(),
            "aggregated_scores": self.get_aggregated_scores(),
            "all_results": [res.model_dump() for res in self.results],
        }

    def get_aggregated_scores(self) -> dict[str, Any]:
        aggregated_scores = defaultdict(list)
        for item_res in self.results:
            for key in self.ie_schema.keys():
                try:
                    aggregated_scores[key].append(item_res.result[key]["score"])
                except KeyError:
                    breakpoint()
                    logger.warning(f"Key {key} not found in result {item_res.result}")
                    aggregated_scores[key].append(0.0)
        return {
            key: {
                "mean_score": sum(scores) / len(scores),
                "std_score": stdev(scores),
            }
            for key, scores in aggregated_scores.items()
        }

    def get_statistics(self) -> dict[str, Any]:
        return {
            "total_docs": len(self.results),
            "num_success_evaluations": sum(res.status == "success" for res in self.results),
            "num_parsing_errors": sum(res.status == "parsing_error" for res in self.results),
            "num_judge_errors": sum(res.status == "judge_error" for res in self.results),
            "missing_keys": dict(
                Counter(itertools.chain.from_iterable(res.missing_keys for res in self.results))
            ),
            "extra_keys": dict(
                Counter(itertools.chain.from_iterable(res.extra_keys for res in self.results))
            ),
            "avg_missing_keys_when_success": mean(
                len(res.missing_keys) for res in self.results if res.status == "success"
            ),
            "avg_extra_keys_when_success": mean(
                len(res.extra_keys) for res in self.results if res.status == "success"
            ),
        }


class StructuredOutputJudgeBase:
    def __init__(self, predictions_dir: Path, judge_name: str) -> None:
        self.predictions_dir = predictions_dir
        self.judge_name = judge_name.replace("/", "__")
        self.schema = load_and_resolve_config(self.config_file)["ie_schema"]
        self.setup_judge_dir()

    @property
    def predictions_file(self) -> Path:
        return self.predictions_dir / "predictions.json"

    @property
    def config_file(self) -> Path:
        return self.predictions_dir / "config.yaml"

    @property
    def judge_dir(self) -> Path:
        return self.predictions_dir / f"judge_{self.judge_name}"

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
    def output_file(self) -> Path:
        return self.judge_dir / f"scores_llm_as_judge_{self.judge_name}.json"

    @cached_property
    def structured_response_schema_from_extraction_schema(self) -> dict[str, Any]:
        return {
            "name": "ScoreEvaluation",
            "description": "Evaluation scores for each field in the output",
            "parameters": {
                "type": "object",
                "properties": {
                    key: {
                        "type": "object",
                        "properties": {
                            "score": {
                                "type": "number",
                                "description": "The score, either 0, 1 for a single item, or average for list items",
                            },
                        },
                        "required": ["score"],
                        "additionalProperties": False,
                    }
                    for key in self.schema.keys()
                },
                "additionalProperties": False,
                "required": list(self.schema.keys()),
            },
        }

    def setup_judge_dir(self) -> None:
        if self.judge_dir.exists():
            logger.warning(f"Judge directory {self.judge_dir} already exists")
        self.judge_dir.mkdir(parents=True, exist_ok=True)

    def load_predictions(self) -> ParsedPredictions:
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
        for idx, pred_item in enumerate(preds):
            # we assume gold should always parse without errors
            gold_pred = parse_json_markdown(pred_item["gold"])
            try:
                parsed_preds[idx] = parse_json_markdown(pred_item["answer"])
                parsed_gold[idx] = gold_pred
                missing_keys[idx] = list(set(self.schema.keys()) - set(parsed_preds[idx].keys()))
                extra_keys[idx] = list(set(parsed_preds[idx].keys()) - set(self.schema.keys()))
            except JSONDecodeError as e:
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

    def prepare_eval_messages(
        self,
        parsed_preds: ParsedPredictions,
    ) -> dict[int, list[dict[str, str]]]:
        dataset_messages = {}
        for idx in parsed_preds.predictions.keys():
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        schema=self.schema,
                        outputs=parsed_preds.predictions[idx],
                        reference_outputs=parsed_preds.gold[idx],
                    ),
                },
            ]
            dataset_messages[idx] = messages
        return dataset_messages

    def get_zero_scores(self) -> dict[str, Any]:
        return dict.fromkeys(self.schema.keys(), 0.0)

    # todo: move to child class
    def merge_judge_results_with_failed_items(
        self,
        parsed_preds: ParsedPredictions,
        eval_results: dict[int, ItemEvalResult],
    ) -> EvalResults:
        results = []
        for idx in range(parsed_preds.num_items):
            try:
                res = eval_results[idx]
            except KeyError:
                res = ItemEvalResult(
                    status="parsing_error",
                    error=parsed_preds.errors[idx],
                    result=self.get_zero_scores(),
                )
            results.append(res)

        return EvalResults(results=results, ie_schema=self.schema)
