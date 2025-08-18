import itertools
from functools import cached_property
from math import sqrt
from statistics import mean, stdev
from typing import Any, Counter, Literal, defaultdict

from loguru import logger
from pydantic import BaseModel, Field

from juddges.llm_as_judge.data_model import ParsedPredictions, PredictionLoader


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
        """Computes mean and standard error of scores for each field and score key.
        See https://arxiv.org/abs/2411.00640 for more details.
        """
        scores_sum = defaultdict(lambda: defaultdict(list))
        for item_res in self.results:
            for key in self.ie_schema.keys():
                try:
                    scores_data = item_res.result[key]
                except KeyError:
                    continue
                for score_key, score_value in scores_data.items():
                    if score_value is None:
                        continue
                    scores_sum[key][score_key].append(score_value)

        aggregated_scores = {}
        for key, scores in scores_sum.items():
            aggregated_scores[key] = {}
            for score_name, score_values in scores.items():
                try:
                    aggregated_scores[key][score_name] = {
                        "mean_score": sum(score_values) / len(score_values),
                        "standard_error": stdev(score_values) / sqrt(len(score_values)),
                    }
                except Exception as e:
                    logger.error(f"Error aggregating scores for {key} {score_name}: {e}")
        return aggregated_scores

    def get_statistics(self) -> dict[str, Any]:
        assert any(res.status == "success" for res in self.results)
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
            "avg_missing_keys_when_success": float(
                mean(len(res.missing_keys) for res in self.results if res.status == "success")
            ),
            "avg_extra_keys_when_success": float(
                mean(len(res.extra_keys) for res in self.results if res.status == "success")
            ),
        }


class StructuredOutputJudgeBase:
    def __init__(
        self,
        pred_loader: PredictionLoader,
        judge_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> None:
        self.pred_loader = pred_loader
        self.judge_name = judge_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        assert self.pred_loader.judge_dir.exists()

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
                    for key in self.pred_loader.schema.keys()
                },
                "additionalProperties": False,
                "required": list(self.pred_loader.schema.keys()),
            },
        }

    def prepare_eval_messages(
        self,
        parsed_preds: ParsedPredictions,
    ) -> dict[int, list[dict[str, str]]]:
        dataset_messages = {}
        for idx in parsed_preds.predictions.keys():
            messages = self.prepare_single_item_messages(
                pred=parsed_preds.predictions[idx],
                gold=parsed_preds.gold[idx],
                user_prompt=self.user_prompt,
                system_prompt=self.system_prompt,
            )
            dataset_messages[idx] = messages
        return dataset_messages

    def prepare_single_item_messages(
        self,
        pred: dict[str, Any],
        gold: dict[str, Any],
    ) -> list[dict[str, str]]:
        messages = [
            {
                "role": "user",
                "content": self.user_prompt.format(
                    schema=self.pred_loader.schema,
                    outputs=pred,
                    reference_outputs=gold,
                ),
            },
        ]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def get_zero_scores(self) -> dict[str, Any]:
        return {key: {"score": 0.0} for key in self.pred_loader.schema.keys()}
