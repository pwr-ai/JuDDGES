import itertools
from functools import cached_property
from statistics import mean, stdev
from typing import Any, Counter, Literal, defaultdict

from pydantic import BaseModel, Field

from juddges.llm_as_judge.data_model import ParsedPredictions, PredictionLoader
from juddges.llm_as_judge.prompts import SYSTEM_PROMPT, USER_PROMPT


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
                aggregated_scores[key].append(item_res.result[key]["score"])
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
    def __init__(self, pred_loader: PredictionLoader, judge_name: str) -> None:
        self.pred_loader = pred_loader
        self.judge_name = judge_name

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
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        schema=self.pred_loader.schema,
                        outputs=parsed_preds.predictions[idx],
                        reference_outputs=parsed_preds.gold[idx],
                    ),
                },
            ]
            dataset_messages[idx] = messages
        return dataset_messages

    def get_zero_scores(self) -> dict[str, Any]:
        return {key: {"score": 0.0} for key in self.pred_loader.schema.keys()}

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

        return EvalResults(results=results, ie_schema=self.pred_loader.schema)
