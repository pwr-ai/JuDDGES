from functools import cached_property
from typing import Any

from loguru import logger
from tqdm import trange

from juddges.evals.metrics import (
    evaluate_date,
    evaluate_enum,
    evaluate_list_greedy,
    evaluate_number,
    evaluate_string_rouge,
)
from juddges.llm_as_judge.base import EvalResults, ItemEvalResult
from juddges.llm_as_judge.data_model import ParsedPredictions


class ExtractionEvaluator:
    """Evaluates the quality of extracted information against a gold standard."""

    def __init__(self, schema: dict[str, Any]):
        self.schema = schema
        self.field_types = {name: props["type"] for name, props in schema.items()}
        self.enum_choices = {
            name: self._get_choices(name)
            for name, props in schema.items()
            if props["type"] == "enum"
        }

    @cached_property
    def zero_scores(self) -> dict[str, Any]:
        return {
            name: self.get_zero_score(field_type) for name, field_type in self.field_types.items()
        }

    def run(self, parsed_preds: ParsedPredictions) -> EvalResults:
        """Runs the evaluation over a list of predictions."""
        per_record_results = []
        parsing_errors = 0
        for idx in trange(parsed_preds.num_items, desc="Evaluating records"):
            try:
                pred_item = parsed_preds.predictions[idx]

            except KeyError:
                parsing_errors += 1
                per_record_results.append(
                    ItemEvalResult(
                        status="parsing_error",
                        result=self.zero_scores,
                        error=parsed_preds.errors[idx],
                    )
                )
                continue
            else:
                try:
                    gold_item = parsed_preds.gold[idx]
                    record_result = self.evaluate_record(pred_item, gold_item)
                    per_record_results.append(
                        ItemEvalResult.from_success(
                            result=record_result,
                            missing_keys=parsed_preds.missing_keys[idx],
                            extra_keys=parsed_preds.extra_keys[idx],
                        )
                    )
                except (TypeError, AttributeError) as e:
                    # todo: remove this
                    logger.error(f"Error evaluating record {idx}: {e}")
                    per_record_results.append(
                        ItemEvalResult(
                            status="judge_error",
                            result=self.zero_scores,
                            error=str(e),
                        )
                    )

        return EvalResults(
            ie_schema=self.schema,
            results=per_record_results,
        )

    def evaluate_record(
        self,
        pred_item: dict[str, Any],
        gold_item: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluates a single record (predicted vs. gold)."""
        field_results: dict[str, Any] = {}

        for key in self.schema.keys():
            field_type = self.field_types[key]

            try:
                pred_val = pred_item[key]
            except KeyError:
                return self.get_zero_score(field_type)

            gold_val = gold_item[key]

            if field_type == "date":
                field_results[key] = {"match": evaluate_date(pred_val, gold_val)}
            elif field_type in ["number", "integer"]:
                field_results[key] = {"match": evaluate_number(pred_val, gold_val)}
            elif field_type == "string":
                field_results[key] = evaluate_string_rouge(pred_val, gold_val)
            elif field_type == "enum":
                field_results[key] = evaluate_enum(pred_val, gold_val, self.enum_choices[key])
            elif field_type == "list":
                field_results[key] = evaluate_list_greedy(pred_val, gold_val)
            else:
                raise ValueError(f"Unknown field type: {field_type}")

        return field_results

    def get_zero_score(self, field_type: str) -> dict[str, Any]:
        if field_type == "date":
            return {"match": 0}
        elif field_type in ["number", "integer"]:
            return {"match": 0}
        elif field_type == "string":
            return {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        elif field_type == "enum":
            return {
                "match": 0,
                "predicted_in_choices": 0,
            }
        elif field_type == "list":
            return {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
            }
        else:
            raise ValueError(f"Unknown field type: {field_type}")

    def _get_choices(self, name: str) -> list[str]:
        """Determines the choices of a field based on schema properties."""
        choices = self.schema[name]["choices"]
        assert isinstance(choices, list) and len(choices) > 0

        try:
            required = self.schema[name]["required"]
        except KeyError:
            logger.warning(f"Field {name} has no required property, assuming it is required")
            required = True

        if not required:
            choices = [None] + choices
        return choices
