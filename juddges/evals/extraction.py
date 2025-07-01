import json
from collections import defaultdict
from typing import Any

from langchain_core.utils.json import parse_json_markdown
from loguru import logger
from tqdm import tqdm

from juddges.evals.metrics import (
    evaluate_date,
    evaluate_enum,
    evaluate_list_greedy,
    evaluate_number,
    evaluate_string_rouge,
)


class ExtractionEvaluator:
    """Evaluates the quality of extracted information against a gold standard."""

    def __init__(self, schema: dict[str, Any]):
        self.schema = schema
        self.field_types = {
            name: self._get_field_type(name, props) for name, props in schema.items()
        }
        self.choices = {
            name: self._get_choices(name)
            for name, props in schema.items()
            if props["type"] == "enum"
        }

    def run(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """Runs the evaluation over a list of predictions."""
        per_record_results = []
        parsing_errors = 0

        filtered_predictions = [
            (idx, p) for idx, p in enumerate(predictions) if p["finish_reason"] == "stop"
        ]

        for idx, pred in tqdm(filtered_predictions, desc="Evaluating records"):
            try:
                answer = parse_json_markdown(pred["answer"])
                gold = parse_json_markdown(pred["gold"])

                record_result = self.evaluate_record(answer, gold)
                per_record_results.append({"index": idx, **record_result})

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing record {idx}: {e}")
                parsing_errors += 1
                per_record_results.append({"index": idx, "error": str(e)})

        return self.aggregate_results(
            per_record_results,
            len(predictions),
            len(filtered_predictions),
            parsing_errors,
        )

    def evaluate_record(
        self,
        predicted_json: dict[str, Any],
        gold_json: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluates a single record (predicted vs. gold)."""
        gold_keys = set(gold_json.keys())
        predicted_keys = set(predicted_json.keys())

        common_keys = gold_keys.intersection(predicted_keys)
        missing_keys = list(gold_keys - predicted_keys)
        extra_keys = list(predicted_keys - gold_keys)

        field_results: dict[str, Any] = {}
        for key in common_keys:
            try:
                field_type = self.field_types[key]
            except KeyError as err:
                raise ValueError(f"Field {key} not found in schema") from err

            pred_val = predicted_json[key]
            gold_val = gold_json[key]

            if field_type == "date":
                field_results[key] = {"match": evaluate_date(pred_val, gold_val)}
            elif field_type == "number":
                field_results[key] = {"match": evaluate_number(pred_val, gold_val)}
            elif field_type == "string":
                field_results[key] = {"rouge": evaluate_string_rouge(pred_val, gold_val)}
            elif field_type == "enum":
                field_results[key] = evaluate_enum(pred_val, gold_val, self.choices[key])
            elif field_type == "list":
                field_results[key] = {"metrics": evaluate_list_greedy(pred_val, gold_val)}
            else:
                raise ValueError(f"Unknown field type: {field_type}")

        return {
            "field_results": field_results,
            "missing_keys": missing_keys,
            "extra_keys": extra_keys,
        }

    def aggregate_results(
        self,
        per_record_results: list[dict[str, Any]],
        total_records: int,
        filtered_records: int,
        parsing_errors: int,
    ) -> dict[str, Any]:
        """Aggregates results from all records."""
        field_metrics: dict[str, Any] = defaultdict(lambda: defaultdict(list))
        total_missing_keys = 0
        total_extra_keys = 0
        records_with_missing_keys = 0
        records_with_extra_keys = 0

        for result in per_record_results:
            if "error" in result:
                continue

            if result.get("missing_keys"):
                total_missing_keys += len(result["missing_keys"])
                records_with_missing_keys += 1
            if result.get("extra_keys"):
                total_extra_keys += len(result["extra_keys"])
                records_with_extra_keys += 1

            for key, res in result["field_results"].items():
                field_type = self.field_types[key]
                if field_type in ["date", "number"]:
                    field_metrics[key]["matches"].append(res["match"])
                elif field_type == "enum":
                    field_metrics[key]["matches"].append(res["match"])
                    field_metrics[key]["hallucinations"].append(res["hallucinated"])
                    field_metrics[key]["predicted_in_choices"].append(res["predicted_in_choices"])
                    field_metrics[key]["gold_in_choices"].append(res["gold_in_choices"])
                elif field_type == "string" and res.get("rouge"):
                    field_metrics[key]["rouge"].append(res["rouge"])
                elif field_type == "list":
                    field_metrics[key]["list_metrics"].append(res["metrics"])

        summary: dict[str, Any] = defaultdict(dict)
        for key, metrics in field_metrics.items():
            field_type = self.field_types[key]
            if "matches" in metrics:
                summary[key]["accuracy"] = sum(metrics["matches"]) / len(metrics["matches"])
            if "hallucinations" in metrics:
                summary[key]["hallucinations"] = sum(metrics["hallucinations"])
                summary[key]["hallucination_rate"] = sum(metrics["hallucinations"]) / len(
                    metrics["hallucinations"]
                )
            if "predicted_in_choices" in metrics:
                summary[key]["predicted_in_choices_rate"] = sum(
                    metrics["predicted_in_choices"]
                ) / len(metrics["predicted_in_choices"])
            if "gold_in_choices" in metrics:
                summary[key]["gold_in_choices_rate"] = sum(metrics["gold_in_choices"]) / len(
                    metrics["gold_in_choices"]
                )
            if "rouge" in metrics:
                avg_rouge: dict[str, float] = defaultdict(float)
                for r in metrics["rouge"]:
                    for r_key, r_val in r.items():
                        avg_rouge[r_key] += r_val
                for r_key in avg_rouge:
                    avg_rouge[r_key] /= len(metrics["rouge"])
                summary[key]["rouge"] = dict(avg_rouge)
            if "list_metrics" in metrics:
                avg_list_metrics: dict[str, float] = defaultdict(float)
                for m in metrics["list_metrics"]:
                    for m_key, m_val in m.items():
                        avg_list_metrics[m_key] += m_val
                for m_key in avg_list_metrics:
                    avg_list_metrics[m_key] /= len(metrics["list_metrics"])
                summary[key]["list_metrics"] = dict(avg_list_metrics)

        return {
            "summary_metrics": {
                "total_records": total_records,
                "filtered_records (finish_reason=='stop')": filtered_records,
                "evaluated_records": len(per_record_results) - parsing_errors,
                "parsing_errors": parsing_errors,
                "total_missing_keys": total_missing_keys,
                "total_extra_keys": total_extra_keys,
                "records_with_missing_keys": records_with_missing_keys,
                "records_with_extra_keys": records_with_extra_keys,
                "field_metrics": dict(summary),
            },
            "per_record_results": per_record_results,
        }

    def _get_field_type(self, field_name: str, properties: dict[str, Any]) -> str:
        """Determines the type of a field based on schema properties."""
        if "date" in field_name:
            return "date"
        elif properties.get("type") == "enum":
            return "enum"
        elif properties.get("type") == "list":
            return "list"
        elif properties.get("type") == "number":
            return "number"
        elif properties.get("type") == "string":
            return "string"
        else:
            raise ValueError(f"Unknown field type: {properties.get('type')}")

    def _get_choices(self, name: str) -> list[str]:
        """Determines the choices of a field based on schema properties."""
        choices = self.schema[name]["choices"]
        assert isinstance(choices, list) and len(choices) > 0
        if self.schema[name]["required"] is False:
            choices = [None] + choices
        return choices
