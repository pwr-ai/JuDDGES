#!/usr/bin/env python
"""Experiment 3: Type-Specific Metrics Aggregation.

Computes metrics stratified by field type from existing evaluation results.
This produces the tables showing that binary fields are easy (0.95+),
free-text fields are hard (0.2-0.6), etc.

Reads existing scores_ngram.json or scores_llm_as_judge_*.json files
and aggregates by field type (binary, categorical, date, free-text, list, structured).

Usage:
    python scripts/neurips_experiments/exp3_type_specific_metrics.py \
        --results-dir results/ \
        --output results/neurips/exp3_type_metrics.json
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import typer
import yaml
from loguru import logger

from juddges.utils.misc import save_json

# Field type taxonomy for the paper
FIELD_TYPE_CATEGORIES = {
    "binary": "Fields with Tak/Nie values",
    "categorical": "Enum fields with 3+ choices",
    "date": "Date fields (YYYY-MM-DD format)",
    "free_text": "Free-form string fields",
    "list": "List/array fields",
    "structured_json": "Complex JSON object fields",
}


def classify_field_type(field_name: str, field_props: dict[str, Any]) -> str:
    """Classify a schema field into one of the 6 difficulty categories."""
    field_type = field_props.get("type", "string")
    choices = field_props.get("choices", [])

    if field_type == "enum":
        if set(choices) <= {"Tak", "Nie", None}:
            return "binary"
        return "categorical"
    elif field_type == "date" or "data" in field_name.lower():
        return "date"
    elif field_type == "list":
        return "list"
    elif field_type == "string":
        # Heuristic: if description mentions JSON or structured
        desc = field_props.get("description", "").lower()
        if "json" in desc or "object" in desc or "array" in desc:
            return "structured_json"
        return "free_text"
    elif field_type in ("number", "integer"):
        return "categorical"  # Treat as categorical for analysis
    else:
        return "free_text"


def aggregate_by_type(
    scores: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Aggregate per-field scores into per-type summaries."""
    type_scores = defaultdict(lambda: defaultdict(list))
    field_to_type = {}

    for field_name, field_props in schema.items():
        field_type = classify_field_type(field_name, field_props)
        field_to_type[field_name] = field_type

        if field_name in scores:
            for metric_name, metric_data in scores[field_name].items():
                score = metric_data.get("mean_score", 0)
                type_scores[field_type][metric_name].append(score)

    # Compute aggregates
    result = {}
    for ftype, metrics in type_scores.items():
        result[ftype] = {
            "num_fields": sum(1 for ft in field_to_type.values() if ft == ftype),
            "fields": [f for f, ft in field_to_type.items() if ft == ftype],
        }
        for metric_name, values in metrics.items():
            result[ftype][metric_name] = {
                "mean": round(mean(values), 4),
                "std": round(stdev(values), 4) if len(values) > 1 else 0.0,
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "n": len(values),
            }

    return result


def find_result_dirs(results_dir: Path) -> list[dict[str, Any]]:
    """Find all evaluation result directories with scores."""
    found = []
    for scores_file in results_dir.rglob("scores_*.json"):
        config_file = scores_file.parent / "config.yaml"
        if not config_file.exists():
            # Check parent for config
            config_file = scores_file.parent.parent / "config.yaml"

        if config_file.exists():
            found.append({
                "scores_file": scores_file,
                "config_file": config_file,
                "model": scores_file.parent.name,
                "dataset": scores_file.parent.parent.name,
                "eval_type": scores_file.stem.replace("scores_", ""),
            })
    return found


def main(
    results_dir: Path = typer.Argument(
        ..., help="Root results directory"
    ),
    output: Path = typer.Option(
        "results/neurips/exp3_type_metrics.json",
        help="Output path",
    ),
):
    """Compute type-stratified metrics from existing evaluation results."""
    result_dirs = find_result_dirs(results_dir)
    logger.info(f"Found {len(result_dirs)} result files")

    all_results = {}

    for rd in result_dirs:
        logger.info(f"Processing {rd['dataset']}/{rd['model']} ({rd['eval_type']})")

        # Load schema
        with open(rd["config_file"]) as f:
            config = yaml.safe_load(f)
        schema = config.get("ie_schema", config)

        # Load scores
        with open(rd["scores_file"]) as f:
            scores_data = json.load(f)

        # Extract aggregated scores
        if "aggregated_scores" in scores_data:
            scores = scores_data["aggregated_scores"]
        else:
            logger.warning(f"No aggregated_scores in {rd['scores_file']}, skipping")
            continue

        # Aggregate by type
        type_agg = aggregate_by_type(scores, schema)

        key = f"{rd['dataset']}/{rd['model']}/{rd['eval_type']}"
        all_results[key] = {
            "dataset": rd["dataset"],
            "model": rd["model"],
            "eval_type": rd["eval_type"],
            "type_aggregation": type_agg,
            "schema_composition": {
                ftype: {
                    "count": data["num_fields"],
                    "pct": round(data["num_fields"] / len(schema) * 100, 1),
                }
                for ftype, data in type_agg.items()
            },
        }

    # Cross-model comparison per type
    comparison = defaultdict(lambda: defaultdict(list))
    for key, data in all_results.items():
        dataset = data["dataset"]
        model = data["model"]
        for ftype, type_data in data["type_aggregation"].items():
            # Get the primary metric for this type
            if "match" in type_data:
                primary = type_data["match"]["mean"]
            elif "rougeL" in type_data:
                primary = type_data["rougeL"]["mean"]
            elif "f1" in type_data:
                primary = type_data["f1"]["mean"]
            elif "score" in type_data:
                primary = type_data["score"]["mean"]
            else:
                continue
            comparison[dataset][ftype].append({
                "model": model,
                "score": primary,
            })

    output_data = {
        "experiment": "type_specific_metrics",
        "per_model_results": all_results,
        "cross_model_comparison": dict(comparison),
        "field_type_definitions": FIELD_TYPE_CATEGORIES,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_data, output, ensure_ascii=False)
    logger.info(f"Results saved to {output}")

    # Print summary table
    print(f"\n{'='*80}")
    print("TYPE-SPECIFIC METRICS SUMMARY")
    print(f"{'='*80}")
    for key, data in all_results.items():
        print(f"\n{key}:")
        print(f"  {'Type':<18} {'Fields':>6} {'Primary Metric':>15}")
        print(f"  {'-'*42}")
        for ftype, type_data in sorted(data["type_aggregation"].items()):
            if "match" in type_data:
                metric_val = f"{type_data['match']['mean']:.3f}"
            elif "rougeL" in type_data:
                metric_val = f"{type_data['rougeL']['mean']:.3f}"
            elif "f1" in type_data:
                metric_val = f"{type_data['f1']['mean']:.3f}"
            else:
                metric_val = "N/A"
            print(f"  {ftype:<18} {type_data['num_fields']:>6} {metric_val:>15}")


if __name__ == "__main__":
    typer.run(main)
