#!/usr/bin/env python
"""Experiment 1: Annotation Validation.

Compares model performance on:
  (a) Semi-automatic annotations (GPT-4 pre-annotated, human-reviewed)
  (b) 100 fully manual expert annotations (no LLM pre-annotation)

This quantifies the annotation circularity bias: how much does GPT-4.1's
high score (0.989) on pl-swiss-franc-loans owe to GPT-4 pre-annotation?

Outputs:
  - Per-field comparison table (semi-auto vs manual)
  - Overall score delta
  - Field-level bias analysis (which fields are most inflated)

Usage:
    python scripts/neurips_experiments/exp1_annotation_validation.py \
        --predictions-dir results/pl-swiss-franc-loans/gpt-4.1/ \
        --manual-annotations-path data/annotations/manual_100.json \
        --output results/neurips/exp1_annotation_validation.json
"""

import json
from pathlib import Path
from typing import Any

import typer
from loguru import logger

from juddges.evals.extraction import ExtractionEvaluator
from juddges.llm_as_judge.base import EvalResults
from juddges.llm_as_judge.data_model import ParsedPredictions, PredictionLoader
from juddges.utils.misc import save_json


def load_manual_annotations(
    manual_path: Path,
    predictions: list[dict[str, Any]],
    id_field: str = "id",
) -> list[dict[str, Any]]:
    """Replace gold annotations with manual-only annotations where available.

    Returns a new predictions list with gold values swapped for manual annotations
    for the 100 documents that have them.
    """
    with open(manual_path) as f:
        manual_anns = json.load(f)

    # Build lookup: id -> manual annotation
    manual_lookup = {ann[id_field]: ann for ann in manual_anns}
    logger.info(f"Loaded {len(manual_lookup)} manual annotations")

    # Filter predictions to only those with manual annotations,
    # and replace gold with manual gold
    matched = []
    for pred in predictions:
        gold = json.loads(pred["gold"]) if isinstance(pred["gold"], str) else pred["gold"]
        pred_id = gold.get(id_field) or gold.get("_id") or gold.get("signature")

        if pred_id in manual_lookup:
            new_pred = {
                "answer": pred["answer"],
                "gold": json.dumps(manual_lookup[pred_id], ensure_ascii=False),
            }
            matched.append(new_pred)

    logger.info(f"Matched {len(matched)} predictions with manual annotations")
    return matched


def compute_delta_analysis(
    semi_auto_scores: dict[str, Any],
    manual_scores: dict[str, Any],
) -> dict[str, Any]:
    """Compute per-field delta between semi-auto and manual annotation scores."""
    deltas = {}
    for field in semi_auto_scores:
        if field not in manual_scores:
            continue

        semi = semi_auto_scores[field]
        manual = manual_scores[field]

        field_deltas = {}
        for metric in semi:
            if metric in manual:
                semi_val = semi[metric].get("mean_score", 0)
                manual_val = manual[metric].get("mean_score", 0)
                field_deltas[metric] = {
                    "semi_auto": round(semi_val, 4),
                    "manual": round(manual_val, 4),
                    "delta": round(semi_val - manual_val, 4),
                    "inflation_pct": round(
                        (semi_val - manual_val) / max(manual_val, 1e-6) * 100, 2
                    ),
                }
        deltas[field] = field_deltas

    return deltas


def main(
    predictions_dir: Path = typer.Argument(
        ..., help="Directory with predictions.json and config.yaml"
    ),
    manual_annotations_path: Path = typer.Option(
        ..., "--manual-annotations-path", help="Path to manual-only annotations JSON"
    ),
    output: Path = typer.Option(
        "results/neurips/exp1_annotation_validation.json",
        help="Output path for results",
    ),
):
    """Compare semi-auto vs manual annotations to quantify pre-annotation bias."""
    # Load predictions and schema
    pred_loader = PredictionLoader(root_dir=predictions_dir)
    schema = pred_loader.schema

    with open(pred_loader.predictions_file) as f:
        all_predictions = json.load(f)

    logger.info(f"Loaded {len(all_predictions)} total predictions")

    # --- Evaluate on SEMI-AUTO annotations (original gold) ---
    logger.info("Evaluating on semi-automatic annotations...")
    parsed_semi = PredictionLoader.load_predictions(schema, all_predictions, verbose=True)
    evaluator = ExtractionEvaluator(schema)
    semi_results = evaluator.run(parsed_semi)
    semi_scores = semi_results.get_aggregated_scores()

    # --- Evaluate on MANUAL annotations ---
    logger.info("Evaluating on manual-only annotations...")
    manual_predictions = load_manual_annotations(
        manual_annotations_path, all_predictions
    )

    if len(manual_predictions) == 0:
        logger.error("No predictions matched manual annotations! Check ID fields.")
        raise typer.Exit(1)

    parsed_manual = PredictionLoader.load_predictions(schema, manual_predictions, verbose=True)
    manual_results = evaluator.run(parsed_manual)
    manual_scores = manual_results.get_aggregated_scores()

    # --- Delta analysis ---
    deltas = compute_delta_analysis(semi_scores, manual_scores)

    # --- Compute overall means ---
    def overall_mean(scores: dict) -> float:
        values = []
        for field_scores in scores.values():
            for metric_data in field_scores.values():
                values.append(metric_data.get("mean_score", 0))
        return sum(values) / len(values) if values else 0

    semi_overall = overall_mean(semi_scores)
    manual_overall = overall_mean(manual_scores)

    # --- Most inflated fields ---
    inflation_ranking = []
    for field, field_deltas in deltas.items():
        for metric, data in field_deltas.items():
            if data["delta"] > 0.01:
                inflation_ranking.append({
                    "field": field,
                    "metric": metric,
                    "delta": data["delta"],
                    "inflation_pct": data["inflation_pct"],
                })
    inflation_ranking.sort(key=lambda x: x["delta"], reverse=True)

    results = {
        "experiment": "annotation_validation",
        "summary": {
            "semi_auto_overall": round(semi_overall, 4),
            "manual_overall": round(manual_overall, 4),
            "overall_delta": round(semi_overall - manual_overall, 4),
            "num_semi_auto_docs": parsed_semi.num_items,
            "num_manual_docs": parsed_manual.num_items,
        },
        "per_field_deltas": deltas,
        "most_inflated_fields": inflation_ranking[:10],
        "semi_auto_stats": semi_results.get_statistics(),
        "manual_stats": manual_results.get_statistics(),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, output, ensure_ascii=False)
    logger.info(f"Results saved to {output}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"ANNOTATION VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Semi-auto annotations ({parsed_semi.num_items} docs): {semi_overall:.4f}")
    print(f"Manual annotations ({parsed_manual.num_items} docs):   {manual_overall:.4f}")
    print(f"Delta (bias):                              {semi_overall - manual_overall:+.4f}")
    print(f"\nTop 5 most inflated fields:")
    for item in inflation_ranking[:5]:
        print(f"  {item['field']}.{item['metric']}: {item['delta']:+.4f} ({item['inflation_pct']:+.1f}%)")


if __name__ == "__main__":
    typer.run(main)
