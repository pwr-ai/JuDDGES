#!/usr/bin/env python
"""Experiment 6: Field Difficulty Taxonomy (Task 5).

Categorizes all 80 extraction fields into 6 difficulty types and computes
per-type performance statistics across all models.

Produces:
  - Field type distribution per schema
  - Performance variance across types (order of magnitude)
  - Per-field difficulty ranking
  - Actionable guidance: which field types need research

Usage:
    python scripts/neurips_experiments/exp6_field_difficulty.py \
        --results-dir results/ \
        --output results/neurips/exp6_field_difficulty.json
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

# Field type taxonomy
FIELD_TYPES = {
    "binary": "Boolean (Tak/Nie) fields",
    "categorical": "Enum fields with 3+ choices",
    "date": "Date extraction (YYYY-MM-DD)",
    "free_text": "Free-form text extraction",
    "list": "List/array extraction",
    "structured_json": "Complex JSON object extraction",
}


def classify_field(field_name: str, field_props: dict[str, Any]) -> str:
    """Classify a schema field into difficulty category."""
    ftype = field_props.get("type", "string")
    choices = field_props.get("choices", [])

    if ftype == "enum":
        if set(choices) <= {"Tak", "Nie", None}:
            return "binary"
        return "categorical"
    elif ftype == "date" or "data" in field_name.lower():
        return "date"
    elif ftype == "list":
        return "list"
    elif ftype == "string":
        desc = field_props.get("description", "").lower()
        if any(kw in desc for kw in ["json", "object", "array", "struct"]):
            return "structured_json"
        # Long descriptions suggest complex extraction
        if len(desc) > 200:
            return "free_text"
        return "free_text"
    return "free_text"


def get_primary_score(field_scores: dict[str, Any]) -> float | None:
    """Extract the primary performance score for a field."""
    # Priority: score > match > rougeL > f1
    if "score" in field_scores:
        return field_scores["score"].get("mean_score", 0)
    if "match" in field_scores:
        return field_scores["match"].get("mean_score", 0)
    if "rougeL" in field_scores:
        return field_scores["rougeL"].get("mean_score", 0)
    if "f1" in field_scores:
        return field_scores["f1"].get("mean_score", 0)
    return None


def load_all_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load all evaluation results from the results directory."""
    entries = []
    for scores_file in results_dir.rglob("scores_*.json"):
        config_file = scores_file.parent / "config.yaml"
        if not config_file.exists():
            config_file = scores_file.parent.parent / "config.yaml"
        if not config_file.exists():
            continue

        with open(config_file) as f:
            config = yaml.safe_load(f)
        schema = config.get("ie_schema", config)

        with open(scores_file) as f:
            scores_data = json.load(f)

        agg_scores = scores_data.get("aggregated_scores", {})
        if not agg_scores:
            continue

        entries.append({
            "model": scores_file.parent.name,
            "dataset": scores_file.parent.parent.name,
            "eval_type": scores_file.stem.replace("scores_", ""),
            "schema": schema,
            "scores": agg_scores,
        })

    return entries


def main(
    results_dir: Path = typer.Argument(..., help="Root results directory"),
    output: Path = typer.Option(
        "results/neurips/exp6_field_difficulty.json",
    ),
):
    """Analyze field difficulty across all models and schemas."""
    entries = load_all_results(results_dir)
    logger.info(f"Loaded {len(entries)} evaluation results")

    if not entries:
        logger.error("No results found. Check --results-dir path.")
        raise typer.Exit(1)

    # Classify all fields across all schemas
    all_fields = {}  # field_name -> {type, dataset, scores_by_model}
    for entry in entries:
        dataset = entry["dataset"]
        model = entry["model"]
        for field_name, field_props in entry["schema"].items():
            key = f"{dataset}/{field_name}"
            if key not in all_fields:
                all_fields[key] = {
                    "field_name": field_name,
                    "dataset": dataset,
                    "field_type": classify_field(field_name, field_props),
                    "scores_by_model": {},
                }
            if field_name in entry["scores"]:
                score = get_primary_score(entry["scores"][field_name])
                if score is not None:
                    all_fields[key]["scores_by_model"][model] = score

    # Aggregate by field type
    type_analysis = defaultdict(lambda: {
        "fields": [],
        "all_scores": [],
        "per_model_scores": defaultdict(list),
    })

    for key, fdata in all_fields.items():
        ftype = fdata["field_type"]
        type_analysis[ftype]["fields"].append(key)
        for model, score in fdata["scores_by_model"].items():
            type_analysis[ftype]["all_scores"].append(score)
            type_analysis[ftype]["per_model_scores"][model].append(score)

    # Compute statistics
    type_stats = {}
    for ftype, data in type_analysis.items():
        scores = data["all_scores"]
        type_stats[ftype] = {
            "num_fields": len(set(data["fields"])),
            "description": FIELD_TYPES.get(ftype, ""),
            "overall": {
                "mean": round(mean(scores), 4) if scores else 0,
                "std": round(stdev(scores), 4) if len(scores) > 1 else 0,
                "min": round(min(scores), 4) if scores else 0,
                "max": round(max(scores), 4) if scores else 0,
                "n_observations": len(scores),
            },
            "per_model": {
                model: {
                    "mean": round(mean(model_scores), 4),
                    "n": len(model_scores),
                }
                for model, model_scores in data["per_model_scores"].items()
            },
            "fields": sorted(set(data["fields"])),
        }

    # Difficulty ranking (by mean score, ascending = hardest first)
    difficulty_ranking = sorted(
        type_stats.items(),
        key=lambda x: x[1]["overall"]["mean"],
    )

    # Per-field difficulty (across all models)
    field_difficulty = []
    for key, fdata in all_fields.items():
        scores = list(fdata["scores_by_model"].values())
        if scores:
            field_difficulty.append({
                "field": key,
                "field_type": fdata["field_type"],
                "mean_score": round(mean(scores), 4),
                "n_models": len(scores),
                "best_model": max(fdata["scores_by_model"], key=fdata["scores_by_model"].get),
                "best_score": round(max(scores), 4),
                "worst_score": round(min(scores), 4),
                "variance": round(max(scores) - min(scores), 4),
            })
    field_difficulty.sort(key=lambda x: x["mean_score"])

    # Performance variance ratio (max type mean / min type mean)
    type_means = [ts["overall"]["mean"] for ts in type_stats.values() if ts["overall"]["mean"] > 0]
    variance_ratio = max(type_means) / min(type_means) if type_means and min(type_means) > 0 else float("inf")

    results = {
        "experiment": "field_difficulty",
        "summary": {
            "total_fields": len(all_fields),
            "total_models": len(set(e["model"] for e in entries)),
            "total_datasets": len(set(e["dataset"] for e in entries)),
            "performance_variance_ratio": round(variance_ratio, 2),
            "easiest_type": difficulty_ranking[-1][0] if difficulty_ranking else None,
            "hardest_type": difficulty_ranking[0][0] if difficulty_ranking else None,
        },
        "type_statistics": type_stats,
        "difficulty_ranking": [
            {"type": ftype, "mean_score": data["overall"]["mean"]}
            for ftype, data in difficulty_ranking
        ],
        "per_field_difficulty": field_difficulty,
        "hardest_fields": field_difficulty[:10],
        "easiest_fields": field_difficulty[-10:],
        "schema_composition": {
            dataset: {
                ftype: sum(1 for f in all_fields.values()
                           if f["dataset"] == dataset and f["field_type"] == ftype)
                for ftype in FIELD_TYPES
            }
            for dataset in set(f["dataset"] for f in all_fields.values())
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, output, ensure_ascii=False)
    logger.info(f"Results saved to {output}")

    # Print summary
    print(f"\n{'='*70}")
    print("FIELD DIFFICULTY TAXONOMY")
    print(f"{'='*70}")
    print(f"Total fields analyzed: {len(all_fields)}")
    print(f"Performance variance ratio: {variance_ratio:.1f}x")
    print(f"\n{'Type':<18} {'Fields':>6} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print(f"{'-'*52}")
    for ftype, data in difficulty_ranking:
        o = data["overall"]
        print(f"{ftype:<18} {data['num_fields']:>6} {o['mean']:>8.3f} {o['min']:>8.3f} {o['max']:>8.3f}")

    print(f"\nTop 5 HARDEST fields:")
    for f in field_difficulty[:5]:
        print(f"  {f['field']:<45} {f['mean_score']:.3f} ({f['field_type']})")

    print(f"\nTop 5 EASIEST fields:")
    for f in field_difficulty[-5:]:
        print(f"  {f['field']:<45} {f['mean_score']:.3f} ({f['field_type']})")


if __name__ == "__main__":
    typer.run(main)
