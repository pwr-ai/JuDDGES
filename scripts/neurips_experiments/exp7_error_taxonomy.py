#!/usr/bin/env python
"""Experiment 7: Error Taxonomy Analysis.

Samples extraction errors and categorizes them into 5 error types:
  1. Format errors: Valid extraction but wrong output format
  2. Omission errors: Field present in document but not extracted
  3. Hallucination errors: Extracted value not in source document
  4. Granularity errors: Correct concept, wrong level of detail
  5. Interpretation errors: Ambiguous field, reasonable but wrong answer

Uses LLM-as-judge to classify sampled errors into these categories.

Usage:
    python scripts/neurips_experiments/exp7_error_taxonomy.py \
        --results-dir results/pl-swiss-franc-loans/ \
        --sample-size 50 \
        --output results/neurips/exp7_error_taxonomy.json
"""

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from juddges.utils.misc import save_json

load_dotenv()

ERROR_CATEGORIES = {
    "format_error": "Valid information extracted but wrong output format (e.g., date as 'January 15' instead of '2024-01-15')",
    "omission_error": "Information present in document but not extracted by the model",
    "hallucination_error": "Extracted information not present in the source document",
    "granularity_error": "Correct concept identified but wrong level of detail (too specific or too general)",
    "interpretation_error": "Ambiguous or subjective field where the model made a reasonable but incorrect interpretation",
}


def sample_errors(
    scores_file: Path,
    predictions_file: Path,
    schema: dict[str, Any],
    sample_size: int = 50,
) -> list[dict[str, Any]]:
    """Sample extraction errors from evaluation results."""
    with open(scores_file) as f:
        scores_data = json.load(f)

    with open(predictions_file) as f:
        predictions = json.load(f)

    agg_scores = scores_data.get("aggregated_scores", {})
    all_results = scores_data.get("all_results", [])

    # Find fields with errors (score < 1.0)
    error_samples = []
    for idx, result in enumerate(all_results):
        if result.get("status") != "success":
            continue
        if idx >= len(predictions):
            continue

        pred_data = predictions[idx]
        for field_name, field_scores in result.get("result", {}).items():
            # Check if this field has an error
            is_error = False
            if "match" in field_scores and field_scores["match"] == 0:
                is_error = True
            elif "score" in field_scores and field_scores["score"] < 0.5:
                is_error = True
            elif "rougeL" in field_scores and field_scores["rougeL"] < 0.3:
                is_error = True
            elif "f1" in field_scores and field_scores["f1"] < 0.3:
                is_error = True

            if is_error:
                try:
                    pred_parsed = json.loads(pred_data["answer"]) if isinstance(pred_data["answer"], str) else pred_data["answer"]
                    gold_parsed = json.loads(pred_data["gold"]) if isinstance(pred_data["gold"], str) else pred_data["gold"]

                    error_samples.append({
                        "doc_idx": idx,
                        "field_name": field_name,
                        "field_type": schema.get(field_name, {}).get("type", "unknown"),
                        "predicted_value": str(pred_parsed.get(field_name, ""))[:500],
                        "gold_value": str(gold_parsed.get(field_name, ""))[:500],
                        "scores": field_scores,
                    })
                except (json.JSONDecodeError, TypeError, AttributeError):
                    continue

    # Sample
    random.seed(42)
    if len(error_samples) > sample_size:
        error_samples = random.sample(error_samples, sample_size)

    logger.info(f"Sampled {len(error_samples)} errors from {len(all_results)} documents")
    return error_samples


def classify_errors_with_llm(
    error_samples: list[dict[str, Any]],
    model: str = "gpt-4.1-mini",
) -> list[dict[str, Any]]:
    """Use LLM to classify each error into one of 5 categories."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set. Using heuristic classification.")
        return classify_errors_heuristic(error_samples)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    system_prompt = f"""You are an error analysis expert for information extraction systems.
Classify each extraction error into exactly ONE of these categories:

{json.dumps(ERROR_CATEGORIES, indent=2)}

Return JSON with: {{"category": "<category_name>", "confidence": <0-1>, "reasoning": "<brief explanation>"}}"""

    classified = []
    for i, sample in enumerate(error_samples):
        if i % 10 == 0:
            logger.info(f"Classifying error {i}/{len(error_samples)}...")

        user_prompt = f"""Field: {sample['field_name']} (type: {sample['field_type']})
Predicted: {sample['predicted_value']}
Gold (correct): {sample['gold_value']}

Classify this extraction error."""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            classification = json.loads(response.choices[0].message.content)
            sample["error_category"] = classification.get("category", "unknown")
            sample["classification_confidence"] = classification.get("confidence", 0)
            sample["classification_reasoning"] = classification.get("reasoning", "")
        except Exception as e:
            logger.error(f"Error classifying sample {i}: {e}")
            sample["error_category"] = "unknown"
            sample["classification_confidence"] = 0
            sample["classification_reasoning"] = str(e)

        classified.append(sample)

    return classified


def classify_errors_heuristic(
    error_samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Heuristic error classification (fallback when no API key)."""
    for sample in error_samples:
        pred = sample["predicted_value"]
        gold = sample["gold_value"]
        ftype = sample["field_type"]

        if not pred or pred in ("None", "null", ""):
            sample["error_category"] = "omission_error"
        elif ftype == "date" and pred.replace("-", "") == gold.replace("-", "").replace(".", ""):
            sample["error_category"] = "format_error"
        elif ftype == "enum" and pred.lower().strip() not in gold.lower().strip():
            sample["error_category"] = "hallucination_error"
        elif len(pred) > len(gold) * 3:
            sample["error_category"] = "granularity_error"
        elif len(pred) < len(gold) * 0.3:
            sample["error_category"] = "granularity_error"
        else:
            sample["error_category"] = "interpretation_error"

        sample["classification_confidence"] = 0.5
        sample["classification_reasoning"] = "heuristic"

    return error_samples


def main(
    results_dir: Path = typer.Argument(
        ..., help="Results directory with scores and predictions"
    ),
    sample_size: int = typer.Option(50, help="Number of errors to sample"),
    classifier_model: str = typer.Option(
        "gpt-4.1-mini", help="LLM for error classification"
    ),
    output: Path = typer.Option(
        "results/neurips/exp7_error_taxonomy.json",
    ),
):
    """Sample and categorize extraction errors into 5 error types."""
    # Find score files and predictions
    all_classified = []
    model_error_counts = defaultdict(lambda: Counter())

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        predictions_file = model_dir / "predictions.json"
        config_file = model_dir / "config.yaml"

        if not predictions_file.exists() or not config_file.exists():
            # Check parent
            config_file = results_dir / "config.yaml"
            if not config_file.exists():
                continue

        with open(config_file) as f:
            config = yaml.safe_load(f)
        schema = config.get("ie_schema", config)

        # Find scores file
        scores_files = list(model_dir.glob("scores_*.json"))
        if not scores_files:
            continue

        for scores_file in scores_files:
            logger.info(f"Processing {model_dir.name}/{scores_file.name}...")

            samples = sample_errors(
                scores_file=scores_file,
                predictions_file=predictions_file,
                schema=schema,
                sample_size=sample_size,
            )

            if not samples:
                logger.info(f"  No errors found")
                continue

            classified = classify_errors_with_llm(samples, classifier_model)

            for c in classified:
                c["model"] = model_dir.name
                c["eval_file"] = scores_file.name
                model_error_counts[model_dir.name][c["error_category"]] += 1

            all_classified.extend(classified)

    if not all_classified:
        logger.error("No errors found to classify")
        raise typer.Exit(1)

    # Aggregate
    overall_counts = Counter(c["error_category"] for c in all_classified)
    total = len(all_classified)

    # Per field type
    type_counts = defaultdict(Counter)
    for c in all_classified:
        type_counts[c["field_type"]][c["error_category"]] += 1

    results = {
        "experiment": "error_taxonomy",
        "total_errors_analyzed": total,
        "classifier_model": classifier_model,
        "error_categories": ERROR_CATEGORIES,
        "overall_distribution": {
            cat: {
                "count": count,
                "pct": round(count / total * 100, 1),
            }
            for cat, count in overall_counts.most_common()
        },
        "per_model": {
            model: dict(counts) for model, counts in model_error_counts.items()
        },
        "per_field_type": {
            ftype: dict(counts) for ftype, counts in type_counts.items()
        },
        "samples": [
            {
                "field": c["field_name"],
                "model": c["model"],
                "category": c["error_category"],
                "predicted": c["predicted_value"][:200],
                "gold": c["gold_value"][:200],
                "reasoning": c.get("classification_reasoning", ""),
            }
            for c in all_classified[:20]  # Save first 20 as examples
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, output, ensure_ascii=False)
    logger.info(f"Results saved to {output}")

    # Print summary
    print(f"\n{'='*60}")
    print("ERROR TAXONOMY ANALYSIS")
    print(f"{'='*60}")
    print(f"Total errors analyzed: {total}")
    print(f"\n{'Category':<25} {'Count':>6} {'%':>6}")
    print(f"{'-'*40}")
    for cat, count in overall_counts.most_common():
        print(f"{cat:<25} {count:>6} {count/total*100:>5.1f}%")

    if model_error_counts:
        print(f"\nPer-model breakdown:")
        for model, counts in model_error_counts.items():
            total_model = sum(counts.values())
            print(f"\n  {model} ({total_model} errors):")
            for cat, count in counts.most_common():
                print(f"    {cat:<25} {count:>4}")


if __name__ == "__main__":
    typer.run(main)
