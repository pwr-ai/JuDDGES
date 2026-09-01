#!/usr/bin/env python
"""Multi-Document Aggregation Experiment (Task 3).

Tests reliability of aggregate statistics from LLM-extracted Schema B fields
at different collection sizes (50, 100, 500, 1000, 5000).

Uses pl-court-raw-enriched from HuggingFace (100K+ Gemini extractions).

Usage:
    python scripts/neurips_experiments/run_multidoc_aggregation.py
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

from datasets import load_dataset
from loguru import logger

random.seed(42)


def compute_ground_truth(data: list[dict]) -> dict:
    """Compute ground truth aggregation statistics from full dataset."""
    gt = {}

    # 1. Count by year
    year_counts = Counter()
    for row in data:
        date = row.get("extracted_date_issued") or ""
        if len(date) >= 4:
            year_counts[date[:4]] += 1
    gt["count_by_year"] = dict(year_counts.most_common(20))

    # 2. Outcome distribution
    outcome_counts = Counter()
    for row in data:
        outcome = row.get("extracted_outcome")
        if isinstance(outcome, dict):
            dt = outcome.get("decision_type", "unknown")
            outcome_counts[dt] += 1
        elif isinstance(outcome, str):
            try:
                o = json.loads(outcome)
                outcome_counts[o.get("decision_type", "unknown")] += 1
            except (json.JSONDecodeError, TypeError):
                outcome_counts["parse_error"] += 1
    gt["outcome_distribution"] = dict(outcome_counts.most_common(20))

    # 3. Documents with thesis
    has_thesis = sum(1 for r in data if r.get("extracted_thesis"))
    gt["thesis_coverage"] = has_thesis / len(data) if data else 0

    # 4. Documents with factual_state
    has_fs = sum(1 for r in data if r.get("extracted_factual_state"))
    gt["factual_state_coverage"] = has_fs / len(data) if data else 0

    # 5. Average keywords count
    kw_counts = []
    for r in data:
        kw = r.get("extracted_keywords")
        if isinstance(kw, list):
            kw_counts.append(len(kw))
        elif isinstance(kw, str):
            try:
                kw_list = json.loads(kw)
                kw_counts.append(len(kw_list) if isinstance(kw_list, list) else 0)
            except (json.JSONDecodeError, TypeError):
                kw_counts.append(0)
    gt["avg_keywords"] = mean(kw_counts) if kw_counts else 0

    # 6. Top legal concepts
    concept_counter = Counter()
    for r in data:
        concepts = r.get("extracted_legal_concepts")
        if isinstance(concepts, list):
            for c in concepts:
                if isinstance(c, str):
                    concept_counter[c.lower().strip()] += 1
                elif isinstance(c, dict):
                    concept_counter[str(c).lower().strip()] += 1
        elif isinstance(concepts, str):
            try:
                cl = json.loads(concepts)
                if isinstance(cl, list):
                    for c in cl:
                        concept_counter[str(c).lower().strip()] += 1
            except (json.JSONDecodeError, TypeError):
                pass
    gt["top_concepts"] = dict(concept_counter.most_common(20))

    return gt


def evaluate_subsample(
    full_data: list[dict],
    ground_truth: dict,
    sample_size: int,
    n_trials: int = 20,
) -> dict:
    """Evaluate aggregation accuracy at given sample size."""
    metrics = defaultdict(list)

    for trial in range(n_trials):
        sample = random.sample(full_data, min(sample_size, len(full_data)))

        # Compute on subsample
        sub_gt = compute_ground_truth(sample)

        # Compare: thesis coverage (proportion query)
        metrics["thesis_coverage_error"].append(
            abs(sub_gt["thesis_coverage"] - ground_truth["thesis_coverage"])
        )

        # Compare: factual_state coverage
        metrics["factual_state_coverage_error"].append(
            abs(sub_gt["factual_state_coverage"] - ground_truth["factual_state_coverage"])
        )

        # Compare: avg keywords
        if ground_truth["avg_keywords"] > 0:
            metrics["avg_keywords_rel_error"].append(
                abs(sub_gt["avg_keywords"] - ground_truth["avg_keywords"]) / ground_truth["avg_keywords"]
            )

        # Compare: outcome distribution (top-3 KL divergence proxy)
        gt_outcomes = ground_truth["outcome_distribution"]
        sub_outcomes = sub_gt["outcome_distribution"]
        total_gt = sum(gt_outcomes.values()) or 1
        total_sub = sum(sub_outcomes.values()) or 1
        all_keys = set(gt_outcomes) | set(sub_outcomes)
        dist_error = sum(
            abs(gt_outcomes.get(k, 0) / total_gt - sub_outcomes.get(k, 0) / total_sub)
            for k in all_keys
        ) / 2  # Total variation distance
        metrics["outcome_tvd"].append(dist_error)

        # Compare: year distribution
        gt_years = ground_truth["count_by_year"]
        sub_years = sub_gt["count_by_year"]
        total_gt_y = sum(gt_years.values()) or 1
        total_sub_y = sum(sub_years.values()) or 1
        all_years = set(gt_years) | set(sub_years)
        year_tvd = sum(
            abs(gt_years.get(y, 0) / total_gt_y - sub_years.get(y, 0) / total_sub_y)
            for y in all_years
        ) / 2
        metrics["year_tvd"].append(year_tvd)

    # Aggregate trial results
    result = {}
    for metric_name, values in metrics.items():
        result[metric_name] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return result


def main():
    logger.info("Loading pl-court-raw-enriched from HuggingFace (streaming)...")

    # Load up to 10K for tractable computation
    ds = load_dataset("JuDDGES/pl-court-raw-enriched", split="train", streaming=True)

    data = []
    for i, row in enumerate(ds):
        data.append(row)
        if i >= 9999:
            break
        if (i + 1) % 2000 == 0:
            logger.info(f"Loaded {i + 1} documents...")

    logger.info(f"Loaded {len(data)} documents total")

    # Check which extracted fields exist
    sample = data[0]
    extracted_fields = [k for k in sample.keys() if k.startswith("extracted_")]
    logger.info(f"Extracted fields available: {extracted_fields}")

    # Compute ground truth on full dataset
    logger.info("Computing ground truth on full dataset...")
    ground_truth = compute_ground_truth(data)

    logger.info(f"Ground truth stats:")
    logger.info(f"  Year distribution: {dict(list(ground_truth['count_by_year'].items())[:5])}...")
    logger.info(f"  Outcome distribution: {dict(list(ground_truth['outcome_distribution'].items())[:5])}...")
    logger.info(f"  Thesis coverage: {ground_truth['thesis_coverage']:.3f}")
    logger.info(f"  Factual state coverage: {ground_truth['factual_state_coverage']:.3f}")
    logger.info(f"  Avg keywords: {ground_truth['avg_keywords']:.1f}")

    # Evaluate at different sample sizes
    sample_sizes = [50, 100, 500, 1000, 5000]
    results = {}

    for size in sample_sizes:
        if size > len(data):
            continue
        logger.info(f"\nEvaluating at sample size {size}...")
        results[size] = evaluate_subsample(data, ground_truth, size, n_trials=20)

        # Print summary
        for metric, stats in results[size].items():
            logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Print final table
    print(f"\n{'='*80}")
    print("Multi-Document Aggregation: Error vs Sample Size")
    print(f"{'='*80}")
    print(f"{'Metric':<35} ", end="")
    for size in sample_sizes:
        if size in results:
            print(f"{'n=' + str(size):>12}", end="")
    print()
    print("-" * 80)

    all_metrics = set()
    for r in results.values():
        all_metrics.update(r.keys())

    for metric in sorted(all_metrics):
        print(f"{metric:<35} ", end="")
        for size in sample_sizes:
            if size in results and metric in results[size]:
                m = results[size][metric]["mean"]
                s = results[size][metric]["std"]
                print(f"  {m:.3f}±{s:.3f}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()
    print(f"{'='*80}")

    # Save
    output_dir = Path("data/experiments/neurips_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "ground_truth": {k: v for k, v in ground_truth.items() if k != "top_concepts"},
        "ground_truth_top_concepts": ground_truth.get("top_concepts", {}),
        "sample_results": {str(k): v for k, v in results.items()},
        "n_documents": len(data),
    }
    with open(output_dir / "multidoc_aggregation.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved to {output_dir / 'multidoc_aggregation.json'}")


if __name__ == "__main__":
    main()
