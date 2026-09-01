#!/usr/bin/env python
"""Experiment 5: Multi-Document Aggregation (Task 3).

Tests whether aggregate statistics computed from LLM-extracted fields
are reliable at different document collection sizes (50, 100, 500, 1000).

Defines 100 aggregation queries over Schema B fields, computes ground truth
from the full Gemini extraction corpus, then evaluates accuracy of
aggregation from subsamples.

Query types:
  - Counting: "How many judgments from Sad Apelacyjny in 2023?"
  - Proportion: "What % of cases were upheld?"
  - Distribution: "Top 5 most-cited legal provisions"
  - Temporal: "Average case count per month in 2022"
  - Cross-tabulation: "Outcome distribution by court type"

Usage:
    python scripts/neurips_experiments/exp5_multidoc_aggregation.py \
        --enriched-dataset JuDDGES/pl-court-raw-enriched \
        --output results/neurips/exp5_multidoc_aggregation.json
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

import polars as pl
import typer
from loguru import logger

from juddges.utils.misc import save_json


def define_aggregation_queries() -> list[dict[str, Any]]:
    """Define 100 aggregation queries over Schema B fields."""
    queries = []

    # --- Counting queries (20) ---
    for year in range(2018, 2025):
        queries.append({
            "id": f"count_year_{year}",
            "type": "count",
            "question": f"How many judgments were issued in {year}?",
            "field": "extracted_date_issued",
            "filter": lambda row, y=year: (
                row.get("extracted_date_issued") or ""
            ).startswith(str(y)),
        })

    queries.append({
        "id": "count_with_summary",
        "type": "count",
        "question": "How many documents have a non-empty summary?",
        "field": "extracted_summary",
        "filter": lambda row: bool(row.get("extracted_summary")),
    })

    queries.append({
        "id": "count_with_thesis",
        "type": "count",
        "question": "How many documents have a thesis?",
        "field": "extracted_thesis",
        "filter": lambda row: bool(row.get("extracted_thesis")),
    })

    queries.append({
        "id": "count_with_factual_state",
        "type": "count",
        "question": "How many documents have factual_state extracted?",
        "field": "factual_state",
        "filter": lambda row: bool(row.get("factual_state")),
    })

    # --- Proportion queries (20) ---
    outcome_types = [
        "uwzgledniono_w_calosci", "uwzgledniono_w_czesci", "oddalono",
        "umorzono", "uchylono", "uchylono_i_przekazano", "zmieniono",
        "utrzymano_w_mocy", "odrzucono",
    ]
    for ot in outcome_types:
        queries.append({
            "id": f"proportion_outcome_{ot}",
            "type": "proportion",
            "question": f"What proportion of judgments have outcome '{ot}'?",
            "field": "extracted_outcome",
            "filter": lambda row, o=ot: o in str(row.get("extracted_outcome", "")),
        })

    # Document type proportions
    for dt in ["judgment", "tax_interpretation", "legal_act"]:
        queries.append({
            "id": f"proportion_doctype_{dt}",
            "type": "proportion",
            "question": f"What proportion are {dt}s?",
            "field": "document_type",
            "filter": lambda row, d=dt: str(row.get("document_type", "")).lower() == d.lower()
            or d in str(row.get("extracted_outcome", "")),
        })

    # --- Distribution queries (20) ---
    queries.append({
        "id": "dist_keywords_top10",
        "type": "distribution",
        "question": "What are the top 10 most frequent keywords?",
        "field": "extracted_keywords",
        "aggregator": "top_keywords",
    })

    queries.append({
        "id": "dist_outcomes",
        "type": "distribution",
        "question": "Distribution of decision types",
        "field": "extracted_outcome",
        "aggregator": "outcome_distribution",
    })

    for year in range(2019, 2025):
        queries.append({
            "id": f"dist_monthly_{year}",
            "type": "temporal",
            "question": f"Monthly document count in {year}",
            "field": "extracted_date_issued",
            "aggregator": "monthly_counts",
            "params": {"year": year},
        })

    # --- Text length queries (20) ---
    for field in ["extracted_summary", "extracted_thesis", "factual_state", "legal_state"]:
        queries.append({
            "id": f"avg_length_{field}",
            "type": "statistic",
            "question": f"Average character length of {field}",
            "field": field,
            "aggregator": "avg_length",
        })
        queries.append({
            "id": f"median_length_{field}",
            "type": "statistic",
            "question": f"Median character length of {field}",
            "field": field,
            "aggregator": "median_length",
        })

    # Fill to ~100 with keyword co-occurrence queries
    for kw in ["kredyt frankowy", "klauzule abuzywne", "VAT", "CIT", "prawo cywilne",
                "Sąd Najwyższy", "TSUE", "konsument", "nieważność", "odszkodowanie",
                "apelacja", "kasacja"]:
        queries.append({
            "id": f"count_keyword_{kw.replace(' ', '_')}",
            "type": "count",
            "question": f"How many documents mention keyword '{kw}'?",
            "field": "extracted_keywords",
            "filter": lambda row, k=kw: k.lower() in str(row.get("extracted_keywords", "")).lower(),
        })

    return queries[:100]  # Cap at 100


def execute_count_query(
    data: list[dict[str, Any]],
    query: dict[str, Any],
) -> float:
    """Execute a count/proportion query on a subset of data."""
    filter_fn = query["filter"]
    matches = sum(1 for row in data if filter_fn(row))

    if query["type"] == "proportion":
        return matches / len(data) if len(data) > 0 else 0.0
    return float(matches)


def execute_distribution_query(
    data: list[dict[str, Any]],
    query: dict[str, Any],
) -> dict[str, int]:
    """Execute a distribution query."""
    aggregator = query.get("aggregator", "")

    if aggregator == "top_keywords":
        all_keywords = []
        for row in data:
            kw_str = row.get("extracted_keywords", "")
            if kw_str:
                try:
                    kws = json.loads(kw_str) if isinstance(kw_str, str) else kw_str
                    if isinstance(kws, list):
                        all_keywords.extend(kws)
                except (json.JSONDecodeError, TypeError):
                    pass
        return dict(Counter(all_keywords).most_common(10))

    elif aggregator == "outcome_distribution":
        outcomes = []
        for row in data:
            outcome_str = str(row.get("extracted_outcome", ""))
            for ot in ["uwzgledniono_w_calosci", "uwzgledniono_w_czesci", "oddalono",
                        "umorzono", "uchylono", "zmieniono", "utrzymano_w_mocy", "odrzucono"]:
                if ot in outcome_str:
                    outcomes.append(ot)
                    break
        return dict(Counter(outcomes).most_common())

    elif aggregator == "monthly_counts":
        year = query.get("params", {}).get("year", 2023)
        monthly = Counter()
        for row in data:
            date_str = row.get("extracted_date_issued", "")
            if date_str and str(date_str).startswith(str(year)):
                try:
                    month = int(date_str[5:7])
                    monthly[month] += 1
                except (ValueError, IndexError):
                    pass
        return {str(m): monthly.get(m, 0) for m in range(1, 13)}

    return {}


def execute_statistic_query(
    data: list[dict[str, Any]],
    query: dict[str, Any],
) -> float:
    """Execute a statistical aggregation query."""
    field = query["field"]
    aggregator = query.get("aggregator", "avg_length")

    lengths = []
    for row in data:
        val = row.get(field)
        if val and isinstance(val, str):
            lengths.append(len(val))

    if not lengths:
        return 0.0

    if aggregator == "avg_length":
        return mean(lengths)
    elif aggregator == "median_length":
        sorted_l = sorted(lengths)
        mid = len(sorted_l) // 2
        return float(sorted_l[mid])
    return 0.0


def compute_aggregation_error(
    ground_truth: float | dict,
    estimated: float | dict,
) -> float:
    """Compute relative error between ground truth and estimated aggregation."""
    if isinstance(ground_truth, dict) and isinstance(estimated, dict):
        # For distributions, compute normalized L1 distance
        all_keys = set(list(ground_truth.keys()) + list(estimated.keys()))
        if not all_keys:
            return 0.0
        total_gt = sum(ground_truth.values()) or 1
        total_est = sum(estimated.values()) or 1
        l1 = sum(
            abs(ground_truth.get(k, 0) / total_gt - estimated.get(k, 0) / total_est)
            for k in all_keys
        )
        return l1 / 2  # Normalize to [0, 1]

    if isinstance(ground_truth, (int, float)) and isinstance(estimated, (int, float)):
        if ground_truth == 0:
            return 0.0 if estimated == 0 else 1.0
        return abs(ground_truth - estimated) / abs(ground_truth)

    return 1.0


def main(
    enriched_dataset: str = typer.Option(
        "JuDDGES/pl-court-raw-enriched",
        help="HuggingFace enriched dataset",
    ),
    max_docs: int = typer.Option(
        0, help="Max documents to load (0 = all)"
    ),
    n_trials: int = typer.Option(
        10, help="Number of random subsample trials per size"
    ),
    output: Path = typer.Option(
        "results/neurips/exp5_multidoc_aggregation.json",
    ),
):
    """Evaluate multi-document aggregation reliability at different scales."""
    from datasets import load_dataset

    logger.info(f"Loading {enriched_dataset}...")
    ds = load_dataset(enriched_dataset, split="train")
    if max_docs > 0:
        ds = ds.select(range(min(max_docs, len(ds))))

    # Convert to list of dicts for query execution
    data = [ds[i] for i in range(len(ds))]
    logger.info(f"Loaded {len(data)} documents")

    # Define queries
    queries = define_aggregation_queries()
    logger.info(f"Defined {len(queries)} aggregation queries")

    # Compute ground truth on FULL dataset
    logger.info("Computing ground truth on full dataset...")
    ground_truth = {}
    for query in queries:
        qid = query["id"]
        if query["type"] in ("count", "proportion"):
            ground_truth[qid] = execute_count_query(data, query)
        elif query["type"] in ("distribution", "temporal"):
            ground_truth[qid] = execute_distribution_query(data, query)
        elif query["type"] == "statistic":
            ground_truth[qid] = execute_statistic_query(data, query)

    # Evaluate at different subsample sizes
    subsample_sizes = [50, 100, 200, 500, 1000]
    subsample_sizes = [s for s in subsample_sizes if s <= len(data)]

    random.seed(42)
    scale_results = {}

    for size in subsample_sizes:
        logger.info(f"Evaluating at subsample size {size}...")
        per_query_errors = defaultdict(list)

        for trial in range(n_trials):
            sample = random.sample(data, size)

            for query in queries:
                qid = query["id"]
                gt = ground_truth[qid]

                if query["type"] in ("count", "proportion"):
                    estimated = execute_count_query(sample, query)
                    if query["type"] == "count":
                        # Scale count to full dataset size
                        estimated = estimated * len(data) / size
                elif query["type"] in ("distribution", "temporal"):
                    estimated = execute_distribution_query(sample, query)
                elif query["type"] == "statistic":
                    estimated = execute_statistic_query(sample, query)
                else:
                    continue

                error = compute_aggregation_error(gt, estimated)
                per_query_errors[qid].append(error)

        # Aggregate errors per query type
        type_errors = defaultdict(list)
        for qid, errors in per_query_errors.items():
            q = next(q for q in queries if q["id"] == qid)
            mean_err = mean(errors)
            type_errors[q["type"]].append(mean_err)

        scale_results[size] = {
            "per_query": {
                qid: {
                    "mean_error": round(mean(errors), 4),
                    "std_error": round(stdev(errors), 4) if len(errors) > 1 else 0.0,
                }
                for qid, errors in per_query_errors.items()
            },
            "per_type": {
                qtype: {
                    "mean_error": round(mean(errors), 4),
                    "num_queries": len(errors),
                }
                for qtype, errors in type_errors.items()
            },
            "overall_mean_error": round(
                mean([mean(e) for e in per_query_errors.values()]), 4
            ),
        }

    results = {
        "experiment": "multidoc_aggregation",
        "dataset": enriched_dataset,
        "total_documents": len(data),
        "num_queries": len(queries),
        "num_trials": n_trials,
        "subsample_sizes": subsample_sizes,
        "scale_results": scale_results,
        "ground_truth_summary": {
            qid: (
                round(gt, 4) if isinstance(gt, (int, float))
                else f"dict({len(gt)} keys)"
            )
            for qid, gt in list(ground_truth.items())[:20]
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, output, ensure_ascii=False)
    logger.info(f"Results saved to {output}")

    # Print summary
    print(f"\n{'='*70}")
    print("MULTI-DOCUMENT AGGREGATION RESULTS")
    print(f"{'='*70}")
    print(f"Total documents: {len(data)}")
    print(f"Queries: {len(queries)}")
    print(f"\n{'Size':>6} {'Overall Error':>14} {'Count':>8} {'Proportion':>12} {'Statistic':>10}")
    print(f"{'-'*55}")
    for size in subsample_sizes:
        sr = scale_results[size]
        pt = sr["per_type"]
        print(
            f"{size:>6} {sr['overall_mean_error']:>14.4f}"
            f" {pt.get('count', {}).get('mean_error', 0):>8.4f}"
            f" {pt.get('proportion', {}).get('mean_error', 0):>12.4f}"
            f" {pt.get('statistic', {}).get('mean_error', 0):>10.4f}"
        )


if __name__ == "__main__":
    typer.run(main)
