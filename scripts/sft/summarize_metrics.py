import json
from pathlib import Path
from statistics import mean
from typing import Any, Literal

import numpy as np
import pandas as pd
import typer
from loguru import logger

from juddges.evals.metrics import evaluate_date
from juddges.llm_as_judge.data_model import PredictionLoader


def main(
    root_dir: Path = typer.Option(...),
) -> None:
    summarize(root_dir, "metrics.json", "ngram")
    summarize(root_dir, "llm_as_judge*.json", "judge")
    summarize_extraction(root_dir)


def summarize(root_dir: Path, file_pattern: str, metric_type: Literal["judge", "ngram"]) -> None:
    results = []
    for llm_dir in Path(root_dir).iterdir():
        llm_results = []
        llm_name = llm_dir.name

        metric_files = list(llm_dir.rglob(file_pattern))
        logger.info(f"Found {len(metric_files)} {metric_type} metrics for {llm_name}")

        for f in llm_dir.rglob(file_pattern):
            if metric_type == "ngram":
                single_trial_metrics = parse_ngram_metrics(f, llm=llm_name)
            elif metric_type == "judge":
                single_trial_metrics = [parse_judge_metrics(f, llm=llm_name)]
            else:
                raise ValueError(f"Invalid metric_type: {metric_type}")

            llm_results.extend(single_trial_metrics)

        if llm_results:
            df = pd.DataFrame(llm_results)
            group_cols = list(df.select_dtypes(include="object").columns)
            metric_mean = df.groupby(group_cols).mean()
            metric_std = df.groupby(group_cols).std()
            stats = (
                metric_mean.map(lambda x: f"{x:0.3f}")
                + " (± "
                + metric_std.map(lambda x: f"{x:0.3f}")
                + ")"
            )

            results.append(stats)

    if not results:
        logger.info(f"No results for {metric_type}")
        return
    summary_file = root_dir / f"metrics_{metric_type}_summary.md"

    pd.concat(results).reset_index().sort_values("llm").to_markdown(summary_file, index=False)


def summarize_extraction(root_dir: Path) -> None:
    """Per-run parse errors, exact match of date/categorical fields and judge score with 95% CI.

    A run is a directory with predictions.json and config.yaml (as written by predict_vllm.py).
    Documents with parsing errors count as 0 in every score.
    """
    runs = []
    for predictions_file in sorted(Path(root_dir).rglob("predictions.json")):
        run_dir = predictions_file.parent
        if not (run_dir / "config.yaml").exists():
            continue
        runs.append(
            {
                "llm": run_dir.relative_to(root_dir).parts[0],
                "run_dir": str(run_dir.relative_to(root_dir)),
                **summarize_run(run_dir),
            }
        )

    if not runs:
        logger.info("No runs for extraction summary")
        return

    with open(root_dir / "metrics_extraction_runs.json", "w") as f:
        json.dump(runs, f, indent="\t", ensure_ascii=False)

    columns = [
        "llm",
        "run_dir",
        "num_docs",
        "parsing_errors",
        "finish_reason_length",
        "exact_match_date",
        "exact_match_categorical",
        "exact_match_ci95",
        "judge_average",
        "judge_ci95",
    ]
    df = pd.DataFrame(runs)[columns]
    df["exact_match_ci95"] = df["exact_match_ci95"].map(_format_ci)
    df["judge_ci95"] = df["judge_ci95"].map(_format_ci)
    df.to_markdown(root_dir / "metrics_extraction_summary.md", index=False)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    with open(run_dir / "predictions.json") as f:
        predictions = json.load(f)
    loader = PredictionLoader(run_dir)
    schema = loader.schema
    parsed = loader.load_predictions(schema, predictions)

    date_fields = [k for k, v in schema.items() if _is_date_field(v)]
    categorical_fields = [k for k, v in schema.items() if v["type"] == "enum"]

    # rows: documents, columns: fields; parsing errors score 0
    scores = {field: [] for field in date_fields + categorical_fields}
    for idx in range(len(predictions)):
        pred, gold = parsed.predictions.get(idx), parsed.gold.get(idx)
        for field in scores:
            if pred is None or field not in pred:  # parsing error or missing key (as in evaluator)
                scores[field].append(0)
            elif field in date_fields:
                scores[field].append(evaluate_date(pred.get(field), gold.get(field))["match"])
            else:
                scores[field].append(int(pred.get(field) == gold.get(field)))

    per_doc_exact = np.mean(list(scores.values()), axis=0)
    result: dict[str, Any] = {
        "num_docs": len(predictions),
        "parsing_errors": len(parsed.errors),
        "finish_reason_length": sum(p.get("finish_reason") == "length" for p in predictions),
        "per_field_exact_match": {k: float(np.mean(v)) for k, v in scores.items()},
        "exact_match_date": _mean_or_none([np.mean(scores[k]) for k in date_fields]),
        "exact_match_categorical": _mean_or_none([np.mean(scores[k]) for k in categorical_fields]),
        "exact_match_ci95": bootstrap_ci(per_doc_exact),
        "judge_average": None,
        "judge_ci95": None,
    }

    judge_files = sorted(run_dir.glob("llm_as_judge*.json"))
    if judge_files:
        with open(judge_files[0]) as f:
            all_results = json.load(f)["all_results"]
        per_doc_judge = [mean(doc_scores.values()) for doc_scores in all_results.values()]
        result["judge_file"] = judge_files[0].name
        result["judge_average"] = float(np.mean(per_doc_judge))
        result["judge_ci95"] = bootstrap_ci(per_doc_judge)
    return result


def bootstrap_ci(
    per_document_values: Any, n_boot: int = 1000, seed: int = 0, confidence: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling documents with a fixed seed."""
    values = np.asarray(per_document_values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lower, upper = np.percentile(means, [50 * (1 - confidence), 50 * (1 + confidence)])
    return float(lower), float(upper)


def _is_date_field(spec: dict[str, Any]) -> bool:
    # date fields are declared as strings; their description gives the YYYY-MM-DD format
    return spec["type"] == "date" or (
        spec["type"] == "string" and "YYYY-MM-DD" in spec.get("description", "")
    )


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _format_ci(ci: tuple[float, float] | None) -> str | None:
    return None if ci is None else f"[{ci[0]:0.3f}, {ci[1]:0.3f}]"


def parse_ngram_metrics(path: Path, **metadata: Any) -> list[dict[str, float]]:
    with path.open() as file:
        metrics = json.load(file)
    return [
        {
            **metadata,
            "full_text_chrf": metrics["full_text_chrf"],
            **metrics["field_chrf"],
        }
    ]


def parse_judge_metrics(path: Path, **metadata: Any) -> list[dict[str, float]]:
    with path.open() as file:
        metrics = json.load(file)

    return {
        **metadata,
        "average_score": mean(metrics["aggregated_results"].values()),
        "parsing_errors": metrics["stats"]["num_parsing_errors"],
        **metrics["aggregated_results"],
    }


if __name__ == "__main__":
    typer.run(main)
