import json
import re
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Literal

import numpy as np
import pandas as pd
import typer
from dateutil import parser as date_parser
from langchain_core.utils.json import parse_json_markdown
from loguru import logger

from juddges.llm_as_judge.data_model import PredictionLoader

ISO_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
SEED_DIR_PATTERN = re.compile(r"seed_\d+")
SEED_METRICS = [
    "date_match_iso",
    "date_match_iso_nonnull_gold",
    "exact_match_categorical",
    "exact_match_categorical_nonnull_gold",
    "judge_average",
]


def main(
    root_dir: Path = typer.Option(...),
    judge_model: str | None = typer.Option(
        None, envvar="JUDGE_MODEL", help="Judge whose llm_as_judge_<name>.json files are used"
    ),
) -> None:
    summarize(root_dir, "scores_ngram.json", "ngram")
    if judge_model:
        summarize(root_dir, f"llm_as_judge_{judge_model}.json", "judge")
    summarize_extraction(root_dir, judge_model)


def summarize(root_dir: Path, file_pattern: str, metric_type: Literal["judge", "ngram"]) -> None:
    results = []
    for llm_dir in Path(root_dir).iterdir():
        if not llm_dir.is_dir():
            continue
        llm_results = []
        llm_name = llm_dir.name

        metric_files = list(llm_dir.rglob(file_pattern))
        logger.info(f"Found {len(metric_files)} {metric_type} metrics for {llm_name}")

        for f in metric_files:
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
        raise RuntimeError(f"No {metric_type} results ({file_pattern}) under {root_dir}")
    summary_file = root_dir / f"metrics_{metric_type}_summary.md"

    pd.concat(results).reset_index().sort_values("llm").to_markdown(summary_file, index=False)


def summarize_extraction(root_dir: Path, judge_model: str | None = None) -> None:
    """Per-run parse errors, date/categorical match and judge score with 95% CI, plus seed spread.

    A run is a directory with predictions.json and config.yaml (as written by predict_vllm.py).
    See `summarize_run` for the scoring rules; `metrics_extraction_seed_summary.md` reports the
    mean and std of the per-run values across seeds (runs differing only in `seed_<n>`).
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
                **summarize_run(run_dir, judge_model),
            }
        )

    if not runs:
        raise RuntimeError(f"No runs (predictions.json + config.yaml) under {root_dir}")

    seed_summary = summarize_seeds(runs)
    with open(root_dir / "metrics_extraction_runs.json", "w") as f:
        json.dump(runs, f, indent="\t", ensure_ascii=False)
    with open(root_dir / "metrics_extraction_seed_summary.json", "w") as f:
        json.dump(seed_summary, f, indent="\t", ensure_ascii=False)

    columns = [
        "llm",
        "run_dir",
        "num_docs",
        "parsing_errors",
        "finish_reason_length",
        "failed_or_truncated",
        "date_match_iso",
        "date_match_iso_nonnull_gold",
        "exact_match_categorical",
        "exact_match_categorical_nonnull_gold",
        "exact_match_ci95",
        "judge_average",
        "judge_ci95",
    ]
    df = pd.DataFrame(runs)[columns]
    df["exact_match_ci95"] = df["exact_match_ci95"].map(_format_ci)
    df["judge_ci95"] = df["judge_ci95"].map(_format_ci)
    df.to_markdown(root_dir / "metrics_extraction_summary.md", index=False)

    seed_df = pd.DataFrame(seed_summary)
    for metric in SEED_METRICS:
        seed_df[metric] = [_format_mean_std(row[metric]) for row in seed_summary]
    seed_df.to_markdown(root_dir / "metrics_extraction_seed_summary.md", index=False)


def summarize_run(run_dir: Path, judge_model: str | None = None) -> dict[str, Any]:
    """Scores one run (predictions.json, config.yaml, optional llm_as_judge_<judge>.json).

    Per document and field the score is 0/1. A parse error (invalid JSON or a JSON value that is
    not an object) and a missing key score 0. Dates match when both normalise to the same ISO
    `YYYY-MM-DD` (see `normalize_iso_date`); categorical fields match on equality. A null gold
    with a null prediction counts as a match in the all-fields metrics (`date_match_iso`,
    `exact_match_categorical`); the `*_nonnull_gold` variants use only documents whose gold value
    for the field is not null (a parse error still scores 0 there).
    """
    with open(run_dir / "predictions.json") as f:
        predictions = json.load(f)
    loader = PredictionLoader(run_dir)
    schema = loader.schema
    parsed = loader.load_predictions(schema, predictions)
    golds = [parse_json_markdown(p["gold"]) for p in predictions]

    date_fields = [k for k, v in schema.items() if _is_date_field(v)]
    categorical_fields = [k for k, v in schema.items() if v["type"] == "enum"]
    fields = date_fields + categorical_fields

    # rows: documents, columns: fields
    scores: dict[str, list[int]] = {field: [] for field in fields}
    nonnull_gold: dict[str, list[bool]] = {field: [] for field in fields}
    for idx, gold in enumerate(golds):
        pred = parsed.predictions.get(idx)
        for field in fields:
            nonnull_gold[field].append(gold.get(field) is not None)
            if pred is None or field not in pred:  # parsing error or missing key (as in evaluator)
                scores[field].append(0)
            elif field in date_fields:
                scores[field].append(_date_match_iso(pred[field], gold.get(field)))
            else:
                scores[field].append(int(pred[field] == gold.get(field)))

    def nonnull_means(field_names: list[str]) -> dict[str, float]:
        return {
            k: float(np.mean(np.asarray(scores[k])[nonnull_gold[k]]))
            for k in field_names
            if any(nonnull_gold[k])
        }

    closed_fields = [k for k, v in schema.items() if v["type"] in ("enum", "boolean")]
    truncated = [p.get("finish_reason") == "length" for p in predictions]
    null_rate: dict[str, float] = {}
    majority: dict[str, float] = {}
    for field in closed_fields:
        gold_vals = [json.dumps(g.get(field)) for g in golds]
        majority[field] = max(gold_vals.count(v) for v in set(gold_vals)) / len(gold_vals)
        pred_vals = [parsed.predictions.get(i) for i in range(len(golds))]
        null_rate[field] = float(np.mean([pr is not None and pr.get(field) is None for pr in pred_vals]))

    date_nonnull = nonnull_means(date_fields)
    categorical_nonnull = nonnull_means(categorical_fields)
    per_doc_exact = np.mean(list(scores.values()), axis=0)
    result: dict[str, Any] = {
        "num_docs": len(predictions),
        "parsing_errors": len(parsed.errors),
        "finish_reason_length": sum(p.get("finish_reason") == "length" for p in predictions),
        "failed_or_truncated": len(set(parsed.errors) | {i for i, x in enumerate(truncated) if x}),
        "pred_null_rate_closed_fields": null_rate,
        "majority_baseline_in_sample": majority,
        "per_field_exact_match": {k: float(np.mean(v)) for k, v in scores.items()},
        "per_field_exact_match_nonnull_gold": {**date_nonnull, **categorical_nonnull},
        "date_match_iso": _mean_or_none([np.mean(scores[k]) for k in date_fields]),
        "date_match_iso_nonnull_gold": _mean_or_none(list(date_nonnull.values())),
        "exact_match_categorical": _mean_or_none([np.mean(scores[k]) for k in categorical_fields]),
        "exact_match_categorical_nonnull_gold": _mean_or_none(list(categorical_nonnull.values())),
        "per_doc_exact_match": [float(x) for x in per_doc_exact],
        "exact_match_ci95": bootstrap_ci(per_doc_exact),
        "judge_average": None,
        "judge_ci95": None,
    }

    judge_file = _select_judge_file(run_dir, judge_model)
    if judge_file is not None:
        all_results = load_judge_file(judge_file)["all_results"]
        if len(all_results) != len(predictions):
            raise ValueError(
                f"{judge_file} scores {len(all_results)} documents, expected {len(predictions)}"
            )
        per_doc_judge = [mean(doc_scores.values()) for doc_scores in all_results.values()]
        result["judge_file"] = judge_file.name
        result["judge_average"] = float(np.mean(per_doc_judge))
        result["judge_ci95"] = bootstrap_ci(per_doc_judge)
        result["per_doc_judge"] = per_doc_judge
    return result


def summarize_seeds(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean and sample std (ddof=1, null for one seed) of the per-run metrics across seeds."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        run_path = Path(run["run_dir"])
        if SEED_DIR_PATTERN.fullmatch(run_path.name):
            groups.setdefault(str(run_path.parent), []).append(run)
        else:
            groups.setdefault(str(run_path), []).append(run)

    rows = []
    for config_dir, group in groups.items():
        row: dict[str, Any] = {
            "llm": group[0]["llm"],
            "config_dir": config_dir,
            "num_seeds": len(group),
        }
        for metric in SEED_METRICS:
            values = [r[metric] for r in group if r.get(metric) is not None]
            row[metric] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
        rows.append(row)
    return rows


def paired_bootstrap(
    per_doc_a: Any,
    per_doc_b: Any,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired bootstrap of mean(b) - mean(a) over documents shared by both runs.

    `per_doc_a` and `per_doc_b` are per-document scores of two runs (e.g. zero-shot and SFT,
    `per_doc_exact_match` in metrics_extraction_runs.json) in the same document order. The same
    resampled document indices are used for both, so document difficulty cancels out.
    """
    a = np.asarray(per_doc_a, dtype=float)
    b = np.asarray(per_doc_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"Need equal-length 1-d per-document scores, got {a.shape} and {b.shape}")
    diff = b - a
    return {
        "mean_diff": float(diff.mean()),
        "ci95": bootstrap_ci(diff, n_boot=n_boot, seed=seed, confidence=confidence),
    }


def _select_judge_file(run_dir: Path, judge_model: str | None) -> Path | None:
    if judge_model:
        judge_file = run_dir / f"llm_as_judge_{judge_model}.json"
        if not judge_file.exists():
            raise FileNotFoundError(f"Judge file {judge_file} does not exist")
        return judge_file
    judge_files = sorted(run_dir.glob("llm_as_judge*.json"))
    if len(judge_files) > 1:
        raise ValueError(f"Several judge files in {run_dir}, select one with --judge-model")
    return judge_files[0] if judge_files else None


def load_judge_file(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not data.get("parse_errors_scored_as_zero"):
        raise ValueError(
            f"{path} predates scoring parse errors as zero (no parse_errors_scored_as_zero); "
            "rerun scripts/sft/llm_as_judge.py"
        )
    return data


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


def normalize_iso_date(value: Any) -> str | None:
    """Normalises a complete date to `YYYY-MM-DD`; None for partial dates and non-dates.

    ISO strings are taken as they are (dateutil's `dayfirst` would swap their month and day);
    anything else is parsed with `dayfirst=True`. A date missing the year, month or day is
    rejected: parsing against two different defaults detects components filled from a default.
    """
    if not isinstance(value, str):
        return None
    value = value.strip()
    try:
        if ISO_DATE_PATTERN.fullmatch(value):
            return date.fromisoformat(value).isoformat()
        first = date_parser.parse(value, dayfirst=True, default=datetime(2001, 1, 1))
        second = date_parser.parse(value, dayfirst=True, default=datetime(2002, 2, 2))
    except (ValueError, OverflowError):
        return None
    if first != second:
        return None
    return first.date().isoformat()


def _date_match_iso(predicted: Any, gold: Any) -> int:
    if predicted is None and gold is None:
        return 1
    normalized = normalize_iso_date(predicted)
    return int(normalized is not None and normalized == normalize_iso_date(gold))


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _format_mean_std(stats: dict[str, float | None]) -> str | None:
    if stats["mean"] is None:
        return None
    std = "n/a" if stats["std"] is None else f"{stats['std']:0.3f}"
    return f"{stats['mean']:0.3f} ± {std}"


def _format_ci(ci: tuple[float, float] | None) -> str | None:
    return None if ci is None else f"[{ci[0]:0.3f}, {ci[1]:0.3f}]"


def parse_ngram_metrics(path: Path, **metadata: Any) -> list[dict[str, float]]:
    """Reads scores_ngram.json (ngram_based_eval.py): mean score of every field and score key."""
    with path.open() as file:
        metrics = json.load(file)
    return [
        {
            **metadata,
            **{
                f"{field}/{score_name}": score["mean_score"]
                for field, field_scores in metrics["aggregated_scores"].items()
                for score_name, score in field_scores.items()
            },
        }
    ]


def parse_judge_metrics(path: Path, **metadata: Any) -> dict[str, float]:
    metrics = load_judge_file(path)

    return {
        **metadata,
        "average_score": mean(metrics["aggregated_results"].values()),
        "parsing_errors": metrics["stats"]["num_parsing_errors"],
        **metrics["aggregated_results"],
    }


if __name__ == "__main__":
    typer.run(main)
