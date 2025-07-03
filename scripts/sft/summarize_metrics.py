import json
from pathlib import Path
from statistics import mean
from typing import Any, Literal

import pandas as pd
import typer
from loguru import logger


def main(
    root_dir: Path = typer.Option(...),
) -> None:
    summarize(root_dir, "metrics.json", "ngram")
    summarize(root_dir, "llm_as_judge*.json", "judge")


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
