import json
import os
from pathlib import Path

import typer

from juddges.evaluation.info_extraction import InfoExtractionEvaluator

NUM_PROC = int(os.environ.get("NUM_PROC", 1))


def evaluate_results(
    output_file: Path = typer.Option(...),
    format: str = typer.Option("json"),
    num_proc: int = typer.Option(NUM_PROC),
) -> None:
    with open(output_file, "r") as file:
        results = json.load(file)

    evaluator = InfoExtractionEvaluator(num_proc=num_proc, format=format)
    results = evaluator.evaluate(results)

    metrics_file = output_file.with_name("metrics.json")
    with open(metrics_file, "w") as file:
        json.dump(results, file, ensure_ascii=False, indent="\t")


if __name__ == "__main__":
    typer.run(evaluate_results)
