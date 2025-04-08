import json
import os
from pathlib import Path

import typer

from juddges.evaluation.info_extraction import InfoExtractionEvaluator

NUM_PROC = int(os.environ.get("NUM_PROC", 1))


def evaluate_results(
    output_file: Path = typer.Option(...),
    num_proc: int = typer.Option(NUM_PROC),
) -> None:
    with open(output_file, "r") as file:
        results = json.load(file)

    evaluator = InfoExtractionEvaluator(num_proc=num_proc)
    results = evaluator.evaluate(results)

    metrics_fname = output_file.name.split("_", maxsplit=1)[1]
    metrics_file = output_file.with_name(f"metrics_{metrics_fname}")
    with open(metrics_file, "w") as file:
        json.dump(results, file, indent="\t")


if __name__ == "__main__":
    typer.run(evaluate_results)
