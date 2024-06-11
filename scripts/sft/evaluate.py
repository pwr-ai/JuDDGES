import json
from pathlib import Path
import typer

from juddges.metrics.info_extraction import evaluate_extraction


def evaluate_results(output_file: Path = typer.Option(...)) -> None:
    with open(output_file, "r") as file:
        results = json.load(file)

    res = evaluate_extraction(results)
    metrics_fname = output_file.name.split("_", maxsplit=1)[1]
    metrics_file = output_file.with_name(f"metrics_{metrics_fname}")
    with open(metrics_file, "w") as file:
        json.dump(res, file, indent="\t")


if __name__ == "__main__":
    typer.run(evaluate_results)
