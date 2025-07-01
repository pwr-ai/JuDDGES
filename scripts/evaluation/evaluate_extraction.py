import json
from pathlib import Path

import typer
import yaml
from loguru import logger

from juddges.evals.extraction import ExtractionEvaluator


def main(
    predictions_path: Path = typer.Option(
        ...,
        "--predictions",
        "-p",
        help="Path to predictions JSON file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    schema_path: Path = typer.Option(
        ...,
        "--schema",
        "-s",
        help="Path to schema YAML file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to save the evaluation results.",
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
    ),
):
    """
    Evaluates the quality of extracted information from LLM predictions.
    """
    logger.info(f"Loading predictions from {predictions_path}")
    with predictions_path.open("r") as f:
        predictions = json.load(f)

    logger.info(f"Loading schema from {schema_path}")
    with schema_path.open("r") as f:
        schema = yaml.safe_load(f)

    # The schema might be nested under a top-level key
    if len(schema) == 1:
        schema = next(iter(schema.values()))

    evaluator = ExtractionEvaluator(schema)
    results = evaluator.run(predictions)

    logger.info(f"Saving evaluation results to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent="\t", ensure_ascii=False)

    logger.info("Evaluation finished.")
    logger.info(
        f"Summary: {results['summary_metrics']['evaluated_records']}/{results['summary_metrics']['total_records']} records evaluated."
    )


if __name__ == "__main__":
    typer.run(main)
