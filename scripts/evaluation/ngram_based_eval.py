from pathlib import Path

import typer
from loguru import logger

from juddges.evals.extraction import ExtractionEvaluator
from juddges.llm_as_judge.data_model import PredictionLoader
from juddges.utils.misc import save_json


def main(
    predictions_dir: Path = typer.Argument(...),
):
    """
    Evaluates the quality of extracted information from LLM predictions.
    """
    pred_loader = PredictionLoader(root_dir=predictions_dir, judge_name=None)
    predictions = pred_loader.load_predictions(verbose=True)

    schema = pred_loader.schema

    # The schema might be nested under a top-level key
    if len(schema) == 1:
        schema = next(iter(schema.values()))

    evaluator = ExtractionEvaluator(schema)
    results = evaluator.run(predictions)

    logger.info(f"Saving evaluation results to {pred_loader.ngram_scores_file}")
    pred_loader.ngram_scores_file.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, pred_loader.ngram_scores_file)

    logger.info("Evaluation finished.")
    logger.info(
        f"Summary: {results['summary_metrics']['evaluated_records']}/{results['summary_metrics']['total_records']} records evaluated."
    )


if __name__ == "__main__":
    typer.run(main)
