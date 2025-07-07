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
    parsed_preds = pred_loader.load_predictions_from_file(verbose=True)

    evaluator = ExtractionEvaluator(pred_loader.schema)
    results = evaluator.run(parsed_preds)

    save_json(results.model_dump(), pred_loader.ngram_scores_file, ensure_ascii=False)
    logger.info(f"Saved evaluation results to {pred_loader.ngram_scores_file}")


if __name__ == "__main__":
    typer.run(main)
