import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from juddges.llm_as_judge.batched_judge import BatchedStructuredOutputJudge
from juddges.llm_as_judge.data_model import PredictionLoader
from juddges.utils.misc import save_json

load_dotenv()

API_KEY = os.environ["OPENAI_API_KEY"]


def main(
    action: str = typer.Argument(..., help="Action to perform"),
    predictions_dir: Path = typer.Argument(..., help="Path to directory with predictions"),
    judge_model: str = typer.Option(..., help="Name of the LLM model to use"),
) -> None:
    """Evaluate predictions using LLM as judge."""
    client = OpenAI(api_key=API_KEY)
    pred_loader = PredictionLoader(root_dir=predictions_dir, judge_name=judge_model)

    if action == "submit":
        pred_loader.setup_judge_dir()

    judge = BatchedStructuredOutputJudge(
        client=client,
        pred_loader=pred_loader,
        judge_model=judge_model,
    )

    results = None
    if action == "submit":
        judge.run_submit_batch_api_pipeline()
    elif action == "download_and_process_results":
        results = judge.run_download_and_process_results_pipeline()
    elif action == "process_results":
        results = judge.process_batch_api_results(None, None, None)
    else:
        raise ValueError(
            f"Invalid action: {action}; expected one of: submit, download_and_process_results, process_results"
        )

    if results is not None:
        save_json(results.model_dump(), pred_loader.llm_judge_scores_file, ensure_ascii=False)
        logger.info(f"Saved results to {pred_loader.llm_judge_scores_file}")


if __name__ == "__main__":
    typer.run(main)
