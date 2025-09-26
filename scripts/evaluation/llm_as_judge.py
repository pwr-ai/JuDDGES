import asyncio
import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_openai import ChatOpenAI
from loguru import logger

from juddges.llm_as_judge.data_model import PredictionLoader
from juddges.llm_as_judge.judge import StructuredOutputJudge
from juddges.utils.config import load_and_resolve_config

load_dotenv()

API_KEY = os.environ["OPENAI_API_KEY"]
API_BASE_URL = os.getenv("OPENAI_API_BASE_URL")

MAX_CONCURRENT_CALLS = 20
CACHE_DB = ".llm_as_judge_cache.db"


def main(
    predictions_dir: Path = typer.Argument(..., help="Path to directory with predictions"),
    judge_model: str = typer.Option(..., help="Name of the LLM model to use"),
    max_concurrent_calls: int = typer.Option(
        MAX_CONCURRENT_CALLS, help="Maximum number of concurrent API calls"
    ),
    cache_db: Path = typer.Option(CACHE_DB, help="Path to SQLite cache database"),
    estimate_cost: bool = typer.Option(False, help="Estimate cost"),
    prompt: Path = typer.Option(..., help="Path to prompt config"),
) -> None:
    """Evaluate predictions using LLM as judge."""
    prompt = load_and_resolve_config(prompt)
    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
        model_name=judge_model,
        temperature=0.0,
    )

    if cache_db:
        logger.info(f"Using cache database: {cache_db}")
        set_llm_cache(SQLiteCache(str(cache_db)))

    pred_loader = PredictionLoader(root_dir=predictions_dir, judge_name=judge_model)
    pred_loader.setup_judge_dir()

    judge = StructuredOutputJudge(
        client=llm,
        pred_loader=pred_loader,
        system_prompt=prompt["system_prompt"],
        user_prompt=prompt["user_prompt"],
        max_concurrent_calls=max_concurrent_calls,
        verbose=True,
    )

    if estimate_cost:
        cost = judge.estimate_prefill_cost()
        logger.info(f"Estimated cost: ${cost:.2f}")
    else:
        results = asyncio.run(judge.evaluate())
        with pred_loader.llm_judge_scores_file.open("w") as f:
            json.dump(results.model_dump(), f, indent="\t", ensure_ascii=False)


if __name__ == "__main__":
    typer.run(main)
