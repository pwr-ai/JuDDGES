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

from juddges.llm_as_judge.judge import StructuredOutputJudge

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
) -> None:
    """Evaluate predictions using LLM as judge."""
    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
        model_name=judge_model,
        temperature=0.0,
    )

    if cache_db:
        logger.info(f"Using cache database: {cache_db}")
        set_llm_cache(SQLiteCache(str(cache_db)))

    judge = StructuredOutputJudge(
        client=llm,
        predictions_dir=predictions_dir,
        max_concurrent_calls=max_concurrent_calls,
        verbose=True,
    )
    results = asyncio.run(judge.evaluate())

    with judge.output_file.open("w") as f:
        json.dump(results.model_dump(), f, indent="\t", ensure_ascii=False)


if __name__ == "__main__":
    typer.run(main)
