import asyncio
import json
import os
from pathlib import Path
from typing import Any

import typer
import yaml
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.json import parse_json_markdown
from langchain_openai import ChatOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

MAX_CONCURRENT_CALLS = 20
CACHE_DB = ".llm_as_judge_cache.db"

# Prompt based on: https://github.com/langchain-ai/openevals/blob/main/python/openevals/json/match.py
SYSTEM_PROMPT = """
You are an LLM that evaluates the accuracy of structured outputs.
* Make sure to evaluate each key the users ask you separately.
* Assign the score for each key based on its own criteria - DO NOT convolute the scores of different keys.
* Only evaluate the output vs. the reference output based on the criteria. DO NOT EVALUATE BASED ON ANYTHING ELSE.
* If the output does not match the reference output in some way that is not mentioned in the criteria that is not a problem and you should ignore those discrepancies.
* If there is a None value being compared to a non-None value, you should assign a score of 0.
* For lists provide average scores for each item, ignore the order of the items.
* You should ignore minor typos and formatting differences.
* If a key is in the reference but missing in the output, assign score 0; ignore extra keys in output.
"""

USER_PROMPT = """Please evaluate the accuracy of the following output keys according to these schema:
{schema}
<Outputs>
{outputs}
</Outputs>
<Expected Outputs>
{reference_outputs}
</Expected Outputs>
"""


def main(
    predictions_file: Path = typer.Argument(..., help="Path to JSON file with predictions"),
    schema_file: Path = typer.Argument(..., help="Path to JSON file with schema"),
    judge_model: str = typer.Option(..., help="Name of the LLM model to use"),
    max_concurrent_calls: int = typer.Option(
        MAX_CONCURRENT_CALLS, help="Maximum number of concurrent API calls"
    ),
    cache_db: Path = typer.Option(CACHE_DB, help="Path to SQLite cache database"),
) -> None:
    """Evaluate predictions using LLM as judge."""
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=judge_model,
        temperature=0.0,
    )

    if cache_db:
        set_llm_cache(SQLiteCache(str(cache_db)))

    with open(schema_file) as f:
        schema = yaml.safe_load(f)

    response_schema = {
        "name": "ScoreEvaluation",
        "description": "Evaluation scores for each field in the output",
        "parameters": {
            "type": "object",
            "properties": {
                key: {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "description": "The score, either 0, 1 for a single item, or average for list items",
                        },
                        "missing": {
                            "type": "boolean",
                            "description": "Flag determining if the key was missing in assessed outputs",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Additional explanation of decision, whenever assessment is controversial",
                        },
                    },
                    "required": ["score", "missing", "rationale"],
                    "additionalProperties": False,
                }
                for key in schema.keys()
            },
            "additionalProperties": False,
            "required": list(schema.keys()),
        },
    }
    structured_llm = llm.with_structured_output(response_schema, method="json_schema", strict=True)

    with open(predictions_file) as f:
        data = json.load(f)

    outputs = {}
    reference_outputs = {}
    results = {}
    num_parsing_errors = 0
    for idx, item in enumerate(data):
        try:
            parsed_output = parse_json_markdown(item["answer"])
        except json.JSONDecodeError:
            results[idx] = {
                key: {"score": 0, "missing": False, "rationale": "JSON parsing error"}
                for key in schema.keys()
            }
            num_parsing_errors += 1
            continue

        if isinstance(parsed_output, dict):
            outputs[idx] = parsed_output
            reference_outputs[idx] = parse_json_markdown(item["gold"])
        else:
            results[idx] = {
                key: {
                    "score": 0,
                    "missing": False,
                    "rationale": "Output is not a valid JSON object",
                }
                for key in schema.keys()
            }
            num_parsing_errors += 1

    judge = LLMJudge(structured_llm, max_concurrent_calls=max_concurrent_calls)
    results = asyncio.run(
        judge.evaluate_batch(
            schema,
            outputs,
            reference_outputs,
        )
    )

    output_file = predictions_file.with_name(f"llm_as_judge_{judge_model}.json")

    # Extract scores for aggregation
    score_data = {}
    missing_data = {}
    for result in results.values():
        for key, evaluation in result.items():
            if key not in score_data:
                score_data[key] = []
                missing_data[key] = []
            score_data[key].append(evaluation["score"])
            missing_data[key].append(evaluation["missing"])

    aggregated_results = {
        key: {
            "mean_score": sum(scores) / len(scores) if scores else 0,
            "missing_rate": sum(missing) / len(missing) if missing else 0,
        }
        for key, scores in score_data.items()
        for missing in [missing_data[key]]
    }

    final_results = {
        "stats": {
            "total_docs": len(data),
            "num_parsing_errors": num_parsing_errors,
        },
        "aggregated_results": aggregated_results,
        "all_results": results,
    }
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent="\t", ensure_ascii=False)


class LLMJudge:
    def __init__(self, client: ChatOpenAI, max_concurrent_calls: int = 5):
        """Initialize the LLM judge with a client and concurrency limit."""
        self.client = client
        self.max_concurrent_calls = max_concurrent_calls
        self.semaphore = asyncio.Semaphore(self.max_concurrent_calls)

    async def evaluate_batch(
        self,
        schema: dict[str, Any],
        outputs: dict[int, dict[str, Any]],
        reference_outputs: dict[int, dict[str, Any]],
    ) -> dict[int, dict[str, float]]:
        """Evaluate a batch of examples concurrently."""
        tasks = [
            self.evaluate_single(schema, outputs[idx], reference_outputs[idx])
            for idx in outputs.keys()
        ]
        eval_results = await tqdm_asyncio.gather(*tasks, desc="Evaluating predictions")
        return {k: v for k, v in zip(outputs.keys(), eval_results, strict=True)}

    async def evaluate_single(
        self,
        schema: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any],
    ) -> dict[str, float]:
        """Evaluate a single example using the LLM judge."""
        async with self.semaphore:
            prompt = ChatPromptTemplate.from_messages(
                [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
            )
            chain = prompt | self.client
            scores = await chain.ainvoke(
                {
                    "schema": json.dumps(
                        schema,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    "outputs": json.dumps(
                        outputs,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    "reference_outputs": json.dumps(
                        reference_outputs,
                        indent=2,
                        ensure_ascii=False,
                    ),
                }
            )
            return scores


if __name__ == "__main__":
    typer.run(main)
