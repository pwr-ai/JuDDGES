import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from langchain_core.utils.json import parse_json_markdown
from loguru import logger
from openai import OpenAI

from juddges.evaluation.batch_utils import (
    create_batch_api_requests,
    create_batch_file,
    create_response_schema,
    download_batch_results,
    handle_parsing_errors,
    parse_batch_results,
    save_evaluation_results,
    upload_and_submit_batch,
    wait_for_batch_completion,
)
from juddges.utils.misc import load_yaml

load_dotenv()

# Prompt based on: https://github.com/langchain-ai/openevals/blob/main/python/openevals/json/match.py
SYSTEM_PROMPT = """
You are an LLM that evaluates the accuracy of structured outputs.
* Make sure to evaluate each key the users ask you separately.
* Assign the score for each key based on its own criteria - DO NOT convolute the scores of different keys.
* Only evaluate the output vs. the reference output based on the criteria. DO NOT EVALUATE BASED ON ANYTHING ELSE.
* If the output does not match the reference output in some way that is not mentioned in the criteria that is not a problem and you should ignore those discrepancies.
* Only focus on finding discrepancies based on the criteria.
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
    predictions_file: Path = typer.Option(..., help="Path to JSON file with predictions"),
    schema_file: Path = typer.Option(..., help="Path to JSON file with schema"),
    judge_model: str = typer.Option(..., help="Name of the LLM model to use"),
    wait_for_completion: bool = typer.Option(True, help="Wait for batch completion before exiting"),
    check_interval: int = typer.Option(30, help="Seconds between status checks"),
) -> None:
    """Evaluate predictions using LLM as judge with OpenAI Batch API."""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    schema = load_yaml(schema_file)
    response_schema = create_response_schema(schema)

    batch_requests, results, num_parsing_errors = prepare_batch_requests(
        predictions_file=predictions_file,
        schema=schema,
        response_schema=response_schema,
        judge_model=judge_model,
    )

    if not batch_requests:
        logger.warning("No valid requests to process. All inputs had parsing errors.")
        return

    batch_file_path = predictions_file.with_name(f"batch_requests_{judge_model}.jsonl")
    create_batch_file(batch_requests, batch_file_path)
    logger.info(f"Created batch file with {len(batch_requests)} requests: {batch_file_path}")

    batch_id = upload_and_submit_batch(
        client,
        batch_file_path,
        f"LLM as judge evaluation - {predictions_file.name}",
    )
    logger.info(f"Batch job {batch_id} submitted.")

    if not wait_for_completion:
        logger.info(f"Batch job {batch_id} submitted. Use the following command to check status:")
        logger.info(
            f"python -c \"from openai import OpenAI; client = OpenAI(); print(client.batches.retrieve('{batch_id}'))\""
        )
        return

    try:
        batch_job = wait_for_batch_completion(client, batch_id, check_interval)
    except KeyboardInterrupt:
        logger.info(
            "\nBatch job monitoring interrupted. Use the following command to check status:"
        )
        logger.info(
            f"python -c \"from openai import OpenAI; client = OpenAI(); print(client.batches.retrieve('{batch_id}'))\""
        )
        return
    else:
        process_results(
            client=client,
            batch_job=batch_job,
            schema=schema,
            predictions_file=predictions_file,
            num_parsing_errors=num_parsing_errors,
            batch_file_path=batch_file_path,
            results=results,
            judge_model=judge_model,
        )


def prepare_batch_requests(
    predictions_file: Path, schema: dict, response_schema: dict, judge_model: str
) -> tuple[list[dict], dict[int, dict], int]:
    """Parse predictions data and create batch requests."""
    with open(predictions_file) as f:
        data = json.load(f)

    messages_batch = {}
    results = {}
    num_parsing_errors = 0

    for idx, item in enumerate(data):
        try:
            parsed_output = parse_json_markdown(item["answer"])
        except json.JSONDecodeError:
            results[idx] = handle_parsing_errors(schema, "JSON parsing error")
            num_parsing_errors += 1
            continue

        if isinstance(parsed_output, dict):
            reference_output = parse_json_markdown(item["gold"])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        schema=json.dumps(schema, indent=2, ensure_ascii=False),
                        outputs=json.dumps(parsed_output, indent=2, ensure_ascii=False),
                        reference_outputs=json.dumps(
                            reference_output, indent=2, ensure_ascii=False
                        ),
                    ),
                },
            ]
            messages_batch[idx] = messages
        else:
            results[idx] = handle_parsing_errors(schema, "Output is not a valid JSON object")
            num_parsing_errors += 1

    batch_requests = create_batch_api_requests(
        messages_batch=messages_batch,
        model=judge_model,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": response_schema["name"],
                "description": response_schema["description"],
                "schema": response_schema["parameters"],
                "strict": True,
            },
        },
    )

    return batch_requests, results, num_parsing_errors


def process_results(
    client: OpenAI,
    batch_job: object,
    schema: dict,
    predictions_file: Path,
    num_parsing_errors: int,
    batch_file_path: Path,
    results: dict[int, dict],
    judge_model: str,
):
    if batch_job.status == "completed":
        logger.success("Batch completed successfully!")

        result_content = download_batch_results(client, batch_job.id)

        batch_results = parse_batch_results(result_content, schema)
        results.update(batch_results)

        with open(predictions_file) as f:
            data = json.load(f)

        output_file = predictions_file.with_name(f"llm_as_judge_batch_{judge_model}.json")
        save_evaluation_results(
            results,
            output_file,
            len(data),
            num_parsing_errors,
            batch_job.id,
            batch_job.status,
        )
        logger.info(f"Results saved to: {output_file}")

        batch_file_path.unlink()
        logger.info("Cleaned up temporary batch file")

    elif batch_job.status == "failed":
        logger.error(f"Batch job failed: {batch_job}")
        if batch_job.errors:
            logger.error("Errors:")
            for error in batch_job.errors.data:
                logger.error(f"  - {error}")

    elif batch_job.status == "cancelled":
        logger.warning("Batch job was cancelled")

    else:
        logger.warning(f"Unexpected batch status: {batch_job.status}")


if __name__ == "__main__":
    typer.run(main)
