import json
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from openai import OpenAI


def create_response_schema(schema: dict) -> dict:
    """Create OpenAI response schema for structured output based on the schema configuration."""
    return {
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


def download_batch_results(client: OpenAI, batch_id: str) -> bytes:
    """Download raw results from a completed batch job."""
    batch_job = client.batches.retrieve(batch_id)

    if batch_job.status != "completed":
        raise ValueError(f"Batch job is not completed. Current status: {batch_job.status}")

    result_file_id = batch_job.output_file_id
    return client.files.content(result_file_id).content


def parse_batch_results(content: bytes, schema: dict) -> dict[int, dict]:
    """Parse raw batch results content into structured format."""
    batch_results = {}

    for line in content.decode("utf-8").strip().split("\n"):
        if line:
            result_obj = json.loads(line)
            custom_id = result_obj["custom_id"]
            idx = int(custom_id.split("-")[1])

            if result_obj.get("error"):
                # Handle API errors
                batch_results[idx] = {
                    key: {
                        "score": 0,
                        "missing": False,
                        "rationale": f"API error: {result_obj['error']['message']}",
                    }
                    for key in schema.keys()
                }
            else:
                # Parse successful response
                response_content = result_obj["response"]["body"]["choices"][0]["message"][
                    "content"
                ]
                try:
                    evaluation = json.loads(response_content)
                    batch_results[idx] = evaluation
                except json.JSONDecodeError:
                    batch_results[idx] = {
                        key: {
                            "score": 0,
                            "missing": False,
                            "rationale": "Failed to parse evaluation response",
                        }
                        for key in schema.keys()
                    }

    return batch_results


def calculate_aggregated_results(results: dict[int, dict]) -> dict[str, dict[str, float]]:
    """Calculate aggregated metrics from individual results."""
    score_data = {}
    missing_data = {}

    for result in results.values():
        for key, evaluation in result.items():
            if key not in score_data:
                score_data[key] = []
                missing_data[key] = []
            score_data[key].append(evaluation["score"])
            missing_data[key].append(evaluation["missing"])

    return {
        key: {
            "mean_score": sum(scores) / len(scores) if scores else 0,
            "missing_rate": sum(missing) / len(missing) if missing else 0,
        }
        for key, scores in score_data.items()
        for missing in [missing_data[key]]
    }


def save_evaluation_results(
    results: dict[int, dict],
    output_file: Path,
    total_docs: int,
    num_parsing_errors: int,
    batch_id: Optional[str] = None,
    batch_status: Optional[str] = None,
) -> None:
    """Save final evaluation results to JSON file."""
    aggregated_results = calculate_aggregated_results(results)

    final_results = {
        "stats": {
            "total_docs": total_docs,
            "num_parsing_errors": num_parsing_errors,
        },
        "aggregated_results": aggregated_results,
        "all_results": results,
    }

    # Add batch-specific stats if provided
    if batch_id:
        final_results["stats"]["batch_job_id"] = batch_id
    if batch_status:
        final_results["stats"]["batch_status"] = batch_status

    with open(output_file, "w") as f:
        json.dump(final_results, f, indent="\t", ensure_ascii=False)


def wait_for_batch_completion(
    client: OpenAI,
    batch_id: str,
    check_interval: int = 30,
    verbose: bool = True,
):
    """Wait for batch job to complete with status updates."""
    if verbose:
        logger.info("Waiting for batch completion...")

    while True:
        batch_job = client.batches.retrieve(batch_id)

        if batch_job.status not in ["validating", "in_progress", "finalizing"]:
            break

        if verbose:
            completed = batch_job.request_counts.completed if batch_job.request_counts else 0
            total = batch_job.request_counts.total if batch_job.request_counts else 0
            logger.info(f"Status: {batch_job.status} | Completed: {completed}/{total}")

        time.sleep(check_interval)

    return batch_job


def handle_parsing_errors(schema: dict, error_rationale: str) -> dict[str, dict[str, Any]]:
    """Create error results for parsing failures."""
    return {
        key: {"score": 0, "missing": False, "rationale": error_rationale} for key in schema.keys()
    }


def print_batch_status(batch_job) -> None:
    """Print formatted batch job status information."""
    logger.info(f"Batch ID: {batch_job.id}")
    logger.info(f"Status: {batch_job.status}")
    logger.info(f"Created at: {batch_job.created_at}")
    logger.info(f"Endpoint: {batch_job.endpoint}")
    logger.info(f"Completion window: {batch_job.completion_window}")

    if batch_job.request_counts:
        logger.info(f"Total requests: {batch_job.request_counts.total}")
        logger.info(f"Completed: {batch_job.request_counts.completed}")
        logger.info(f"Failed: {batch_job.request_counts.failed}")

    if batch_job.metadata:
        logger.info(f"Metadata: {batch_job.metadata}")

    if batch_job.status == "completed":
        logger.info(f"Output file ID: {batch_job.output_file_id}")

    if batch_job.status == "failed" and batch_job.errors:
        logger.error("Errors:")
        for error in batch_job.errors.data:
            logger.error(f"  - {error}")


def create_batch_file(batch_requests: list[dict], file_path: Path) -> None:
    """Create a JSONL batch file from requests."""
    with open(file_path, "w") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")


def upload_and_submit_batch(
    client: OpenAI,
    batch_file_path: Path,
    description: str | None = None,
) -> str:
    """Upload batch file and submit batch job, return batch ID."""
    batch_file = client.files.create(file=batch_file_path, purpose="batch")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description} if description else None,
    )

    return batch_job.id


def create_batch_api_requests(
    messages_batch: dict[int, list[dict]],
    model: str,
    response_format: dict,
    url: str = "/v1/chat/completions",
    temperature: float = 0.0,
    custom_id_prefix: str = "request",
) -> list[dict]:
    """
    Create batch requests for OpenAI API from a list of messages and parameters.
    Each item in messages_list should be a list of message dicts (role/content pairs).
    """
    batch_requests = []
    for idx, messages in messages_batch.items():
        batch_request = {
            "custom_id": f"{custom_id_prefix}-{idx}",
            "method": "POST",
            "url": url,
            "body": {
                "model": model,
                "temperature": temperature,
                "response_format": response_format,
                "messages": messages,
            },
        }
        batch_requests.append(batch_request)
    return batch_requests
