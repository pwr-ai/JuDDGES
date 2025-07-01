import json
import os
from pathlib import Path
from typing import Optional

import typer
import yaml
from dotenv import load_dotenv
from openai import OpenAI

from juddges.evaluation.batch_utils import (
    download_batch_results,
    get_batch_status,
    handle_parsing_errors,
    parse_batch_results,
    print_batch_status,
    save_evaluation_results,
)

load_dotenv()

app = typer.Typer(help="Manage OpenAI Batch API jobs")


@app.command()
def status(batch_id: str = typer.Argument(..., help="Batch job ID")) -> None:
    """Check the status of a batch job."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        batch_job = client.batches.retrieve(batch_id)
        print_batch_status(batch_job)
    except Exception as e:
        print(f"Error retrieving batch job: {e}")


@app.command()
def download(
    batch_id: str = typer.Argument(..., help="Batch job ID"),
    output_file: Path = typer.Argument(..., help="Path to save results"),
    predictions_file: Optional[Path] = typer.Option(
        None, help="Original predictions file for processing"
    ),
    schema_file: Optional[Path] = typer.Option(None, help="Schema file for processing"),
) -> None:
    """Download results from a completed batch job."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        batch_job = get_batch_status(client, batch_id)

        if batch_job.status != "completed":
            print(f"Batch job is not completed. Current status: {batch_job.status}")
            return

        print("Downloading results...")
        result_content = download_batch_results(client, batch_id)

        # Save raw results
        with open(output_file, "wb") as f:
            f.write(result_content)
        print(f"Raw results saved to: {output_file}")

        # If additional files provided, process into final format
        if predictions_file and schema_file:
            print("Processing results into final format...")
            process_batch_results_helper(
                batch_results_file=output_file,
                predictions_file=predictions_file,
                schema_file=schema_file,
                batch_job_id=batch_id,
            )

    except Exception as e:
        print(f"Error downloading batch results: {e}")


@app.command()
def cancel(batch_id: str = typer.Argument(..., help="Batch job ID")) -> None:
    """Cancel a batch job."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        batch_job = client.batches.cancel(batch_id)
        print(f"Batch job {batch_id} cancelled successfully")
        print(f"New status: {batch_job.status}")

    except Exception as e:
        print(f"Error cancelling batch job: {e}")


@app.command()
def list_jobs() -> None:
    """List recent batch jobs."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        batches = client.batches.list(limit=20)

        if not batches:
            print("No batch jobs found.")
            return

        print(f"{'Batch ID':<30} {'Status':<15} {'Created':<20} {'Requests':<15}")
        print("-" * 80)

        for batch in batches:
            created_time = batch.created_at
            requests_info = (
                f"{batch.request_counts.completed}/{batch.request_counts.total}"
                if batch.request_counts
                else "N/A"
            )
            print(f"{batch.id:<30} {batch.status:<15} {created_time:<20} {requests_info:<15}")

    except Exception as e:
        print(f"Error listing batch jobs: {e}")


def process_batch_results_helper(
    batch_results_file: Path,
    predictions_file: Path,
    schema_file: Path,
    batch_job_id: str,
) -> None:
    """Process raw batch results into final evaluation format."""
    with open(schema_file) as f:
        schema = yaml.safe_load(f)

    with open(predictions_file) as f:
        data = json.load(f)

    # Parse batch results
    with open(batch_results_file, "rb") as f:
        content = f.read()

    results = parse_batch_results(content, schema)

    # Count original parsing errors (items that weren't sent to batch)
    original_parsing_errors = 0
    for idx, item in enumerate(data):
        if idx not in results:
            # These were parsing errors from the original data
            results[idx] = handle_parsing_errors(schema, "JSON parsing error in original data")
            original_parsing_errors += 1

    # Save processed results using shared utility
    output_file = predictions_file.with_name("llm_as_judge_batch_processed.json")
    save_evaluation_results(
        results,
        output_file,
        len(data),
        original_parsing_errors,
        batch_job_id,
        "completed",
    )

    print(f"Processed results saved to: {output_file}")


if __name__ == "__main__":
    app()
