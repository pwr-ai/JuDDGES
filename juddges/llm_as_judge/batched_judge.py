import json
from functools import cached_property
from pathlib import Path
from typing import Any

from loguru import logger
from openai import OpenAI
from openai.types import Batch

from juddges.llm_as_judge.base import (
    EvalResults,
    ItemEvalResult,
    ParsedPredictions,
    StructuredOutputJudgeBase,
)
from juddges.utils.misc import load_json, load_jsonl, save_json, save_jsonl


class BatchedStructuredOutputJudge(StructuredOutputJudgeBase):
    URL = "/v1/chat/completions"
    TEMPERATURE = 0.0

    def __init__(
        self,
        client: OpenAI,
        judge_model: str,
        predictions_dir: Path,
        temperature: float = TEMPERATURE,
    ) -> None:
        super().__init__(predictions_dir=predictions_dir, judge_name=judge_model)
        self.client = client
        self.judge_model = judge_model
        self.temperature = temperature

    @property
    def batch_api_errors_file(self) -> Path:
        return self.judge_dir / "batch_api_errors.jsonl"

    @cached_property
    def structured_response_schema_from_extraction_schema(self) -> dict[str, Any]:
        """Override to return OpenAI batch API format instead of LangChain format."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "ScoreEvaluation",
                "description": "Evaluation scores for each field in the output",
                "schema": {
                    "type": "object",
                    "properties": {
                        key: {
                            "type": "object",
                            "properties": {
                                "score": {
                                    "type": "number",
                                    "description": "The score, either 0, 1 for a single item, or average for list items",
                                },
                            },
                            "required": ["score"],
                            "additionalProperties": False,
                        }
                        for key in self.schema.keys()
                    },
                    "additionalProperties": False,
                    "required": list(self.schema.keys()),
                },
            },
        }

    @cached_property
    def batch_id(self) -> str:
        return load_json(self.batch_request_info_file)["id"]

    def run_download_and_process_results_pipeline(self) -> EvalResults | None:
        try:
            batch_id = self.batch_id
        except FileNotFoundError:
            raise ValueError("Batch ID not found. Please run the submit pipeline first.")

        batch = self.client.batches.retrieve(batch_id)
        if batch.status == "completed":
            save_json(batch.model_dump(), self.batch_request_info_file)
            if batch.request_counts.completed == 0:
                logger.error(f"Batch {batch_id} has no completed requests: {batch.request_counts}")
                self.download_batch_api_errors()
                return None
            else:
                if batch.request_counts.failed > 0:
                    errors = self.download_batch_api_errors()
                else:
                    errors = {}

                results = self.download_batch_api_results()

                return self.process_batch_api_results(
                    batch=batch,
                    errors=errors,
                    results=results,
                )

        elif batch.status in ["validating", "in_progress", "finalizing"]:
            raise ValueError(f"Batch {batch_id} is still processing. Please wait for completion.")
        else:
            save_json(batch.model_dump(), self.batch_request_info_file)
            raise ValueError(f"Batch {batch_id} has status {batch.status}.")

    def process_batch_api_results(
        self,
        batch: Batch | None,
        errors: dict[int, dict[str, Any]] | None,
        results: dict[int, dict[str, Any]] | None,
    ) -> EvalResults | None:
        # todo: make it more efficient and consistent
        if batch is None:
            batch = Batch(**load_json(self.batch_request_info_file))

        if batch.request_counts.completed > 0:
            results = load_jsonl(self.batch_api_results_file)
        else:
            logger.error(f"Batch {self.batch_id} has no completed requests: {batch.request_counts}")
            return None

        if batch.request_counts.failed > 0:
            errors = load_jsonl(self.batch_api_errors_file)
        else:
            errors = {}

        parsed_preds = self.load_predictions()

        final_results = []
        for idx in range(parsed_preds.num_items):
            try:
                # todo: make add validation for the response
                raw_res = json.loads(
                    results[idx]["response"]["body"]["choices"][0]["message"]["content"]
                )
                res = ItemEvalResult.from_success(
                    result=raw_res,
                    missing_keys=parsed_preds.missing_keys[idx],
                    extra_keys=parsed_preds.extra_keys[idx],
                )
            except KeyError:
                try:
                    res = ItemEvalResult(
                        status="judge_error",
                        error=errors[idx],
                        result=self.get_zero_scores(),
                    )
                except KeyError:
                    res = ItemEvalResult(
                        status="parsing_error",
                        error=parsed_preds.errors[idx],
                        result=self.get_zero_scores(),
                    )
            final_results.append(res)
        return EvalResults(results=final_results, ie_schema=self.schema)

    def run_submit_batch_api_pipeline(self) -> Batch:
        """Prepares and submits batch API requests to OpenAI."""
        parsed_preds = self.load_predictions()
        self.prepare_and_save_batch_api_requests(parsed_preds)
        batch_input_file = self.client.files.create(
            file=self.batch_api_requests_file,
            purpose="batch",
        )
        response = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint=self.URL,
            completion_window="24h",
            metadata={"description": "Evaluation of predictions using LLM as judge"},
        )
        save_json(response.model_dump(), self.batch_request_info_file)
        logger.info(f"Submitted batch API requests to OpenAI. Batch ID: {response.id}")
        return response

    def prepare_and_save_batch_api_requests(
        self,
        parsed_preds: ParsedPredictions,
    ) -> Path:
        """Prepares batch API requests and stores the file with requests for further submission."""
        dataset_messages = self.prepare_eval_messages(parsed_preds)
        batch_requests = []
        for idx, messages in dataset_messages.items():
            batch_request = {
                "custom_id": str(idx),
                "method": "POST",
                "url": self.URL,
                "body": {
                    "model": self.judge_model,
                    "temperature": self.temperature,
                    "response_format": self.structured_response_schema_from_extraction_schema,
                    "messages": messages,
                },
            }
            batch_requests.append(batch_request)

        save_jsonl(batch_requests, self.batch_api_requests_file)
        save_json(self.get_batch_api_metadata(), self.batch_api_metadata_file)
        logger.info(f"Saved batch API requests to {self.batch_api_requests_file}")
        return self.batch_api_requests_file

    def get_batch_api_metadata(self) -> dict[str, Any]:
        return {
            "model": self.judge_model,
            "temperature": self.temperature,
            "predictions_dir": str(self.predictions_dir),
        }

    def download_batch_api_errors(self) -> dict[int, dict[str, Any]]:
        batch = self.client.batches.retrieve(batch_id=self.batch_id)
        error_file_id = batch.error_file_id
        error_file = self.client.files.content(file_id=error_file_id)
        errors = [json.loads(line) for line in error_file.text.splitlines()]
        save_jsonl(errors, self.batch_api_errors_file)
        return {int(err["custom_id"]): err for err in errors}

    def download_batch_api_results(self) -> dict[int, dict[str, Any]]:
        batch = self.client.batches.retrieve(batch_id=self.batch_id)
        result_file_id = batch.output_file_id
        result_file = self.client.files.content(file_id=result_file_id)
        results = [json.loads(line) for line in result_file.text.splitlines()]
        save_jsonl(results, self.batch_api_results_file)
        return {int(res["custom_id"]): res for res in results}
