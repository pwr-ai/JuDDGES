import json
from functools import cached_property
from pathlib import Path
from typing import Any

from loguru import logger
from openai import OpenAI
from openai.types import Batch
from openai.types.chat import ChatCompletion

from juddges.llm_as_judge.base import (
    EvalResults,
    ItemEvalResult,
    StructuredOutputJudgeBase,
)
from juddges.llm_as_judge.data_model import ParsedPredictions, PredictionLoader
from juddges.utils.misc import load_json, load_jsonl, save_json, save_jsonl


class BatchedStructuredOutputJudge(StructuredOutputJudgeBase):
    URL = "/v1/chat/completions"
    TEMPERATURE = 0.0

    def __init__(
        self,
        client: OpenAI,
        judge_model: str,
        pred_loader: PredictionLoader,
        temperature: float = TEMPERATURE,
    ) -> None:
        super().__init__(pred_loader=pred_loader, judge_name=judge_model)
        self.client = client
        self.temperature = temperature

    @cached_property
    def structured_response_schema_from_extraction_schema(self) -> dict[str, Any]:
        """Override to return OpenAI batch API format instead of LangChain format."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "ScoreEvaluation",
                "description": "Evaluation scores for each field in the output",
                "strict": True,
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
                        for key in self.pred_loader.schema.keys()
                    },
                    "additionalProperties": False,
                    "required": list(self.pred_loader.schema.keys()),
                },
            },
        }

    @cached_property
    def batch_id(self) -> str:
        try:
            return load_json(self.pred_loader.batch_request_info_file)["id"]
        except FileNotFoundError:
            raise ValueError("Batch ID not found. Please run the submit pipeline first.")

    def run_download_and_process_results_pipeline(self) -> EvalResults | None:
        batch_id = self.batch_id
        batch = self.client.batches.retrieve(batch_id)
        if batch.status == "completed":
            save_json(batch.model_dump(), self.pred_loader.batch_request_info_file)
            results = self.download_and_parse_file_from_oai(
                file_id=batch.output_file_id,
                save_file=self.pred_loader.batch_api_results_file,
            )
            errors = self.download_and_parse_file_from_oai(
                file_id=batch.error_file_id,
                save_file=self.pred_loader.batch_api_errors_file,
            )
            return self.process_batch_api_results(
                batch=batch,
                results=results,
                errors=errors,
            )
        elif batch.status in ["validating", "in_progress", "finalizing"]:
            raise ValueError(
                f"Batch {batch_id} is still processing (status: {batch.status}). Please wait for completion."
            )
        else:
            save_json(batch.model_dump(), self.pred_loader.batch_request_info_file)
            raise ValueError(f"Batch {batch_id} has unexpected status {batch.status}.")

    def process_batch_api_results(
        self,
        batch: Batch | None,
        results: dict[int, dict[str, Any]],
        errors: dict[int, dict[str, Any]],
    ) -> EvalResults | None:
        if batch is None:
            batch = Batch(**load_json(self.pred_loader.batch_request_info_file))

        try:
            results, errors = self.load_batch_api_results(batch)
        except ValueError as err:
            logger.error(err)
            return None

        parsed_preds = self.pred_loader.load_predictions_from_file()

        results_indexed = {int(res["custom_id"]): res for res in results}

        final_results = []
        for idx in range(parsed_preds.num_items):
            try:
                batch_response = results_indexed[idx]
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
            else:
                response = batch_response["response"]
                assert response["status_code"] == 200
                completion = ChatCompletion(**response["body"])
                raw_res = json.loads(completion.choices[0].message.content)
                res = ItemEvalResult.from_success(
                    result=raw_res,
                    missing_keys=parsed_preds.missing_keys[idx],
                    extra_keys=parsed_preds.extra_keys[idx],
                )
            final_results.append(res)
        return EvalResults(results=final_results, ie_schema=self.pred_loader.schema)

    def run_submit_batch_api_pipeline(self) -> Batch:
        """Prepares and submits batch API requests to OpenAI."""
        logger.info("Loading predictions...")
        parsed_preds = self.pred_loader.load_predictions_from_file()
        logger.info("Preparing batch API requests...")
        self.prepare_and_save_batch_api_requests(parsed_preds)
        logger.info("Submitting batch API requests...")
        batch_input_file = self.client.files.create(
            file=self.pred_loader.batch_api_requests_file,
            purpose="batch",
        )
        response = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint=self.URL,
            completion_window="24h",
            metadata={"description": "Evaluation of predictions using LLM as judge"},
        )
        save_json(response.model_dump(), self.pred_loader.batch_request_info_file)
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
                    "model": self.judge_name,
                    "temperature": self.temperature,
                    "response_format": self.structured_response_schema_from_extraction_schema,
                    "messages": messages,
                },
            }
            batch_requests.append(batch_request)

        save_jsonl(batch_requests, self.pred_loader.batch_api_requests_file)
        save_json(self.get_batch_api_metadata(), self.pred_loader.batch_api_metadata_file)
        logger.info(f"Saved batch API requests to {self.pred_loader.batch_api_requests_file}")
        return self.pred_loader.batch_api_requests_file

    def get_batch_api_metadata(self) -> dict[str, Any]:
        return {
            "model": self.judge_name,
            "temperature": self.temperature,
            "predictions_dir": str(self.pred_loader.root_dir),
        }

    def download_and_parse_file_from_oai(
        self,
        file_id: str | None,
        save_file: Path,
    ) -> dict[int, dict[str, Any]]:
        if file_id is None:
            return {}
        else:
            file = self.client.files.content(file_id=file_id)
            results = [json.loads(line) for line in file.text.splitlines()]
            save_jsonl(results, save_file)
            return {int(res["custom_id"]): res for res in results}

    def load_batch_api_results(
        self,
        batch: Batch,
    ) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
        if batch.request_counts.completed > 0:
            results = load_jsonl(self.pred_loader.batch_api_results_file)
        else:
            raise ValueError(
                f"Batch {self.batch_id} has no completed requests: {batch.request_counts}"
            )

        if batch.request_counts.failed > 0:
            errors = load_jsonl(self.pred_loader.batch_api_errors_file)
        else:
            errors = {}

        return results, errors
