import asyncio
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import litellm
import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain_community.cache import SQLiteCache
from langchain_openai import ChatOpenAI
from loguru import logger
from openai import (
    BadRequestError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    RateLimitError,
)
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel, Field, model_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async

from juddges.utils.misc import load_yaml, save_yaml

load_dotenv()

_GEMINI_DISABLE_THINKING_KWARGS = {"extra_body": {"thinking": {"budget_tokens": 0}}}
_GEMINI_MINIMAL_THINKING_KWARGS = {"extra_body": {"thinking": {"budget_tokens": 128}}}
MODEL_KWARGS = {
    "gemini-2.5-flash": _GEMINI_DISABLE_THINKING_KWARGS,
    "gemini-2.5-pro": _GEMINI_MINIMAL_THINKING_KWARGS,
}

LANGCHAIN_CACHE = ".langchain_cache.db"
DEFAULT_MAX_CONCURRENT_CALLS = 5

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_BASE_URL = os.environ["OPENAI_API_BASE_URL"]


def main(
    dataset_name: str = typer.Option(..., help="HuggingFace dataset name"),
    schema_path: Path = typer.Option(..., help="Path to JSON schema file"),
    prompt_path: Path = typer.Option(..., help="Path to prompt file"),
    output_dir: Path = typer.Option(..., help="Path to output JSON file"),
    model: str = typer.Option(..., help="LLM model to use"),
    text_column: str = typer.Option("text", help="Column name containing text data"),
    id_column: str | None = typer.Option(None, help="Column name containing document IDs"),
    dataset_split: str = typer.Option("train", help="Dataset split to process"),
    max_concurrent_calls: int = typer.Option(
        DEFAULT_MAX_CONCURRENT_CALLS, help="Maximum concurrent API calls"
    ),
    skip_cost_estimation: bool = typer.Option(False, help="Skip cost estimation"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_with_metadata_output_file = output_dir / "predictions_with_metadata.json"
    preds_output_file = output_dir / "predictions.json"

    assert (
        not preds_with_metadata_output_file.exists()
    ), f"Output fil {preds_with_metadata_output_file} already exists, remove stale outputs first"
    assert (
        not preds_output_file.exists()
    ), f"Output fil {preds_output_file} already exists, remove stale outputs first"

    logger.info(f"Preparing dataset {dataset_name} (split: {dataset_split})...")
    dataset = prepare_dataset(
        dataset_name,
        dataset_split,
        text_column,
        id_column,
    )

    schema = load_yaml(schema_path)
    prompt = load_yaml(prompt_path)
    config = Config(
        llm_name=model,
        prompt_file=prompt_path,
        schema_file=schema_path,
        prompt=prompt["content"],
        ie_schema=schema,
    )

    extractor = InformationExtractor(
        config=config,
        max_concurrent_calls=max_concurrent_calls,
    )

    if not skip_cost_estimation:
        prefill_cost = extractor.estimate_prefill_cost(dataset)
        cost = f"${prefill_cost:.2f}"
    else:
        logger.info("Skipping cost estimation")
        cost = "<skipped>"

    if typer.confirm(f"Estimated prefill cost: {cost}. Continue?", default=True):
        logger.info(f"Starting processing {len(dataset)} documents...")
        all_results = asyncio.run(
            extractor.extract_from_dataset(
                dataset=dataset,
            )
        )

        logger.info(f"Saving results to {preds_with_metadata_output_file}...")
        save_results(all_results, preds_with_metadata_output_file)

        logger.info(f"Saving formatted predictions to {preds_output_file}...")
        all_results_indexed = {res.document_id: res for res in all_results}
        eval_compatible_preds = []
        for doc in dataset:
            eval_compatible_preds.append(
                {
                    "answer": json.dumps(all_results_indexed[doc["id"]].extracted),
                    "gold": doc["output"],
                }
            )
        with open(preds_output_file, "w") as f:
            json.dump(eval_compatible_preds, f, indent=2, ensure_ascii=False)
        save_yaml(config.model_dump(), output_dir / "config.yaml")


def prepare_dataset(
    dataset_name: str,
    dataset_split: str,
    text_column: str,
    id_column: str | None,
) -> Dataset:
    dataset = load_dataset(dataset_name, split=dataset_split)
    if text_column != "context":
        dataset = dataset.rename_column(text_column, "context")
    else:
        assert "context" in dataset.column_names

    if id_column is None:
        dataset = dataset.add_column(
            "id",
            list(range(len(dataset))),
        )
    elif id_column != "id":
        dataset = dataset.rename_column(id_column, "id")
    else:
        assert "id" in dataset.column_names

    return dataset


class Config(BaseModel):
    hash: str | None = Field(None)
    prompt_file: Path
    schema_file: Path
    prompt: str
    ie_schema: dict[str, Any]
    llm_name: str
    llm_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_llm_kwargs(self) -> "Config":
        if self.llm_name in MODEL_KWARGS:
            kwargs = MODEL_KWARGS[self.llm_name]
            self.llm_kwargs |= kwargs

        self.hash = hashlib.sha256(self.model_dump_json(exclude={"hash"}).encode()).hexdigest()

        return self

    @property
    def template_format(self) -> Literal["f-string", "jinja2"]:
        if self.prompt_file.suffix == ".jinja2":
            return "jinja2"
        elif self.prompt_file.suffix in {".txt", ".yaml"}:
            return "f-string"
        else:
            raise ValueError(f"Unsupported prompt file format: {self.prompt_file.suffix}")


class CompletionMetadata(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None
    finish_reason: str | None

    @classmethod
    def from_oai_response_metadata(cls, response_metadata: dict[str, Any]) -> "CompletionMetadata":
        token_usage = response_metadata["token_usage"]
        completion_tokens_details = token_usage.get("completion_tokens_details") or {}
        return cls(
            prompt_tokens=token_usage["prompt_tokens"],
            completion_tokens=token_usage["completion_tokens"],
            reasoning_tokens=completion_tokens_details.get("reasoning_tokens"),
            finish_reason=response_metadata["finish_reason"],
        )

    @classmethod
    def from_chat_completion(
        cls,
        chat_completion: ChatCompletion,
        finish_reason: str,
    ) -> "CompletionMetadata":
        completion_tokens_details = chat_completion.usage.completion_tokens_details
        if completion_tokens_details is None:
            reasoning_tokens = None
        else:
            reasoning_tokens = completion_tokens_details.reasoning_tokens
        return cls(
            prompt_tokens=chat_completion.usage.prompt_tokens,
            completion_tokens=chat_completion.usage.completion_tokens,
            reasoning_tokens=reasoning_tokens,
            finish_reason=finish_reason,
        )


class ExtractionResults(BaseModel):
    document_id: int | str
    config_hash: str
    completion_metadata: CompletionMetadata | None
    created_at: datetime
    status: Literal["success", "error"]
    error: str | None = None
    extracted: dict[str, Any] | None = None


class InformationExtractor:
    def __init__(
        self,
        config: Config,
        max_concurrent_calls: int,
    ):
        self.config = config

        client = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_BASE_URL,
            model_name=self.config.llm_name,
            **self.config.llm_kwargs,
        )
        set_llm_cache(SQLiteCache(str(LANGCHAIN_CACHE)))
        client = client.with_structured_output(
            self.raw_schema_to_structured_output(self.config.ie_schema),
            method="json_schema",
            strict=True,
            include_raw=True,
        )
        self.prompt_template = PromptTemplate.from_template(
            self.config.prompt,
            template_format=self.config.template_format,
        )
        assert set(self.prompt_template.input_variables) == {"context"}
        self.chain = self.prompt_template | client
        self.semaphore = asyncio.Semaphore(max_concurrent_calls)

    async def extract_from_dataset(
        self,
        dataset: Dataset | list[dict[str, Any]],
    ) -> list[ExtractionResults]:
        return await tqdm_async.gather(
            *[self.extract_from_single_document(doc) for doc in dataset],
        )

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, httpx.ConnectError, httpx.TimeoutException, ConnectionError)
        ),
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, min=1, max=60),
    )
    async def extract_from_single_document(self, document: dict[str, Any]) -> ExtractionResults:
        assert "id" in document and "context" in document

        async with self.semaphore:
            try:
                result = await self.chain.ainvoke(
                    {
                        "context": document["context"],
                    }
                )

                return ExtractionResults(
                    document_id=document["id"],
                    config_hash=self.config.hash,
                    completion_metadata=CompletionMetadata.from_oai_response_metadata(
                        result["raw"].response_metadata
                    ),
                    created_at=datetime.now(),
                    status="success",
                    extracted=result["parsed"],
                )

            except (BadRequestError, ContentFilterFinishReasonError) as err:
                logger.error(f"Error processing document {document['id']}: {err}")
                return ExtractionResults(
                    document_id=document["id"],
                    config_hash=self.config.hash,
                    completion_metadata=None,
                    created_at=datetime.now(),
                    status="error",
                    error=str(err),
                    extracted=None,
                )

            except LengthFinishReasonError as err:
                logger.error(f"Error processing document {document['id']}: {err}")
                return ExtractionResults(
                    document_id=document["id"],
                    config_hash=self.config.hash,
                    completion_metadata=CompletionMetadata.from_chat_completion(
                        err.completion,
                        "length_finish_reason_error",
                    ),
                    created_at=datetime.now(),
                    status="error",
                    error=str(err),
                    extracted=None,
                )

    def estimate_prefill_cost(self, dataset: Dataset | list[dict[str, Any]]) -> float:
        total_cost = sum(
            litellm.completion_cost(
                model=self.config.llm_name,
                prompt=self.prompt_template.invoke({"context": doc["context"]}).text,
            )
            for doc in tqdm(dataset, desc="Estimating prefill cost")
        )
        return total_cost

    @staticmethod
    def raw_schema_to_structured_output(raw_schema: dict[str, Any]) -> dict[str, Any]:
        schema = deepcopy(raw_schema)

        required = []
        for field_name, field_schema in schema.items():
            del field_schema["required"]
            required.append(field_name)

        return {
            "name": "InformationExtraction",
            "strict": True,
            "description": "Extracted information from the text",
            "parameters": {
                "type": "object",
                "properties": schema,
                "additionalProperties": False,
                "required": required,
            },
        }


def save_results(results: list[ExtractionResults], output_file: Path) -> None:
    output_data = [res_item.model_dump(mode="json") for res_item in results]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    typer.run(main)
