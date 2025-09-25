import ast
import asyncio
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import typer
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
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

from juddges.data.database import BatchedDatabaseCursor, MongoInterface
from juddges.utils.misc import load_yaml

load_dotenv()

_GEMINI_DISABLE_THINKING_KWARGS = {"extra_body": {"thinking": {"budget_tokens": 0}}}
_GEMINI_MINIMAL_THINKING_KWARGS = {"extra_body": {"thinking": {"budget_tokens": 128}}}
MODEL_KWARGS = {
    "gemini-2.5-flash": _GEMINI_DISABLE_THINKING_KWARGS,
    "gemini-2.5-pro": _GEMINI_MINIMAL_THINKING_KWARGS,
}


MONGO_URI = os.environ["MONGO_URI"]
DEFAULT_MAX_CONCURRENT_CALLS = 10
DEFAULT_BATCH_SIZE = 50
DEFAULT_FILTER_QUERY = {
    "full_text": {"$ne": None},
    "judgment_type": {"$regex": "REASON", "$options": "i"},
}

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
LLM_BASE_URL = os.environ["LLM_BASE_URL"]


def main(
    schema_path: Path = typer.Option(..., help="Path to JSON schema file"),
    prompt_path: Path = typer.Option(..., help="Path to prompt file"),
    input_mongo_uri: str = typer.Option(MONGO_URI, help="Input MongoDB connection URI"),
    input_db_name: str = typer.Option(..., help="Input MongoDB database name"),
    input_collection_name: str = typer.Option(..., help="Input MongoDB collection name"),
    output_mongo_uri: str = typer.Option(MONGO_URI, help="Output MongoDB connection URI"),
    output_db_name: str = typer.Option(..., help="Output MongoDB database name"),
    output_collection_name: str = typer.Option(..., help="Output MongoDB collection name"),
    max_concurrent_calls: int = typer.Option(
        DEFAULT_MAX_CONCURRENT_CALLS, help="Maximum concurrent API calls"
    ),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE, help="Number of documents to process in each batch"
    ),
    model: str = typer.Option(..., help="LLM model to use"),
    document_filter_query: str = typer.Option(
        None, help='MongoDB filter query as string (e.g., \'{"field": "value"}\')'
    ),
    input_ids_file: Path = typer.Option(None, help="Path to file with input IDs"),
) -> None:
    if document_filter_query:
        filter_query = ast.literal_eval(document_filter_query)
    else:
        filter_query = DEFAULT_FILTER_QUERY

    if input_ids_file:
        with open(input_ids_file, "r") as f:
            input_ids = f.read().splitlines()
        assert (
            "_id" not in filter_query
        ), "'_id' cannot be in the filter query if input IDs are provided"
        filter_query["_id"] = {"$in": input_ids}

    llm_config = LLMConfig(llm_name=model)
    prompt_config = PromptConfig(
        prompt_id=prompt_path.stem,
        schema_id=schema_path.stem,
        prompt=load_yaml(prompt_path)["content"],
        ie_schema=load_yaml(schema_path),
    )

    extractor = InformationExtractor(
        llm_config=llm_config,
        prompt_config=prompt_config,
        max_concurrent_calls=max_concurrent_calls,
    )

    collection_processor = CollectionProcessor(
        input_db=MongoInterface(
            input_mongo_uri,
            input_db_name,
            input_collection_name,
            batch_size,
        ),
        output_db=MongoInterface(
            output_mongo_uri,
            output_db_name,
            output_collection_name,
            batch_size,
        ),
    )

    with collection_processor:
        total_docs_to_process = collection_processor.get_total_docs_to_process(filter_query)
        if typer.confirm(f"Total documents to process: {total_docs_to_process}. Continue?"):
            collection_processor.process_collection(
                info_extractor=extractor,
                batch_size=batch_size,
                filter_query=filter_query,
            )
        else:
            logger.info("Exiting...")


class LLMConfig(BaseModel):
    id: str | None = None
    llm_name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_id(self) -> "LLMConfig":
        id_ = _compute_md5_hash(
            self.llm_name,
            self.kwargs,
        )

        if self.id is not None:
            assert self.id == id_

        self.id = id_

        if self.llm_name not in MODEL_KWARGS:
            kwargs = MODEL_KWARGS[self.llm_name]
            self.kwargs |= kwargs
            logger.info(f"Adding default kwargs for {self.llm_name}: {kwargs}")

        return self


class PromptConfig(BaseModel):
    id: str | None = None
    prompt_id: str
    schema_id: str
    prompt: str | None = Field(None, exclude=True)
    ie_schema: dict[str, Any] | None = Field(None, exclude=True)

    @model_validator(mode="after")
    def set_id(self) -> "PromptConfig":
        id_ = _compute_md5_hash(
            self.prompt_id,
            self.schema_id,
        )

        if self.id is not None:
            assert self.prompt is not None
            assert self.ie_schema is not None
            assert self.id == id_

        self.id = id_
        return self


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
    id: str | None = None
    document_id: str
    llm_config: LLMConfig
    prompt_config: PromptConfig
    completion_metadata: CompletionMetadata | None
    created_at: datetime
    status: Literal["success", "error"]
    error: str | None = None
    extracted: dict[str, Any] | None = None

    @model_validator(mode="after")
    def set_id(self) -> "ExtractionResults":
        id_ = self.compute_id(
            self.llm_config,
            self.prompt_config,
            self.document_id,
        )

        if self.id is not None:
            assert self.id == id_

        self.id = id_
        return self

    @staticmethod
    def compute_id(
        model_config: LLMConfig,
        prompt_config: PromptConfig,
        document_id: str,
    ) -> str:
        # defined for consistency
        return _compute_md5_hash(
            model_config.id,
            prompt_config.id,
            document_id,
        )


class InformationExtractor:
    def __init__(
        self,
        llm_config: LLMConfig,
        prompt_config: PromptConfig,
        max_concurrent_calls: int,
    ):
        self.llm_config = llm_config
        self.prompt_config = prompt_config

        client = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=LLM_BASE_URL,
            model=self.llm_config.llm_name,
            **self.llm_config.kwargs,
        )
        client = client.with_structured_output(
            self.raw_schema_to_structured_output(self.prompt_config.ie_schema),
            method="json_schema",
            strict=True,
            include_raw=True,
        )
        prompt_template = PromptTemplate.from_template(
            self.prompt_config.prompt,
            template_format="f-string",
        )
        self.chain = prompt_template | client
        self.semaphore = asyncio.Semaphore(max_concurrent_calls)

    async def __call__(
        self,
        documents: list[dict[str, Any]],
    ) -> list[ExtractionResults]:
        return await asyncio.gather(
            *[self.extract_information(doc) for doc in documents],
        )

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, httpx.ConnectError, httpx.TimeoutException, ConnectionError)
        ),
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, min=1, max=60),
    )
    async def extract_information(self, document: dict[str, Any]) -> ExtractionResults:
        assert (field in document for field in ["_id", "full_text"])

        async with self.semaphore:
            try:
                result = await self.chain.ainvoke(
                    {
                        "context": document["full_text"],
                    }
                )

                return ExtractionResults(
                    document_id=document["_id"],
                    llm_config=self.llm_config,
                    prompt_config=self.prompt_config,
                    completion_metadata=CompletionMetadata.from_oai_response_metadata(
                        result["raw"].response_metadata
                    ),
                    created_at=datetime.now(),
                    status="success",
                    extracted=result["parsed"],
                )

            except (BadRequestError, ContentFilterFinishReasonError) as err:
                logger.error(f"Error processing document {document['_id']}: {err}")
                return ExtractionResults(
                    document_id=document["_id"],
                    llm_config=self.llm_config,
                    prompt_config=self.prompt_config,
                    completion_metadata=None,
                    created_at=datetime.now(),
                    status="error",
                    error=str(err),
                    extracted=None,
                )

            except LengthFinishReasonError as err:
                logger.error(f"Error processing document {document['_id']}: {err}")
                return ExtractionResults(
                    document_id=document["_id"],
                    llm_config=self.llm_config,
                    prompt_config=self.prompt_config,
                    completion_metadata=CompletionMetadata.from_chat_completion(
                        err.completion,
                        "length_finish_reason_error",
                    ),
                    created_at=datetime.now(),
                    status="error",
                    error=str(err),
                    extracted=None,
                )

    @staticmethod
    def raw_schema_to_structured_output(raw_schema: dict[str, Any]) -> dict[str, Any]:
        """Schema processed to be compatible with OpenAI JSON Schema output."""
        schema = deepcopy(raw_schema)

        # according to the OpenAI JSON Schema format, all fields must be required
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


class CollectionProcessor:
    """Processes a collection of documents from one database and stores results in another."""

    def __init__(self, input_db: MongoInterface, output_db: MongoInterface):
        self.input_db = input_db
        self.output_db = output_db

    def __enter__(self):
        self.input_db.__enter__()
        self.output_db.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.input_db.__exit__(exc_type, exc_value, traceback)
        self.output_db.__exit__(exc_type, exc_value, traceback)

    def process_collection(
        self,
        info_extractor: InformationExtractor,
        batch_size: int,
        filter_query: dict[str, Any],
    ) -> None:
        asyncio.run(self.process_collection_async(info_extractor, batch_size, filter_query))

    def get_total_docs_to_process(self, filter_query: dict[str, Any]) -> int:
        return self.input_db.collection.count_documents(filter_query)

    async def process_collection_async(
        self,
        info_extractor: InformationExtractor,
        batch_size: int,
        filter_query: dict[str, Any],
    ) -> None:
        with self.input_db, self.output_db:
            total_docs_to_process = self.get_total_docs_to_process(filter_query)
            cursor = self.input_db.collection.find(
                filter_query,
                {"_id": 1, "full_text": 1},
                batch_size=batch_size,
            )

            batched_cursor = BatchedDatabaseCursor(cursor, batch_size, prefetch=False)

            processed_count = 0
            skipped_and_processed_count = 0
            with tqdm(total=total_docs_to_process, desc="Total progress") as pbar:
                for batch in batched_cursor:
                    num_docs = len(batch)
                    batch = await self.filter_processed_docs_from_batch(info_extractor, batch)
                    results = await info_extractor(batch)
                    await self.store_results(results)

                    processed_count += len(batch)
                    skipped_and_processed_count += num_docs
                    pbar.update(num_docs)

            logger.info(
                f"Completed processing {processed_count} documents ({skipped_and_processed_count} skipped or processed)"
            )

    async def filter_processed_docs_from_batch(
        self,
        info_extractor: InformationExtractor,
        batch: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ids = [
            ExtractionResults.compute_id(
                info_extractor.llm_config,
                info_extractor.prompt_config,
                doc["_id"],
            )
            for doc in batch
        ]
        return [
            doc
            for idx, doc in enumerate(batch)
            if ids[idx] not in await self.check_if_processed(ids)
        ]

    async def check_if_processed(self, document_ids: list[str]) -> list[bool]:
        found_ids = self.output_db.collection.find({"_id": {"$in": document_ids}})
        return {doc["_id"] for doc in found_ids}

    async def store_results(
        self,
        results: list[ExtractionResults],
    ) -> None:
        docs_to_save = []
        for res in results:
            doc = res.model_dump()
            doc["_id"] = doc.pop("id")
            docs_to_save.append(doc)

        if docs_to_save:
            self.output_db.update_or_insert_documents(docs_to_save)


def _compute_md5_hash(*values) -> str:
    processed_values = []
    for value in values:
        if isinstance(value, dict | list):
            processed_values.append(json.dumps(value, sort_keys=True))
        elif isinstance(value, int | float | str):
            processed_values.append(str(value))
        else:
            raise ValueError(f"Unsupported type for md5 hash: {type(value)}")

    return hashlib.md5(f"{'-'.join(processed_values)}".encode("utf-8")).hexdigest()


if __name__ == "__main__":
    typer.run(main)
