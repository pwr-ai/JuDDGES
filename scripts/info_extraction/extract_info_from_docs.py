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
from pydantic import BaseModel, model_validator
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

app = typer.Typer()

MONGO_URI = os.environ["MONGO_URI"]

DEFAULT_MAX_CONCURRENT_CALLS = 25
DEFAULT_BATCH_SIZE = 50
DEFAULT_FILTER_QUERY = {
    "full_text": {"$ne": None},
    "last_update": {"$gte": datetime(2025, 8, 1)},
    "judgment_type": {"$regex": "REASON", "$options": "i"},
}

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
LLM_BASE_URL = os.environ["LLM_BASE_URL"]


@app.command()
def main(
    schema_path: Path = typer.Option(..., help="Path to JSON schema file"),
    prompt_path: Path = typer.Option(..., help="Path to prompt JSON file"),
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
    filter_query: str = typer.Option(
        None, help='MongoDB filter query as string (e.g., \'{"field": "value"}\')'
    ),
) -> None:
    if filter_query:
        parsed_filter_query = ast.literal_eval(filter_query)
    else:
        parsed_filter_query = DEFAULT_FILTER_QUERY

    schema = load_yaml(schema_path)
    prompt = load_yaml(prompt_path)["content"]

    client = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=LLM_BASE_URL,
        model=model,
    )

    metadata = {
        "schema_id": schema_path.stem,
        "schema_hash": InfoExtractionResults.get_schema_hash(schema),
        "prompt_id": prompt_path.stem,
        "prompt_hash": InfoExtractionResults.get_prompt_hash(prompt),
    }

    extractor = InformationExtractor(
        client=client,
        schema=schema,
        prompt_template=prompt,
        metadata=metadata,
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
        total_docs_to_process = collection_processor.get_total_docs_to_process(parsed_filter_query)
        if typer.confirm(f"Total documents to process: {total_docs_to_process}. Continue?"):
            collection_processor.process_collection(
                info_extractor=extractor,
                batch_size=batch_size,
                filter_query=parsed_filter_query,
            )
        else:
            logger.info("Exiting...")


class ApiRequestInfo(BaseModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    finish_reason: str | None


class InfoExtractionResults(BaseModel):
    id: str | None = None
    document_id: str
    prompt_id: str
    prompt_hash: str
    schema_id: str
    schema_hash: str
    api_request_info: ApiRequestInfo
    created_at: datetime
    status: Literal["success", "error"]
    error: str | None = None
    extraction_result: dict[str, Any] | None = None

    @model_validator(mode="after")
    def set_id(self) -> "InfoExtractionResults":
        id_ = self.get_id(
            self.prompt_hash,
            self.schema_hash,
            self.document_id,
            self.api_request_info.model,
        )

        if self.id is not None:
            assert self.id == id_

        self.id = id_
        return self

    @staticmethod
    def get_prompt_hash(prompt_template: str) -> str:
        return hashlib.md5(prompt_template.encode("utf-8")).hexdigest()

    @staticmethod
    def get_schema_hash(schema: dict[str, Any]) -> str:
        return hashlib.md5(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def get_id(prompt_hash: str, schema_hash: str, document_id: str, model: str) -> str:
        concat_str = f"{prompt_hash}-{schema_hash}-{document_id}-{model}"
        return hashlib.md5(concat_str.encode("utf-8")).hexdigest()


class InformationExtractor:
    def __init__(
        self,
        client: ChatOpenAI,
        schema: dict[str, Any],
        prompt_template: str,
        metadata: dict[str, Any],
        max_concurrent_calls: int,
    ):
        self.prompt_template = PromptTemplate.from_template(
            prompt_template,
            template_format="f-string",
        )

        self.client = client.with_structured_output(
            self.raw_schema_to_structured_output(schema),
            method="json_schema",
            strict=True,
            include_raw=True,
        )
        self.chain = self.prompt_template | self.client
        self.semaphore = asyncio.Semaphore(max_concurrent_calls)

        self.model_name = client.model_name
        self.metadata = metadata

    async def __call__(
        self,
        documents: list[dict[str, Any]],
    ) -> list[InfoExtractionResults]:
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
    async def extract_information(self, document: dict[str, Any]) -> InfoExtractionResults:
        assert (field in document for field in ["_id", "full_text"])

        async with self.semaphore:
            try:
                result = await self.chain.ainvoke(
                    {
                        "context": document["full_text"],
                    }
                )
                token_usage = result["raw"].response_metadata["token_usage"]

                return InfoExtractionResults(
                    **self.metadata,
                    document_id=document["_id"],
                    api_request_info=ApiRequestInfo(
                        model=self.model_name,
                        prompt_tokens=token_usage["prompt_tokens"],
                        completion_tokens=token_usage["completion_tokens"],
                        reasoning_tokens=(
                            token_usage.get("completion_tokens_details", {}).get("reasoning_tokens")
                        ),
                        finish_reason=result["raw"].response_metadata["finish_reason"],
                    ),
                    created_at=datetime.now(),
                    status="success",
                    extraction_result=result["parsed"],
                )

            except (BadRequestError, ContentFilterFinishReasonError) as err:
                logger.error(f"Error processing document {document['_id']}: {err}")
                return InfoExtractionResults(
                    **self.metadata,
                    document_id=document["_id"],
                    api_request_info=ApiRequestInfo(
                        model=self.model_name,
                        prompt_tokens=0,
                        completion_tokens=0,
                        reasoning_tokens=0,
                        finish_reason=None,
                    ),
                    created_at=datetime.now(),
                    status="error",
                    error=str(err),
                    extraction_result=None,
                )

            except LengthFinishReasonError as err:
                logger.error(f"Error processing document {document['_id']}: {err}")
                return InfoExtractionResults(
                    **self.metadata,
                    document_id=document["_id"],
                    api_request_info=ApiRequestInfo(
                        model=self.model_name,
                        prompt_tokens=err.completion.usage.prompt_tokens,
                        completion_tokens=err.completion.usage.completion_tokens,
                        reasoning_tokens=err.completion.usage.completion_tokens_details.reasoning_tokens,
                        finish_reason="length_finish_reason_error",
                    ),
                    created_at=datetime.now(),
                    status="error",
                    error=str(err),
                    extraction_result=None,
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
            InfoExtractionResults.get_id(
                info_extractor.metadata["prompt_hash"],
                info_extractor.metadata["schema_hash"],
                doc["_id"],
                info_extractor.model_name,
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
        results: list[InfoExtractionResults],
    ) -> None:
        docs_to_save = []
        for res in results:
            doc = res.model_dump()
            doc["_id"] = doc.pop("id")
            docs_to_save.append(doc)

        if docs_to_save:
            self.output_db.update_or_insert_documents(docs_to_save)


if __name__ == "__main__":
    app()
