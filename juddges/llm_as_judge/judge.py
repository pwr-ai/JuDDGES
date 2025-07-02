import asyncio
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from juddges.llm_as_judge.base import EvalResults, ItemEvalResult, StructuredOutputJudgeBase


class StructuredOutputJudge(StructuredOutputJudgeBase):
    DEFAULT_MAX_CONCURRENT_CALLS = 5

    def __init__(
        self,
        client: ChatOpenAI,
        predictions_dir: Path,
        max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS,
        verbose: bool = False,
    ):
        """Initialize the LLM judge with a client and concurrency limit."""
        super().__init__(predictions_dir, client.model_name)
        self.client = client
        self.client = self.client.with_structured_output(
            self.structured_response_schema_from_extraction_schema,
            method="json_schema",
            strict=True,
        )
        self.max_concurrent_calls = max_concurrent_calls
        self.semaphore = asyncio.Semaphore(self.max_concurrent_calls)
        self.verbose = verbose

    async def evaluate(self) -> EvalResults:
        """Evaluate a batch of examples concurrently."""
        parsed_preds = self.load_predictions()
        dataset_messages = self.prepare_eval_messages(parsed_preds)
        tasks = [
            self.evaluate_single_item(
                dataset_messages[idx],
                missing_keys=parsed_preds.missing_keys[idx],
                extra_keys=parsed_preds.extra_keys[idx],
            )
            for idx in dataset_messages.keys()
        ]
        results_list = await tqdm_asyncio.gather(
            *tasks,
            desc="Evaluating predictions",
            disable=not self.verbose,
        )
        eval_results = {idx: res for idx, res in zip(dataset_messages.keys(), results_list)}

        results = []
        for idx in range(parsed_preds.num_items):
            try:
                res = eval_results[idx]
            except KeyError:
                res = ItemEvalResult(
                    status="parsing_error",
                    error=parsed_preds.errors[idx],
                    result=self.get_zero_scores(),
                )
            results.append(res)

        return EvalResults(results=results, ie_schema=self.schema)

    async def evaluate_single_item(
        self,
        messages: list[dict[str, str]],
        **results_kwargs: Any,
    ) -> ItemEvalResult:
        """Evaluate a single example using the LLM judge."""
        async with self.semaphore:
            try:
                scores_dict = await self.client.ainvoke(messages)
            except Exception as e:
                logger.error(f"Error evaluating item: {e}")
                return ItemEvalResult(
                    status="judge_error",
                    error=str(e),
                    result=self.get_zero_scores(),
                    **results_kwargs,
                )
        return ItemEvalResult.from_success(scores_dict, **results_kwargs)
