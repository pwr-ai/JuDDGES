import asyncio
from typing import Any

import litellm
from langchain_openai import ChatOpenAI
from loguru import logger
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm

from juddges.llm_as_judge.base import EvalResults, ItemEvalResult, StructuredOutputJudgeBase
from juddges.llm_as_judge.data_model import ParsedPredictions, PredictionLoader


class StructuredOutputJudge(StructuredOutputJudgeBase):
    DEFAULT_MAX_CONCURRENT_CALLS = 5

    def __init__(
        self,
        client: ChatOpenAI,
        pred_loader: PredictionLoader,
        system_prompt: str,
        user_prompt: str,
        max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS,
        verbose: bool = False,
    ):
        """Initialize the LLM judge with a client and concurrency limit."""
        super().__init__(
            pred_loader=pred_loader,
            judge_name=client.model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        self.client = client
        self.chain = client.with_structured_output(
            self.structured_response_schema_from_extraction_schema,
            method="json_schema",
            strict=True,
        )
        self.max_concurrent_calls = max_concurrent_calls
        self.semaphore = asyncio.Semaphore(self.max_concurrent_calls)
        self.verbose = verbose

    async def evaluate(self) -> EvalResults:
        """Evaluate a batch of examples concurrently."""
        parsed_preds = self.pred_loader.load_predictions_from_file()
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

        return self.merge_judge_results_with_failed_items(parsed_preds, eval_results)

    async def evaluate_single_item(
        self,
        messages: list[dict[str, str]],
        **results_kwargs: Any,
    ) -> ItemEvalResult:
        """Evaluate a single example using the LLM judge."""
        async with self.semaphore:
            try:
                scores_dict = await self.chain.ainvoke(messages)
            except Exception as e:
                logger.error(f"Error evaluating item: {e}")
                return ItemEvalResult(
                    status="judge_error",
                    error=str(e),
                    result=self.get_zero_scores(),
                    **results_kwargs,
                )
        return ItemEvalResult.from_success(scores_dict, **results_kwargs)

    def estimate_prefill_cost(self) -> float:
        """Estimate the cost of evaluating a single item."""
        parsed_preds = self.pred_loader.load_predictions_from_file()
        dataset_messages = self.prepare_eval_messages(parsed_preds)
        return sum(
            litellm.completion_cost(
                model=self.client.model_name,
                messages=messages,
            )
            for messages in tqdm(dataset_messages.values(), desc="Estimating prefill cost")
        )

    def merge_judge_results_with_failed_items(
        self,
        parsed_preds: ParsedPredictions,
        eval_results: dict[int, ItemEvalResult],
    ) -> EvalResults:
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

        return EvalResults(results=results, ie_schema=self.pred_loader.schema)
