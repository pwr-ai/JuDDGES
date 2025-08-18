import asyncio
from typing import Any

import tiktoken
from langchain_openai import ChatOpenAI
from loguru import logger
from tqdm.asyncio import tqdm_asyncio

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

    def estimate_token_count(self) -> float:
        """Estimate the cost of evaluating a single item."""
        logger.warning(
            "Estimating token count ignores json_schema and computes number of tokens only for input messages"
        )
        if self.judge_name.startswith("gpt-4.1"):
            enc = tiktoken.encoding_for_model("gpt-4o")
            logger.warning(
                "For gpt-4.1 defaults to gpt-4o encoding (gpt-4.1 not handled by tiktoken yet)."
            )
        else:
            enc = tiktoken.encoding_for_model(self.judge_name)
        parsed_preds = self.pred_loader.load_predictions_from_file()
        dataset_messages = self.prepare_eval_messages(parsed_preds)
        return sum(self.count_tokens(enc, messages) for messages in dataset_messages.values())

    def count_tokens(self, enc: tiktoken.Encoding, messages: list[dict[str, str]]) -> int:
        """Credit: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb"""
        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(enc.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

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
