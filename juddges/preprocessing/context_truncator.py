import warnings
from abc import ABC, abstractmethod
from typing import Any

import tiktoken
from tokenizers.implementations import BaseTokenizer


class ContextTruncatorBase(ABC):
    def __init__(self, max_length: int):
        self.max_length = max_length

    @abstractmethod
    def __call__(self, prompt: str, context: str, output: str | None = None) -> dict[str, Any]:
        """Truncates and returns the inner context of the input."""


class ContextTruncator(ContextTruncatorBase):
    def __init__(self, tokenizer: BaseTokenizer, max_length: int):
        super().__init__(max_length)
        self.tokenizer = tokenizer

        empty_messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
        ]

        try:
            self.empty_messages_length = len(
                self.tokenizer.apply_chat_template(empty_messages, tokenize=True)
            )
        except ValueError:
            self.empty_messages_length = 0

    def __call__(self, prompt: str, context: str, output: str | None = None) -> dict[str, Any]:
        if output:
            prompt_length, output_length = self.tokenizer(
                [prompt, output], return_length=True, add_special_tokens=False
            )["length"]
        else:
            prompt_length, output_length = (
                self.tokenizer([prompt], return_length=True, add_special_tokens=False)["length"][0],
                0,
            )

        max_context_length = (
            self.max_length - prompt_length - output_length - self.empty_messages_length
        )
        if max_context_length <= 0:
            warnings.warn(
                f"Context was truncated to 0 tokens. "
                f"The prompt and output are too long for the max_length of {self.max_length}."
            )
            return {"context": "", "num_truncated_tokens": None}

        full_context_ids = self.tokenizer(
            context,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]
        truncated_context_ids = full_context_ids[:max_context_length]
        num_truncated_tokens = len(full_context_ids) - len(truncated_context_ids)
        truncated_ratio = num_truncated_tokens / len(full_context_ids)
        return {
            "context": self.tokenizer.decode(truncated_context_ids),
            "num_truncated_tokens": num_truncated_tokens,
            "truncated_ratio": truncated_ratio,
        }


class ContextTruncatorTiktoken(ContextTruncatorBase):
    """Simplified context truncation for OpenAI models."""

    def __init__(self, model: str, max_length: int):
        super().__init__(max_length)

        warnings.warn("Truncator for OpenAI doesn't account for special tokens!")
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(model)

    def __call__(self, prompt: str, context: str, output: str | None = None) -> dict[str, Any]:
        prompt_length = len(self.tokenizer.encode(prompt))

        if output is not None:
            output_length = len(self.tokenizer.encode(output))
        else:
            output_length = 0

        context_length = self.max_length - prompt_length - output_length
        if context_length <= 0:
            raise ValueError("Prompt and output too long to keep the context!")

        full_context_ids = self.tokenizer.encode(context)
        truncated_context_ids = full_context_ids[:context_length]
        truncated_context = self.tokenizer.decode(truncated_context_ids)
        num_truncated_tokens = len(full_context_ids) - len(truncated_context_ids)
        truncated_ratio = num_truncated_tokens / len(full_context_ids)
        return {
            "context": truncated_context,
            "num_truncated_tokens": num_truncated_tokens,
            "truncated_ratio": truncated_ratio,
        }
