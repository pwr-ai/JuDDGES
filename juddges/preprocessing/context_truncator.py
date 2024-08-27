import warnings
from abc import ABC, abstractmethod

import tiktoken
from tokenizers.implementations import BaseTokenizer


class ContextTruncatorBase(ABC):
    def __init__(self, max_length: int):
        self.max_length = max_length

    @abstractmethod
    def __call__(self, prompt: str, context: str, output: str | None = None) -> str:
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

    def __call__(self, prompt: str, context: str, output: str | None = None) -> str:
        if output:
            prompt_length, output_length = self.tokenizer(
                [prompt, output], return_length=True, add_special_tokens=False
            )["length"]
        else:
            prompt_length, output_length = (
                self.tokenizer([prompt], return_length=True, add_special_tokens=False)["length"][0],
                0,
            )

        context_length = (
            self.max_length - prompt_length - output_length - self.empty_messages_length
        )
        if context_length <= 0:
            warnings.warn(
                f"Context was truncated to 0 tokens. "
                f"The prompt and output are too long for the max_length of {self.max_length}."
            )
            return ""
        context_ids = self.tokenizer(
            context, max_length=context_length, truncation=True, add_special_tokens=False
        )["input_ids"]
        return self.tokenizer.decode(context_ids)


class ContextTruncatorTiktoken(ContextTruncatorBase):
    """Simplified context truncation for OpenAI models."""

    def __init__(self, model: str, max_length: int):
        super().__init__(max_length)

        warnings.warn("Truncator for OpenAI doesn't account for special tokens!")
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(model)

    def __call__(self, prompt: str, context: str, output: str | None = None) -> str:
        prompt_length = len(self.tokenizer.encode(prompt))

        if output is not None:
            output_length = len(self.tokenizer.encode(output))
        else:
            output_length = 0

        context_length = self.max_length - prompt_length - output_length
        if context_length <= 0:
            raise ValueError("Prompt and output too long to keep the context!")

        return self.tokenizer.decode(self.tokenizer.encode(context)[:context_length])
