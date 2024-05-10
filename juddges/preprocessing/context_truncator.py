import warnings

from tokenizers.implementations import BaseTokenizer


class ContextTruncator:
    def __init__(self, tokenizer: BaseTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

        empty_messages = [
            {"role": "user", "content": ""},
        ]
        empty_messages.append({"role": "assistant", "content": ""})

        self.empty_messages_length = len(
            self.tokenizer.apply_chat_template(
                empty_messages, tokenize=True, return_dict=True
            ).data["input_ids"]
        )

    def __call__(self, prompt: str, context: str, output: str | None = None) -> str:
        if output:
            prompt_length, output_length = self.tokenizer(
                [prompt, output], return_length=True, add_special_tokens=False
            )["length"]
        else:
            prompt_length, output_length = self.tokenizer(
                [prompt], return_length=True, add_special_tokens=False
            )["length"][0], 0

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
