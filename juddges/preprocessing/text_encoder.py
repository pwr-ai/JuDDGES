from torch import Tensor
from transformers import PreTrainedTokenizer
from juddges.preprocessing.context_truncator import ContextTruncator


class TextEncoderForEval:
    """Text encoder preparing evaluation data for instruction dataset (chat-formatted)."""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int, padding: bool | str):
        self.tokenizer = tokenizer
        self.truncator = ContextTruncator(tokenizer, max_length)
        self.max_length = max_length
        self.padding = padding

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, Tensor]:
        texts = []

        for prompt, context in zip(batch["prompt"], batch["context"]):
            truncated_context = self.truncator(prompt, context)
            input_message = prompt.format(context=truncated_context)
            input_chat = [{"role": "user", "content": input_message}]
            final_input = self.tokenizer.apply_chat_template(
                input_chat,
                add_generation_prompt=True,
                tokenize=False,
            )
            texts.append(final_input)

        tokenized = self.tokenizer(
            texts,
            padding=self.padding,
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
            return_attention_mask=False,
            return_special_tokens_mask=False,
        )

        return tokenized
