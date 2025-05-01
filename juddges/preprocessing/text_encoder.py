from typing import Any

from torch import Tensor
from transformers import PreTrainedTokenizer

from juddges.preprocessing.context_truncator import ContextTruncatorTiktoken


class TokenizerEncoder:
    def __init__(self, final_input_field: str, tokenizer: PreTrainedTokenizer):
        self.final_input_field = final_input_field
        self.tokenizer = tokenizer

    def __call__(self, batch: dict[str, list[Any]]) -> dict[str, Tensor]:
        return self.tokenizer(
            batch[self.final_input_field],
            padding=False,
            truncation=False,
            return_tensors="pt",
            return_attention_mask=True,
            return_special_tokens_mask=False,
        )


class TextEncoderForEvalPlainTextFormat:
    def __init__(self, truncator: ContextTruncatorTiktoken):
        self.truncator = truncator

    def __call__(self, item: dict[str, Any]) -> dict[str, str]:
        truncated_context = self.truncator(
            prompt=item["prompt"], context=item["context"], output=item["output"]
        )
        return {"final_input": item["prompt"].format(context=truncated_context)}
