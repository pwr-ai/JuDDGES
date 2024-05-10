from transformers import PreTrainedTokenizer
from juddges.preprocessing.context_truncator import ContextTruncator


class EvalEncoder:
    """Text encoder preparing evaluation data for instruction dataset (chat-formatted)."""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int, padding: bool | str):
        self.tokenizer = tokenizer
        self.truncator = ContextTruncator(tokenizer, max_length)
        self.max_length = max_length
        self.padding = padding

    def __call__(self, item: dict[str, str]):
        truncated_context = self.truncator(item["prompt"], item["context"])
        input_message = item["prompt"].format(context=truncated_context)
        input_chat = [{"role": "user", "content": input_message}]
        encoded = self.tokenizer.apply_chat_template(
            input_chat,
            add_generation_prompt=True,
            padding=self.padding,
            max_length=self.max_length,
            truncation=False,
        )
        if self.padding:
            assert len(encoded) == self.max_length

        return {"tokens": encoded}
