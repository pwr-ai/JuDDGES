import warnings
from abc import ABC, abstractmethod
from typing import Any

from transformers import PreTrainedTokenizer

from juddges.config import PromptInfoExtractionConfig


class Formatter(ABC):
    @abstractmethod
    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        pass


class ConversationFormatter(Formatter):
    FINAL_INPUT_FIELD = "final_input"
    MESSAGES_FIELD = "messages"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt: PromptInfoExtractionConfig,
        dataset_context_field: str,
        dataset_output_field: str | None,
        use_output: bool,
        format_as_chat: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.dataset_context_field = dataset_context_field
        self.dataset_output_field = dataset_output_field
        self.use_output = use_output
        self.format_as_chat = format_as_chat

    def __call__(self, item: dict[str, Any]) -> dict[str, str]:
        final_input = self.prompt.render(context=item[self.dataset_context_field])
        if self.use_output:
            messages = [
                {"role": "user", "content": final_input},
                {"role": "assistant", "content": item[self.dataset_output_field]},
            ]

        else:
            messages = [{"role": "user", "content": final_input}]

        if self.format_as_chat:
            return {
                self.FINAL_INPUT_FIELD: self.tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False,
                    add_special_tokens=True,
                    add_generation_prompt=True,
                ),
            }
        else:
            return {self.MESSAGES_FIELD: messages}


def format_to_conversations(
    entry: dict[str, Any],
    prompt_field: str,
    context_field: str,
    output_field: str,
) -> dict[str, list[dict[str, str]]]:
    warnings.warn(
        "format_to_conversations is deprecated. Use ConversationFormatter instead.",
        DeprecationWarning,
    )

    first_message = entry[prompt_field].format(context=entry[context_field])
    messages = [
        {"role": "user", "content": first_message},
        {"role": "assistant", "content": entry[output_field]},
    ]
    return {"messages": messages}
