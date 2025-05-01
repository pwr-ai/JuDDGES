from unittest.mock import MagicMock

import pytest

from juddges.config import PromptInfoExtractionConfig
from juddges.preprocessing.formatters import ConversationFormatter

PROMPT_TEMPLATE = """
LANGUAGE: {language}
SCHEMA: {schema}

Extract information from: {context}
"""
FORMATTED_PROMPT = """
LANGUAGE: en
SCHEMA: {
  "field": {
    "type": "string"
  }
}

Extract information from: Sample text
"""


@pytest.fixture
def prompt():
    return PromptInfoExtractionConfig(
        language="en",
        ie_schema={"field": {"type": "string"}},
        content=PROMPT_TEMPLATE,
    )


@pytest.fixture
def formatted_prompt():
    return FORMATTED_PROMPT.strip()


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = (
        "<s>[INST] {user_message} [/INST] {assistant_message}</s>"
    )
    return tokenizer


@pytest.fixture
def formatter_with_output(prompt, mock_tokenizer):
    return ConversationFormatter(
        tokenizer=mock_tokenizer,
        prompt=prompt,
        dataset_context_field="text",
        dataset_output_field="output",
        use_output=True,
    )


@pytest.fixture
def formatter_without_output(prompt, mock_tokenizer):
    return ConversationFormatter(
        tokenizer=mock_tokenizer,
        prompt=prompt,
        dataset_context_field="text",
        dataset_output_field="output",
        use_output=False,
    )


def test_formatter_with_output(formatter_with_output, formatted_prompt, mock_tokenizer):
    item = {"text": "Sample text", "output": "Extracted info"}

    result = formatter_with_output(item)

    mock_tokenizer.apply_chat_template.assert_called_once_with(
        conversation=[
            {"role": "user", "content": formatted_prompt},
            {"role": "assistant", "content": "Extracted info"},
        ],
        tokenize=False,
        add_special_tokens=True,
        add_generation_prompt=True,
    )
    assert result == {"final_input": "<s>[INST] {user_message} [/INST] {assistant_message}</s>"}


def test_formatter_without_output(formatter_without_output, formatted_prompt, mock_tokenizer):
    item = {"text": "Sample text", "output": "Extracted info"}

    result = formatter_without_output(item)

    mock_tokenizer.apply_chat_template.assert_called_once_with(
        conversation=[{"role": "user", "content": formatted_prompt}],
        tokenize=False,
        add_special_tokens=True,
        add_generation_prompt=True,
    )
    assert result == {"final_input": "<s>[INST] {user_message} [/INST] {assistant_message}</s>"}


def test_formatter_with_missing_output_field(prompt, mock_tokenizer):
    formatter = ConversationFormatter(
        tokenizer=mock_tokenizer,
        prompt=prompt,
        dataset_context_field="text",
        dataset_output_field=None,
        use_output=True,
    )

    item = {"text": "Sample text"}

    with pytest.raises(KeyError):
        formatter(item)


def test_formatter_with_missing_context_field(prompt, mock_tokenizer):
    formatter = ConversationFormatter(
        tokenizer=mock_tokenizer,
        prompt=prompt,
        dataset_context_field="text",
        dataset_output_field="output",
        use_output=True,
    )

    item = {"output": "Extracted info"}

    with pytest.raises(KeyError):
        formatter(item)
