import pytest

from juddges.config import PromptInfoExtractionConfig
from juddges.preprocessing.formatters import ConversationFormatter

PROMPT_TEMPLATE = """
LANGUAGE: {{language}}
SCHEMA: {{schema}}

Extract information from: {{context}}
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
def formatter_with_output(prompt):
    return ConversationFormatter(
        prompt=prompt, dataset_context_field="text", dataset_output_field="output", use_output=True
    )


@pytest.fixture
def formatter_without_output(prompt):
    return ConversationFormatter(
        prompt=prompt, dataset_context_field="text", dataset_output_field="output", use_output=False
    )


def test_formatter_with_output(formatter_with_output, formatted_prompt):
    item = {"text": "Sample text", "output": "Extracted info"}

    result = formatter_with_output(item)

    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == formatted_prompt
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["content"] == "Extracted info"


def test_formatter_without_output(formatter_without_output, formatted_prompt):
    item = {"text": "Sample text", "output": "Extracted info"}

    result = formatter_without_output(item)

    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == formatted_prompt


def test_formatter_with_missing_output_field(prompt):
    formatter = ConversationFormatter(
        prompt=prompt,
        dataset_context_field="text",
        dataset_output_field=None,
        use_output=True,
    )

    item = {"text": "Sample text"}

    with pytest.raises(KeyError):
        formatter(item)


def test_formatter_with_missing_context_field(prompt):
    formatter = ConversationFormatter(
        prompt=prompt,
        dataset_context_field="text",
        dataset_output_field="output",
        use_output=True,
    )

    item = {"output": "Extracted info"}

    with pytest.raises(KeyError):
        formatter(item)
