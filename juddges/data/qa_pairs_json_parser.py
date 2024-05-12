import re
from json.decoder import JSONDecodeError
from typing import Any, Callable, Dict, List

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.json import parse_partial_json  # type: ignore
from langchain_core.outputs import Generation
from langchain_core.utils.json import _parse_json

CUSTOM_PARSE_JSON_MARKDOWN = re.compile(
    r"""
        ```(?:json)  # Start of JSON string, `json` is required
        ([^`]+)      # JSON string content, JSON cannot contain backticks
        (?:```)?     # Optional closing backticks. Handles scenario where LLM output is cut off
                     # and JSON is still valid for `json` Python module
    """,
    flags=re.IGNORECASE + re.VERBOSE,
)


class QAPairsJsonParser(JsonOutputParser):
    """JsonOutputParser for QA pairs output that handles multiple JSON strings
    with modified `parse_json_markdown`.
    """

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        text = result[0].text
        text = text.strip()
        if partial:
            try:
                return parse_json_markdown(text)
            except JSONDecodeError:
                return None
        else:
            try:
                return parse_json_markdown(text)
            except JSONDecodeError as e:
                msg = f"Invalid json output: {text}"
                raise OutputParserException(msg, llm_output=text) from e


def parse_json_markdown(
    json_string: str, *, parser: Callable[[str], Any] = parse_partial_json
) -> Dict[Any, Any]:
    """Modified version of `langchain_core.output_parsers.json:parse_json_markdown`

    Fixes: JSONDecodeError when parsing Chain-of-thoughts like prompt/output that contains multiple JSON strings
    """
    try:
        return _parse_json(json_string, parser=parser)
    except JSONDecodeError:
        # Try to find the last JSON string within triple backticks
        match = CUSTOM_PARSE_JSON_MARKDOWN.findall(json_string)

        # If no match found, assume the entire string is a JSON string
        if match is None:
            json_str = json_string
        else:
            # If match found, use the content within the backticks
            json_str = match[-1]
    return _parse_json(json_str, parser=parser)
