import re
from typing import Any, List, Callable
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.json import parse_partial_json, _parse_json
from langchain_core.outputs import Generation
from langchain_core.exceptions import OutputParserException
from json.decoder import JSONDecodeError

from juddges.data.synthetic.patterns import CUSTOM_PARSE_JSON_MARKDOWN


class QAPairsJsonParser(JsonOutputParser):

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
) -> dict:
    """Modified version of `langchain_core.output_parsers.json:parse_json_markdown`

        Fixes: JSONDecodeError when parsing CoT like output that contains multiple JSON strings
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
