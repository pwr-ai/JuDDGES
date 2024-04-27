import re


CUSTOM_PARSE_JSON_MARKDOWN = re.compile(
    pattern=r"```(?:json)?([^`]+)(?:```)?\s*$", flags=re.IGNORECASE
)
