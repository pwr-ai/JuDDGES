from collections import defaultdict
from typing import Any, Literal

import dateparser
from langchain_core.utils.json import parse_json_markdown

from juddges.utils.misc import parse_yaml


def validate_output_structure(
    markdown_str: str,
    schema: dict[str, dict[str, Any]],
    format: Literal["json", "yaml"] = "yaml",
) -> dict[str, Any]:
    """
    Validate YAML against schema.
    Returns a dictionary of errors found during validation.
    """
    if format == "json":
        data = parse_json_markdown(markdown_str)
    elif format == "yaml":
        data = parse_yaml(markdown_str)
    else:
        raise ValueError(f"Invalid format: {format}")

    if not isinstance(data, dict):
        raise ValueError(f"Invalid data parsed (must be a dictionary): {data}")

    errors = defaultdict(list)
    for field, field_schema in schema.items():
        if field not in data:
            errors["error_type"].append("missing")
            errors["field_name"].append(field)
            errors["field_type"].append(field_schema["type"])
            errors["field_schema"].append(field_schema)
            errors["value"].append(None)
            continue

        value = data[field]
        if value is None or value == "":
            continue

        field_type = field_schema["type"]

        if field_type == "list":
            if not isinstance(value, list):
                errors["error_type"].append("bad_value")
                errors["field_name"].append(field)
                errors["field_type"].append(field_type)
                errors["field_schema"].append(field_schema)
                errors["value"].append(value)
            else:
                # Validate each item in the list
                for item in value:
                    if "items" in field_schema:
                        item_errors = validate_single_item(item, field_schema["items"])
                        for item_error in item_errors:
                            errors["error_type"].append(item_error["error_type"])
                            errors["field_name"].append(field)
                            errors["field_type"].append(item_error["field_type"])
                            errors["field_schema"].append(item_error["field_schema"])
                            errors["value"].append(item_error["value"])
        else:
            # Handle non-list fields
            item_errors = validate_single_item(value, field_schema)
            for item_error in item_errors:
                errors["error_type"].append(item_error["error_type"])
                errors["field_name"].append(field)
                errors["field_type"].append(item_error["field_type"])
                errors["field_schema"].append(item_error["field_schema"])
                errors["value"].append(item_error["value"])

    for field, value in data.items():
        if field not in schema:
            errors["error_type"].append("unknown_field")
            errors["field_name"].append(field)
            errors["field_type"].append(None)
            errors["field_schema"].append(None)
            errors["value"].append(value)

    errors["num_errors"] = len(errors["error_type"])

    return dict(errors)


def validate_single_item(value: Any, item_schema: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Validate a single item against its schema.
    Returns a list of errors for this item.
    """
    errors = []
    item_type = item_schema["type"]

    if item_type == "enum":
        if value not in item_schema["choices"]:
            errors.append(
                {
                    "error_type": "bad_value",
                    "field_type": item_type,
                    "field_schema": item_schema,
                    "value": value,
                }
            )
    elif item_type == "date":
        if not isinstance(value, str):
            errors.append(
                {
                    "error_type": "bad_value",
                    "field_type": item_type,
                    "field_schema": item_schema,
                    "value": value,
                }
            )
        else:
            try:
                dateparser.parse(value)
            except ValueError:
                errors.append(
                    {
                        "error_type": "invalid_date_format",
                        "field_type": item_type,
                        "field_schema": item_schema,
                        "value": value,
                    }
                )
    elif item_type == "int":
        try:
            int(value)
        except (ValueError, TypeError):
            errors.append(
                {
                    "error_type": "bad_value",
                    "field_type": item_type,
                    "field_schema": item_schema,
                    "value": value,
                }
            )
    elif item_type == "string":
        if not isinstance(value, str):
            errors.append(
                {
                    "error_type": "bad_value",
                    "field_type": item_type,
                    "field_schema": item_schema,
                    "value": value,
                }
            )

    return errors
