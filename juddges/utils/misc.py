import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from loguru import logger

yaml_pattern: re.Pattern = re.compile(r"```(?:ya?ml)?(?P<yaml>[^`]*)", re.MULTILINE | re.DOTALL)


def validate_yaml(
    yaml_str: str | dict[str, Any],
    schema: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Simple function to validate YAML against schema.
    Returns (is_valid, list_of_errors)
    """
    data = parse_yaml(yaml_str)

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

        if field_type == "enum" and value not in field_schema["choices"]:
            errors["error_type"].append("bad_value")
            errors["field_name"].append(field)
            errors["field_type"].append(field_type)
            errors["field_schema"].append(field_schema)
            errors["value"].append(value)
        elif field_type == "date" and not isinstance(value, str):
            errors["error_type"].append("bad_value")
            errors["field_name"].append(field)
            errors["field_type"].append(field_type)
            errors["field_schema"].append(field_schema)
            errors["value"].append(value)
        elif field_type == "list" and not isinstance(value, list):
            errors["error_type"].append("bad_value")
            errors["field_name"].append(field)
            errors["field_type"].append(field_type)
            errors["field_schema"].append(field_schema)
            errors["value"].append(value)

    for field, value in data.items():
        if field not in schema:
            errors["error_type"].append("unknown_field")
            errors["field_name"].append(field)
            errors["field_type"].append(None)
            errors["field_schema"].append(None)
            errors["value"].append(value)

    errors["num_errors"] = len(errors["error_type"])

    return dict(errors)


def parse_yaml(text: str) -> Any:
    """YAML parser taken from langchain.
    Credit: https://github.com/langchain-ai/langchain.
    """
    match = re.search(yaml_pattern, text.strip())
    yaml_str = ""
    if match:
        yaml_str = match.group("yaml")
    else:
        yaml_str = text

    return yaml.safe_load(yaml_str)


def save_yaml(data: Any, path: Path) -> None:
    """Saves a dictionary to a YAML file."""
    with open(path, "w") as f:
        yaml.dump(data, f)


def sort_dataset_by_input_length(ds: Dataset, field: str) -> tuple[Dataset, list[int]]:
    """Sorts a dataset by the length of a field.

    Args:
        ds (Dataset): dataset to sort
        field (str): field to sort by

    Returns:
        tuple[Dataset, list[int]]: sorted dataset and the reverse sort index
    """
    item_lenghts = torch.tensor(
        ds.map(lambda item: {"length": len(item[field])}, remove_columns=ds.column_names)["length"]
    )
    sort_idx = torch.argsort(item_lenghts, stable=True, descending=True)
    reverse_sort_idx = torch.argsort(sort_idx, stable=True).tolist()
    return ds.select(sort_idx), reverse_sort_idx


def log_size_change(func):
    """Decorator that logs the size change of a collection after applying a function.

    Args:
        func: Function that takes and returns a collection

    Returns:
        Wrapped function that logs size changes
    """

    def wrapper(*args, **kwargs):
        input_collection = args[0]
        input_size = len(input_collection)
        result = func(*args, **kwargs)
        output_size = len(result)

        logger.info(f"'{func.__name__}' changed size from {input_size} to {output_size}")
        return result

    return wrapper


def parse_ie_schema(spec: str) -> dict[str, dict[str, Any]]:
    """
    Very minimal parser:
      - splits each line on the three fixed separators
      - detects enum / list-of-enum and grabs the bracketed values
      - coerces "true"/"false" examples into bools
    """
    result = {}
    for line in spec.splitlines():
        line = line.strip()
        if not line:
            continue

        # 1) Split off the name
        name, _, rest = line.partition(":")

        # 2) Split out the type (before ", description:")
        typ_part, _, after_desc = rest.partition(", description:")

        # 3) Split out the description (before ", example:")
        desc_part, _, example_part = after_desc.partition(", example:")

        typ = typ_part.strip()
        desc = desc_part.strip().strip('"')
        ex_raw = example_part.strip()

        # 4) Coerce true/false to bool, strip quotes from strings
        if ex_raw.lower() == "true":
            example = True
        elif ex_raw.lower() == "false":
            example = False
        elif ex_raw.startswith('"') and ex_raw.endswith('"'):
            example = ex_raw[1:-1]
        else:
            example = ex_raw  # leave dates, numbers, lists, etc.

        # 5) Detect enums
        if typ.startswith("list of enum"):
            # extract inside the [ ... ]
            vals = typ[typ.find("[") + 1 : typ.rfind("]")]
            values = [v.strip() for v in vals.split(",")]
            result[name.strip()] = {
                "type": "list",
                "item_type": "enum",
                "choices": values,
                "description": desc,
                "example": example,
            }

        elif typ.startswith("enum"):
            vals = typ[typ.find("[") + 1 : typ.rfind("]")]
            values = [v.strip() for v in vals.split(",")]
            result[name.strip()] = {
                "type": "enum",
                "choices": values,
                "description": desc,
                "example": example,
            }

        else:
            # string, boolean, date (...)
            result[name.strip()] = {"type": typ, "description": desc, "example": example}

    return result
