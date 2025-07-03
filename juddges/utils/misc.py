import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from loguru import logger

yaml_pattern: re.Pattern = re.compile(r"```(?:ya?ml)?(?P<yaml>[^`]*)", re.MULTILINE | re.DOTALL)


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


def load_yaml(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: Path, **kwargs) -> None:
    """Saves a dictionary to a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4, **kwargs)


def save_jsonl(data: list[dict], path: Path) -> None:
    """Saves a list of dictionaries to a JSONL file."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_json(path: Path) -> Any:
    """Loads a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    """Loads a JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


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
