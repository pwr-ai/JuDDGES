import re
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
