from pprint import pprint
from typing import Any

import yaml
from datasets import Dataset, load_dataset

from juddges.utils.validate_schema import validate_output_structure

DATASET = "data/datasets/en/en_appealcourt_coded"
SCHEMA = "configs/ie_schema/en_appealcourt.yaml"


def main():
    with open(SCHEMA, "r") as f:
        schema = yaml.safe_load(f)

    ds = load_dataset(DATASET)

    for split, data in ds.items():
        errors = check_schema_mismatch(schema, data)
        print(f"Split: {split}")
        print(f"Number of errors: {sum(err['num_errors'] for err in errors)}")

        errors = [error for error in errors if error["num_errors"] > 0]
        pprint(errors)


def check_schema_mismatch(
    schema: dict[str, dict[str, Any]],
    dataset: Dataset,
) -> list[dict[str, str]]:
    return [validate_output_structure(item, schema, format="json") for item in dataset["output"]]


if __name__ == "__main__":
    main()
