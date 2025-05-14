import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import typer
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from tabulate import tabulate
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from juddges.utils.misc import log_size_change
from juddges.utils.validate_schema import validate_output_structure

DEFAULT_MAX_TOKENS = 64_000
DEFAULT_TOKENIZER_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def main(
    dataset_source_path: Path = typer.Option(..., help="Path to the dataset source file"),
    threshold_tokens: int | None = typer.Option(
        DEFAULT_MAX_TOKENS, help="Maximum number of tokens to use for the dataset"
    ),
    schema_path: Path = typer.Option(..., help="Path to the schema file"),
    tokenizer_name: str = typer.Option(DEFAULT_TOKENIZER_NAME, help="Name of the tokenizer to use"),
    output_dir: Path = typer.Option(..., help="Path to the output directory"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        str(dataset_source_path),
        data_files={"annotated": "annotated.json", "test": "test.json"},
    )

    assert all(
        set(split_ds.column_names) == {"context", "output"} for split_ds in dataset.values()
    ), "All splits must have the same columns"

    dataset["test"] = get_test_items_matching_annotated(dataset)

    with open(schema_path, "r") as f:
        schema = yaml.safe_load(f)

    for split_name, split_ds in dataset.items():
        original_split_size = split_ds.num_rows
        dataset[split_name] = filter_by_schema_mismatch(split_ds, schema_path)
        logger.info(
            f"[{split_name}] Filtered {original_split_size - split_ds.num_rows} items out of {original_split_size}"
        )

    if threshold_tokens is not None:
        for split_name, split_ds in dataset.items():
            original_split_size = split_ds.num_rows
            dataset[split_name] = filter_too_long_contexts(
                split_ds,
                tokenizer_name,
                threshold_tokens,
            )
            logger.info(
                f"[{split_name}] Filtered {original_split_size - split_ds.num_rows} items out of {original_split_size}"
            )

    # assert annotated is the same as test
    df_test = dataset["test"].to_pandas()
    df_annotated = dataset["annotated"].to_pandas()
    assert (df_test["context"] == df_annotated["context"]).all()

    for split_name, split_ds in dataset.items():
        split_ds.to_json(output_dir / f"{split_name}.json", orient="records", indent=4)

    # Print dataset sizes
    sizes_table = [
        ["Split", "Size"],
        *[[split_name, split_ds.num_rows] for split_name, split_ds in dataset.items()],
    ]
    print("\nDataset sizes:")
    print(tabulate(sizes_table, headers="firstrow", tablefmt="grid"))

    # Save dataset info
    dataset_info = {
        "source_files": {
            split_name: str(output_dir / f"{split_name}.json") for split_name in dataset.keys()
        },
        "parameters": {"tokenizer_name": tokenizer_name, "threshold_tokens": threshold_tokens},
        "output_sizes": {split_name: split_ds.num_rows for split_name, split_ds in dataset.items()},
        "schema": schema,
    }

    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent="\t", ensure_ascii=False)


@log_size_change
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df["hash"] = df.apply(lambda row: hashlib.md5(str(row["text"]).encode()).hexdigest(), axis=1)
    df = df.drop_duplicates(subset=["hash"]).drop(columns=["hash"])
    return df


def filter_by_schema_mismatch(ds: Dataset, schema_path: Path) -> Dataset:
    with open(schema_path, "r") as f:
        schema = yaml.safe_load(f)

    return ds.filter(
        lambda item: validate_output_structure(item["output"], schema, format="json")["num_errors"]
        == 0
    )


def get_test_items_matching_annotated(dataset: DatasetDict) -> Dataset:
    def get_matching_test_item(annotated_item: dict[str, Any]) -> dict[str, Any]:
        found_items = []
        for test_item in dataset["test"]:
            if annotated_item["context"] == test_item["context"]:
                found_items.append(test_item)
        if not len(found_items) == 1:
            raise ValueError(f"Found {len(found_items)} items")
        return found_items[0]

    test_items_matching_annotated = []
    for annotated_item in tqdm(dataset["annotated"], "Matching test and annotated items"):
        found_item = get_matching_test_item(annotated_item)
        assert found_item["context"] == annotated_item["context"]
        test_items_matching_annotated.append(found_item)

    return Dataset.from_pandas(pd.DataFrame(test_items_matching_annotated))


def format_instruction(
    df: pd.DataFrame,
    schema: dict[str, dict[str, Any]],
    output_format: Literal["yaml", "json"],
) -> list[dict[str, str]]:
    instruct_dataset = []
    for _, item in tqdm(df.iterrows(), total=len(df), desc="Generating instructions"):
        if output_format == "yaml":
            gold = yaml.dump({key: item[key] for key in schema.keys()}, allow_unicode=True)
        elif output_format == "json":
            gold = json.dumps({key: item[key] for key in schema.keys()})
        else:
            raise ValueError(f"Invalid output format: {output_format}")

        instruct_dataset.append(
            {
                "context": item["text"],
                "output": gold,
            }
        )
    return instruct_dataset


@log_size_change
def filter_too_long_contexts(
    dataset: Dataset,
    tokenizer_name: str,
    threshold_tokens: int,
) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    lengths = tokenizer(
        [item["context"] for item in dataset],
        return_length=True,
    )["length"]
    return dataset.filter(lambda _, idx: lengths[idx] <= threshold_tokens, with_indices=True)


if __name__ == "__main__":
    typer.run(main)
