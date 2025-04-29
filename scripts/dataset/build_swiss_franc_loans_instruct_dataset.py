import hashlib
import json
from pathlib import Path
from textwrap import dedent
from typing import Any

import pandas as pd
import typer
import yaml
from loguru import logger
from tabulate import tabulate
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from juddges.prompts.information_extraction import SWISS_FRANC_LOAN_SCHEMA
from juddges.utils.misc import log_size_change, parse_ie_schema, validate_yaml

YAML_OUTPUT = "```yaml\n{label}\n```"
INSTRUCTION_TEMPLATE = """
LANGUAGE = "PL"
Jesteś asystentem odpowiedzialnym za wydobywanie informacji z polskich orzeczeń sądowych.
Ściśle na podstawie poniżej podanego SCHEMATU, wydobądź informacje z zadanego ORZECZENIA. Jeśli dana informacja nie jest zawarta w orzeczeniu, pozostaw pole z wartością null.
Wynik zwróc w formacie YAML.

=====
SHEMAT
{schema}
=====
ORZECZENIE
{{context}}
======
Wynik:
"""
DEFAULT_MAX_TOKENS = 64_000
DEFAULT_TOKENIZER_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def main(
    train_ds_path: Path = typer.Option(..., help="Path to the train dataset"),
    test_ds_path: Path = typer.Option(..., help="Path to the test dataset"),
    threshold_tokens: int | None = typer.Option(
        DEFAULT_MAX_TOKENS, help="Maximum number of tokens to use for the dataset"
    ),
    tokenizer_name: str = typer.Option(DEFAULT_TOKENIZER_NAME, help="Name of the tokenizer to use"),
    output_dir: Path = typer.Option(..., help="Path to the output directory"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_pickle(train_ds_path)
    test_df = pd.read_pickle(test_ds_path)

    assert train_df.columns.tolist() == test_df.columns.tolist()

    train_df = remove_duplicates(train_df)
    test_df = remove_duplicates(test_df)

    # assert train_df["schema"][0] == SWISS_FRANC_LOAN_SCHEMA
    schema = parse_ie_schema(SWISS_FRANC_LOAN_SCHEMA)
    instruction_template = dedent(INSTRUCTION_TEMPLATE).strip()

    train_instruct_dataset = format_instruction(
        train_df,
        instruction_template,
        schema,
        SWISS_FRANC_LOAN_SCHEMA,
    )
    original_train_size = len(train_instruct_dataset)
    train_instruct_dataset = filter_by_schema_mismatch(schema, train_instruct_dataset)
    logger.info(
        f"Filtered {original_train_size - len(train_instruct_dataset)} items out of {original_train_size}"
    )

    test_instruct_dataset = format_instruction(
        test_df,
        instruction_template,
        schema,
        SWISS_FRANC_LOAN_SCHEMA,
    )
    original_test_size = len(test_instruct_dataset)
    test_instruct_dataset = filter_by_schema_mismatch(schema, test_instruct_dataset)
    logger.info(
        f"Filtered {original_test_size - len(test_instruct_dataset)} items out of {original_test_size}"
    )

    if threshold_tokens is not None:
        train_instruct_dataset = filter_by_num_tokens(
            train_instruct_dataset,
            tokenizer_name,
            threshold_tokens,
        )
        test_instruct_dataset = filter_by_num_tokens(
            test_instruct_dataset,
            tokenizer_name,
            threshold_tokens,
        )

    with open(output_dir / "train.jsonl", "w") as f:
        json.dump(train_instruct_dataset, f, indent="\t", ensure_ascii=False)

    with open(output_dir / "test.jsonl", "w") as f:
        json.dump(test_instruct_dataset, f, indent="\t", ensure_ascii=False)

    # Print dataset sizes
    sizes_table = [
        ["Split", "Size"],
        ["Train", len(train_instruct_dataset)],
        ["Test", len(test_instruct_dataset)],
    ]
    print("\nDataset sizes:")
    print(tabulate(sizes_table, headers="firstrow", tablefmt="grid"))

    # Save dataset info
    dataset_info = {
        "source_files": {"train": str(train_ds_path), "test": str(test_ds_path)},
        "parameters": {"tokenizer_name": tokenizer_name, "threshold_tokens": threshold_tokens},
        "output_sizes": {"train": len(train_instruct_dataset), "test": len(test_instruct_dataset)},
        "instruction_template": instruction_template,
        "schema": schema,
    }

    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent="\t", ensure_ascii=False)


@log_size_change
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df["hash"] = df.apply(lambda row: hashlib.md5(str(row["text"]).encode()).hexdigest(), axis=1)
    df = df.drop_duplicates(subset=["hash"]).drop(columns=["hash"])
    return df


def format_instruction(
    df: pd.DataFrame,
    instruction_template: str,
    schema: dict[str, dict[str, Any]],
    schema_formatted: str,
) -> list[dict[str, str]]:
    instruct_dataset = []
    for _, item in tqdm(df.iterrows(), total=len(df), desc="Generating instructions"):
        gold = YAML_OUTPUT.format(
            label=yaml.dump({key: item[key] for key in schema.keys()}, allow_unicode=True)
        )
        instruct_dataset.append(
            {
                "prompt": instruction_template.format(schema=schema_formatted),
                "context": item["text"],
                "output": gold,
            }
        )
    return instruct_dataset


@log_size_change
def filter_by_num_tokens(
    dataset: list[dict[str, str]],
    tokenizer_name: str,
    threshold_tokens: int,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    lengths = tokenizer(
        [item["prompt"].format(context=item["context"]) for item in dataset],
        return_length=True,
    )["length"]
    return [item for item, length in zip(dataset, lengths) if length <= threshold_tokens]


def filter_by_schema_mismatch(
    schema: dict[str, dict[str, Any]],
    dataset: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [item for item in dataset if validate_yaml(item["output"], schema)["num_errors"] == 0]


if __name__ == "__main__":
    typer.run(main)
