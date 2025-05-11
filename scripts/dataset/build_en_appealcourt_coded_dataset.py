import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import typer
import yaml
from datasets import load_dataset
from langchain_core.utils.json import parse_json_markdown
from loguru import logger
from transformers import AutoTokenizer

EN_DATASET_REPO = "JuDDGES/en-appealcourt-coded-instruct_v02"
OUTPUT_COLUMN = "output_2"

NUM_PROC = int(os.getenv("NUM_PROC", 1))
TOKENIZER = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_OUTPUT_TOKENS = 5_000
MAX_INPUT_TOKENS = 30_000


def main(
    target_dir: Path = typer.Option(..., help="Path to save the processed dataset"),
    schema_file: Path = typer.Option(..., help="Path to the schema file"),
):
    target_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(EN_DATASET_REPO)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    stats = defaultdict(dict)
    stats["filtering_params"] = {
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "tokenizer": TOKENIZER,
        "output_column": OUTPUT_COLUMN,
    }

    with open(schema_file, "r") as f:
        schema = yaml.safe_load(f)

    for split_name, data in ds.items():
        data = data.rename_column(OUTPUT_COLUMN, "output").select_columns(["context", "output"])

        stats[split_name]["initial_size"] = len(data)
        data = data.map(
            lambda item: {
                "num_output_tokens": len(tokenizer(item["output"])[0]),
                "num_context_tokens": len(tokenizer(item["context"])[0]),
            },
            batched=False,
            num_proc=NUM_PROC,
            desc=f"[{split_name}] Calculating context/output tokens",
        )
        data = data.filter(
            lambda item: (item["num_output_tokens"] <= MAX_OUTPUT_TOKENS)
            and (item["num_context_tokens"] <= MAX_INPUT_TOKENS),
            desc=f"[{split_name}] Filtering by context/output tokens",
        )

        logger.info(f"[{split_name}] Filtered by context/output tokens: {data.num_rows}")
        stats[split_name]["filtered_by_context_or_output_tokens_num"] = (
            stats[split_name]["initial_size"] - data.num_rows
        )

        stats[split_name]["min_output_tokens"] = min(data["num_output_tokens"])
        stats[split_name]["max_output_tokens"] = max(data["num_output_tokens"])
        stats[split_name]["min_context_tokens"] = min(data["num_context_tokens"])
        stats[split_name]["max_context_tokens"] = max(data["num_context_tokens"])

        # Trying to parse the output as JSON
        df = data.select_columns(["context", "output"]).to_pandas()
        df["output"] = df["output"].apply(parse_json_markdown)
        df["output"] = df["output"].apply(dict_value_or_none)
        size_before_filtering = len(df)
        df = df[df["output"].notna()]

        if not df["output"].apply(lambda x: set(x.keys()) == set(schema.keys())).all():
            raise ValueError("Schema columns do not match dataset columns")

        df["output"] = df["output"].apply(json.dumps)

        stats[split_name]["filtered_by_all_missing_values"] = size_before_filtering - len(df)
        stats[split_name]["final_size"] = len(df)
        logger.info(f"[{split_name}] Final dataset size: {len(df)}")

        f_name = target_dir / f"{split_name}.json"
        logger.info(f"[{split_name}] Saving to {f_name}")
        df.to_json(f_name, orient="records", indent=4)

    stats_path = target_dir / "dataset_info.json"
    logger.info(f"[{split_name}] Saving stats to {stats_path}")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    # check dataset loads
    ds = load_dataset("json", data_dir=target_dir)
    logger.info(f"Dataset correctly loaded: {ds}")


def dict_value_or_none(data: dict[str, Any]) -> dict[str, Any] | None:
    for key, value in data.items():
        data[key] = value_or_none(value)

    if all(not value for value in data.values()):
        return None
    return data


def value_or_none(x: str | list[str]) -> str | list[str]:
    """Replaces missing value comments with empty strings."""
    if (x == ["data not available"]) or (x == "data not available"):
        return ""
    elif (x == "Don't know") or (x == ["Don't know"]):
        return ""
    else:
        return x


if __name__ == "__main__":
    typer.run(main)
