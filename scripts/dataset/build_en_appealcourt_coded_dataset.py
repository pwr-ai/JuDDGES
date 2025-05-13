import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import typer
import yaml
from datasets import Dataset, load_dataset
from langchain_core.utils.json import parse_json_markdown
from loguru import logger
from transformers import AutoTokenizer

from juddges.utils.validate_schema import validate_output_structure

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

        df["output"] = df["output"].apply(map_remand_custody_time)

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

    logger.info("Checking schema mismatch for train split")
    check_schema_mismatch(schema, ds["train"])
    logger.info("Train split correct")

    logger.info("Checking schema mismatch for test split")
    check_schema_mismatch(schema, ds["test"])
    logger.info("Test split correct")


def dict_value_or_none(data: dict[str, Any]) -> dict[str, Any] | None:
    for key, value in data.items():
        data[key] = value_or_none(value)

    if all(not value for value in data.values()):
        return None

    return data


def value_or_none(item: list[str] | str) -> str | list[str] | None:
    if isinstance(item, list):
        # removes nested lists
        fixed_item = []
        for x in item:
            if isinstance(x, list):
                fixed_item.extend(x)
            else:
                fixed_item.append(x)

        filtered_item = [
            x
            for x in fixed_item
            if x.strip().lower() not in ["don't know", "data not available", None]
        ]
        if len(filtered_item) == 0:
            return None
        else:
            return filtered_item
    elif isinstance(item, str):
        item_proc = item.lower().strip()
        if (item_proc == "don't know") or item_proc == ("data not available"):
            return None
        else:
            return item
    else:
        raise ValueError(f"Invalid value type: {type(item)}")


def map_remand_custody_time(item: dict[str, Any]) -> list[int] | None:
    if item["RemandCustodyTime"] is None:
        return item

    _CUSTODY_TIME_MAP = {
        "Don't know": None,
        "data not available": None,
        "94 days": 94,
        "265 days spent on remand": 265,
        "104 days spent on remand.": 104,
        "125 days was ordered to count as time served": 125,
        "less 25 days.": 25,
        "less 173 days.": 173,
        "404 days": 404,
        "208 days spent on remand": 208,
        "59 days": 59,
        "Akehurst": None,
        "(31 days)": 31,
        "155 days": 155,
        "28 days": 28,
        "595 days spent on remand": 595,
        "105 days": 105,
        "194 days spent on remand": 194,
        "Barnes' case 286 days": 286,
        "Burton's case, that 423 days": 423,
        "total time during which the appellant was held solely on remand was therefore 112 days.": 112,
        "253 days spent on remand.": 253,
        "275": 275,
        "91 days": 91,
        "71 days spent on remand": 71,
        "51 days which the appellant spent in prison": 51,
        "208 days on remand": 208,
        "219 days": 219,
        "five months and 25 days": 25,
        "186 days": 186,
        "two days": 2,
        "287 days": 287,
        "240 days": 240,
        "70 days": 70,
        "169 days": 169,
        "All appellants": None,
        "248 days": 248,
        "28 days on remand": 28,
        "546 days": 546,
        "176 days": 176,
        "24 days": 24,
        "70 days spent on remand": 70,
        "14 months": 420,
        "418 days": 418,
        "Creighton": None,
        "458 days": 458,
        "460 days": 460,
        "262 days": 262,
        "203 days": 203,
        "323 days": 323,
        "195 days": 195,
        "185 days spent in custody on remand": 185,
        "106 days": 106,
        "3 months": 90,
        "200 days": 200,
        "custody for one year and 270 days": 270,
        "427 days he had spent": 427,
        "been in custody for three weeks": 21,
        "465 days": 465,
        "28": 28,
        "four days": 4,
        "109 days": 109,
        "238 days": 238,
        "163 days": 163,
        "nearly three months": 90,
        "99 days": 99,
        "161 days spent on remand": 161,
        "360 days spent on remand should count": 360,
        "5 months": 150,
        "209 days spent on curfew": 209,
        "98 days.": 98,
        "spent some seven months in custody": 210,
        "spent a week in custody": 7,
        "28 of the days": 28,
        "38 days that he had spent on remand": 38,
        "160 days on remand": 160,
        "361 days": 361,
        "572 days": 572,
        "curfew for a period of 96 days": 96,
        "ten days spent on remand on custody": 10,
        "108 days spent on curfew": 108,
    }
    item["RemandCustodyTime"] = [_CUSTODY_TIME_MAP[elem] for elem in item["RemandCustodyTime"]]
    item["RemandCustodyTime"] = [x for x in item["RemandCustodyTime"] if x is not None]
    if len(item["RemandCustodyTime"]) == 0:
        item["RemandCustodyTime"] = None

    return item


def check_schema_mismatch(
    schema: dict[str, dict[str, Any]],
    dataset: Dataset,
) -> list[dict[str, str]]:
    assert not any(
        validate_output_structure(item, schema, format="json")["num_errors"]
        for item in dataset["output"]
    )


if __name__ == "__main__":
    typer.run(main)
