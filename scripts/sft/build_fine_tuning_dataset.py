from pathlib import Path
from typing import Any
from typing import Optional
from dotenv import load_dotenv
from loguru import logger
import typer
from datasets import load_dataset
from datetime import datetime

import yaml

from juddges.settings import PL_JUDGEMENTS_PATH_INSTRUCT, PL_JUDGEMENTS_PATH_RAW

load_dotenv()

MAX_SHARD_SIZE = "4GB"

SCHEMA_TEMPLATE = "```yaml\n{schema}\n```"
INSTRUCTION_TEMPLATE = """
You are extracting information from the Polish court judgments.
Extract specified values strictly from the provided judgement. If information is not provided in the judgement, leave the field with null value.
Please return the response in the identical YAML format:
{schema}
=====
{{context}}
======
"""
SCHEMA_DESC = {
    "date": "<data, date in format YYYY-MM-DD>",
    "judges": "<sędziowie, list of judge full names>",
    "recorder": "<protokolant, string containing the name of the recorder>",
    "signature": "<sygnatura, string contraining the signature of the judgment>",
}
PROMPT = INSTRUCTION_TEMPLATE.format(
    schema=SCHEMA_TEMPLATE.format(schema=yaml.dump(SCHEMA_DESC).strip())
)

FEATURES = ["date", "judges", "recorder", "signature", "text"]


def main(
    dataset_dir: Path = typer.Option(PL_JUDGEMENTS_PATH_RAW, help="Path to the dataset directory"),
    repo_id: Optional[str] = typer.Option(None),
    target_dir: Path = typer.Option(
        PL_JUDGEMENTS_PATH_INSTRUCT,
        help="Path to the target directory",
    ),
    test_size: int = typer.Option(2_000, help="Size of the test set"),
    random_seed: int = typer.Option(42, help="Random seed"),
    num_jobs: Optional[int] = typer.Option(
        None,
        envvar="NUM_JOBS",
        help="Number of parallel jobs to use",
    ),
) -> None:
    feature_cols = ["_id"] + FEATURES
    ds = load_dataset("parquet", name="pl_judgements", data_dir=dataset_dir)
    logger.info("Loading dataset...")
    ds = ds.select_columns(feature_cols)

    logger.info("Cleaning dataset...")
    ds = ds.filter(_pre_filter)
    ds = ds.map(_preprocess)
    ds = ds.filter(_filter)

    logger.info("Generating instructions...")
    ds = ds.map(to_instruction_fmt, num_proc=num_jobs, remove_columns=FEATURES)
    ds = ds["train"].train_test_split(test_size=test_size, seed=random_seed)

    logger.info("Built dataset with following parameters: {ds_info}", ds_info=str(ds))

    if repo_id:
        ds.push_to_hub(repo_id, max_shard_size=MAX_SHARD_SIZE)
    else:
        ds.save_to_disk(target_dir, max_shard_size=MAX_SHARD_SIZE, num_proc=num_jobs)


def _pre_filter(item: dict[str, Any]) -> bool:
    return all(item[feat] is not None for feat in FEATURES)


def _preprocess(item: dict[str, Any]) -> dict[str, Any]:
    item = _simplify_date(item)
    item = _split_multiple_names(item)
    return item


def _simplify_date(item: dict[str, Any]) -> dict[str, Any]:
    item["date"], *_ = item["date"].split()
    datetime.strptime(item["date"], "%Y-%m-%d")  # raises ValueError on invalid format
    return item


def _split_multiple_names(item: dict[str, Any]) -> dict[str, Any]:
    """Splits judges names that are joined by 'i'."""
    judges = []
    for j in item["judges"]:
        for j_part in j.split(" i "):
            judges.append(j_part)

    item["judges"] = judges
    return item


def _filter(item: dict[str, Any]) -> bool:
    all_judges_in_text = all(j in item["text"] for j in item["judges"])
    recorder_in_text = item["recorder"] in item["text"]
    signature_in_text = item["signature"] in item["text"]

    return all_judges_in_text and recorder_in_text and signature_in_text


def to_instruction_fmt(item: dict[str, Any]) -> dict[str, str]:
    output = SCHEMA_TEMPLATE.format(
        schema=yaml.dump({k: item[k] for k in SCHEMA_DESC.keys()}).strip()
    )

    return {"prompt": PROMPT, "context": item["text"], "output": output}


if __name__ == "__main__":
    typer.run(main)
