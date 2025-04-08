import datetime
import re
from pathlib import Path
from typing import Any, Optional

import typer
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger

from juddges.settings import PL_JUDGMENTS_PATH_INSTRUCT

load_dotenv()

MAX_SHARD_SIZE = "4GB"
SOURCE_DATASET_PATH = "JuDDGES/en-court-raw"
TEXT_FIELD = "content"
TEXT_MIN_LEN = 1_000

SCHEMA_TEMPLATE = "```yaml\n{schema}\n```"
INSTRUCTION_TEMPLATE = """
You are extracting information from the court judgments.
Extract specified values strictly from the provided judgement. If information is not provided in the judgement, leave the field with null value.
Please return the response in the identical YAML format:
{schema}
=====
{{context}}
======
"""
""
SCHEMA_DESC = {
    "judges": "<list of judge full names>",
    "citation": "<string containing the neutral citation number>",
    "date": "<date in format YYYY-MM-DD>",
}

PROMPT = INSTRUCTION_TEMPLATE.format(
    schema=SCHEMA_TEMPLATE.format(schema=yaml.dump(SCHEMA_DESC).strip())
)

FEATURES = [
    "judges",
    "citation",
    TEXT_FIELD,
]
SCHEMA_2_FEATURES = dict(zip(FEATURES, FEATURES)) | {"date": "date"}


date_pattern = r"\b(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?(?:,?\s*)?(\d{1,2})(?:\s*(?:st|nd|rd|th))?(?:,?\s*)?(January|February|March|April|May|June|July|August|September|October|November|December)\s*(\d{4})|(\d{2})/(\d{2})/(\d{4}))\b"
date_pattern = re.compile(date_pattern)


# todo: In the future one might make this single script (for any language) with configurable preprocessing
def main(
    dataset_dir: str = typer.Option(SOURCE_DATASET_PATH, help="Path to the dataset directory"),
    repo_id: Optional[str] = typer.Option(None),
    target_dir: Path = typer.Option(
        PL_JUDGMENTS_PATH_INSTRUCT,
        help="Path to the target directory",
    ),
    test_size: int = typer.Option(2_000, help="Size of the test set"),
    random_seed: int = typer.Option(42, help="Random seed"),
    num_jobs: Optional[int] = typer.Option(
        None,
        envvar="NUM_JOBS",
        help="Number of parallel jobs to use",
    ),
    branch: Optional[str] = typer.Option(None, help="Branch to push the dataset to"),
    commit_message: Optional[str] = typer.Option(
        None, help="Commit message", envvar="COMMIT_MESSAGE"
    ),
) -> None:
    feature_cols = ["_id"] + FEATURES
    logger.info("Loading dataset...")
    ds = load_dataset(dataset_dir, columns=feature_cols)
    assert all(col in ds.column_names["train"] for col in feature_cols)

    initial_size = ds["train"].num_rows
    logger.info(f"Pre-filtering dataset (initial size={initial_size})...")

    ds = ds.filter(_pre_filter)
    ds = ds.map(_extract_date, batched=False)

    pre_filtered_size = ds["train"].num_rows
    logger.info(
        f"Finished pre-filtering (size={pre_filtered_size}, removed {initial_size - pre_filtered_size})"
    )

    ds = ds.filter(_filter)

    filtered_size = ds["train"].num_rows
    logger.info(
        f"Finished filtering (size={filtered_size}, "
        f"removed {initial_size - filtered_size} from original, "
        f"and {pre_filtered_size - filtered_size} from pre-filtered)"
    )

    logger.info("Generating instructions...")
    ds = ds.map(to_instruction_fmt, num_proc=num_jobs, remove_columns=ds["train"].column_names)
    ds = ds["train"].train_test_split(test_size=test_size, seed=random_seed)

    logger.info("Built dataset with following parameters: {ds_info}", ds_info=str(ds))

    if repo_id:
        ds.push_to_hub(
            repo_id,
            max_shard_size=MAX_SHARD_SIZE,
            commit_message=commit_message,
            revision=branch,
        )
    else:
        ds.save_to_disk(target_dir, max_shard_size=MAX_SHARD_SIZE, num_proc=num_jobs)


def _pre_filter(item: dict[str, Any]) -> bool:
    """Discards items with missing features or too short text"""
    return (not any(item[feat] is None for feat in FEATURES)) and (
        len(item[TEXT_FIELD]) > TEXT_MIN_LEN
    )


def _extract_date(item: dict[str, Any]) -> dict[str, str]:
    """Extracts the date from the text according to the identified pattern."""
    match = date_pattern.search(item["content"])
    assert match
    if match.group(4) and match.group(5) and match.group(6):
        day = match.group(4)
        month = match.group(5)
        year = match.group(6)
    else:
        day = match.group(1)
        month_name = match.group(2)
        year = match.group(3)
        month = datetime.datetime.strptime(month_name, "%B").month

    return {"date": datetime.date(int(year), int(month), int(day)).strftime("%Y-%m-%d")}


def _filter(item: dict[str, Any]) -> bool:
    all_judges_in_text = not any(j not in item[TEXT_FIELD] for j in item["judges"])
    return all_judges_in_text


def to_instruction_fmt(item: dict[str, Any]) -> dict[str, str]:
    yaml_output = yaml.dump(
        {k: item[SCHEMA_2_FEATURES[k]] for k in SCHEMA_DESC.keys()},
        allow_unicode=True,
    ).strip()
    output = SCHEMA_TEMPLATE.format(schema=yaml_output)

    return {"prompt": PROMPT, "context": item[TEXT_FIELD], "output": output}


if __name__ == "__main__":
    typer.run(main)
