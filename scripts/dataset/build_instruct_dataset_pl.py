from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import typer
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger

from juddges.settings import PL_JUDGMENTS_PATH_INSTRUCT, PL_JUDGMENTS_PATH_RAW

load_dotenv()

MAX_SHARD_SIZE = "4GB"
DATE_FORMAT = "%Y-%m-%d"
JUDGE_SEPARATOR = " i "

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
    "court_name": "<nazwa sądu, string containing the full name of the court>",
    "department_name": "<nazwa wydziału, string containing the full name of the court's department>",
    "legal_bases": "<podstawy prawne, list of strings containing legal bases (legal regulations)>",
}
PROMPT = INSTRUCTION_TEMPLATE.format(
    schema=SCHEMA_TEMPLATE.format(schema=yaml.dump(SCHEMA_DESC).strip())
)

FEATURES = [
    "date",
    "judges",
    "recorder",
    "signature",
    "court_name",
    "department_name",
    "text_legal_bases",
    "text",
]
SCHEMA_2_FEATURES = {
    "date": "date",
    "judges": "judges",
    "recorder": "recorder",
    "signature": "signature",
    "court_name": "court_name",
    "department_name": "department_name",
    "legal_bases": "text_legal_bases",
}


def main(
    dataset_dir: Path = typer.Option(
        PL_JUDGMENTS_PATH_RAW, help="Path to the dataset directory"
    ),
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
    ds = load_dataset(
        "parquet", name="pl_judgements", data_dir=dataset_dir, columns=feature_cols
    )
    assert all(col in ds.column_names["train"] for col in feature_cols)

    initial_size = ds["train"].num_rows
    logger.info(f"Pre-filtering dataset (initial size={initial_size})...")

    ds = ds.filter(_pre_filter)

    pre_filtered_size = ds["train"].num_rows
    logger.info(
        f"Finished pre-filtering (size={pre_filtered_size}, removed {initial_size - pre_filtered_size})"
    )

    ds = ds.map(_preprocess)
    ds = ds.filter(_filter)

    filtered_size = ds["train"].num_rows
    logger.info(
        f"Finished filtering (size={filtered_size}, "
        f"removed {initial_size - filtered_size} from original, "
        f"and {pre_filtered_size - filtered_size} from pre-filtered)"
    )

    logger.info("Generating instructions...")
    ds = ds.map(to_instruction_fmt, num_proc=num_jobs, remove_columns=FEATURES)
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
    return not any(item[feat] is None for feat in FEATURES)


def _preprocess(item: dict[str, Any]) -> dict[str, Any]:
    item = _simplify_date(item)
    item = _split_multiple_names(item)
    item = _legal_bases_to_texts(item)
    return item


def _simplify_date(item: dict[str, Any]) -> dict[str, Any]:
    item["date"], *_ = item["date"].split()
    datetime.strptime(item["date"], DATE_FORMAT)  # raises ValueError on invalid format
    return item


def _split_multiple_names(item: dict[str, Any]) -> dict[str, Any]:
    """Splits judges names that are joined by 'i'."""
    judges = []
    for j in item["judges"]:
        for j_part in j.split(JUDGE_SEPARATOR):
            judges.append(j_part)

    item["judges"] = judges
    return item


def _legal_bases_to_texts(item: dict[str, Any]) -> dict[str, Any]:
    item["text_legal_bases"] = [x["text"] for x in item["text_legal_bases"]]
    return item


def _filter(item: dict[str, Any]) -> bool:
    all_judges_in_text = not any(j not in item["text"] for j in item["judges"])
    recorder_in_text = item["recorder"] in item["text"]
    signature_in_text = item["signature"] in item["text"]

    return all_judges_in_text and recorder_in_text and signature_in_text


def to_instruction_fmt(item: dict[str, Any]) -> dict[str, str]:
    yaml_output = yaml.dump(
        {k: item[SCHEMA_2_FEATURES[k]] for k in SCHEMA_DESC.keys()},
        allow_unicode=True,
    ).strip()
    output = SCHEMA_TEMPLATE.format(schema=yaml_output)

    return {"prompt": PROMPT, "context": item["text"], "output": output}


if __name__ == "__main__":
    typer.run(main)
