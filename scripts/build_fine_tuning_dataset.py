from pathlib import Path
from typing import Any
from typing import Optional
from dotenv import load_dotenv
from loguru import logger
import typer
from datasets import load_dataset, Version, DatasetDict
from datetime import datetime

from juddges.settings import PL_JUDGEMENTS_PATH_INSTRUCT, PL_JUDGEMENTS_PATH_RAW
from juddges.utils import VersionBump, bump_version

load_dotenv()

INSTRUCTION = "Complete the following YAML with information extracted from the court judgment {{context}}: {schema}"
SCHEMA_TEMPLATE = """
```yaml
date: {date}
judges: {judges}
signature: {signature}
```
"""


def main(
    dataset_dir: Path = typer.Option(PL_JUDGEMENTS_PATH_RAW, help="Path to the dataset directory"),
    repo_id: Optional[str] = typer.Option(None),
    target_dir: Path = typer.Option(
        PL_JUDGEMENTS_PATH_INSTRUCT, help="Path to the target directory"
    ),
    test_size: int = typer.Option(2_000, help="Size of the test set"),
    bump: VersionBump = typer.Option(..., envvar="BUMP", help="Version bump"),
    random_seed: int = typer.Option(42, help="Random seed"),
    num_jobs: Optional[int] = typer.Option(
        None, envvar="NUM_JOBS", help="Number of parallel jobs to use"
    ),
):
    feature_cols = [
        "_id",
        "signature",
        "date",
        "chairman",
        "judges",
        "text",
    ]
    ds = load_dataset("parquet", name="pl_judgements", data_dir=dataset_dir)
    logger.info("Loading dataset...")
    ds = prepare_metadata(ds, bump)
    ds = ds.select_columns(feature_cols)

    logger.info("Cleaning dataset...")
    ds = ds.filter(_pre_filter)
    ds = ds.map(_preprocess)
    ds = ds.filter(_filter)

    logger.info("Generating instructions...")
    ds = ds.map(to_instruction_fmt, num_proc=num_jobs, remove_columns=feature_cols)
    ds = ds["train"].train_test_split(test_size=test_size, seed=random_seed)

    if repo_id:
        ds.push_to_hub(repo_id, max_shard_size="4GB")
    else:
        ds.save_to_disk(target_dir, max_shard_size="4GB", num_proc=num_jobs)


def prepare_metadata(ds: DatasetDict, bump: VersionBump) -> DatasetDict:
    current_version = ds["train"].info.version.version_str
    ds["train"].info.version = Version(bump_version(current_version, bump))
    ds["train"].info.download_checksums = None
    return ds


def _pre_filter(item: dict[str, Any]) -> bool:
    return item["date"] is not None and item["judges"] is not None and item["text"] is not None


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
    signature_in_text = item["signature"] in item["text"]

    return all_judges_in_text and signature_in_text


def to_instruction_fmt(item: dict[str, Any]) -> str:
    schema = SCHEMA_TEMPLATE.format(
        date="date in format YYYY-MM-DD",
        judges="list of judge names [judge_name1, judge_name2, ...]",
        signature="string contraining the signature of the judgment",
    )

    output = SCHEMA_TEMPLATE.format(
        date=item["date"],
        judges=item["judges"],
        signature=item["signature"],
    )

    return {
        "prompt": INSTRUCTION.format(schema=schema),
        "context": item["text"],
        "output": output,
    }


if __name__ == "__main__":
    typer.run(main)
