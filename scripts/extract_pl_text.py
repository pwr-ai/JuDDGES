from pathlib import Path
from typing import Optional

import typer
from datasets import load_dataset

from juddges.preprocessing.pl_court_parser import SimplePlJudgementsParser


def main(
    dataset_dir: Path = typer.Option(..., help="Path to the dataset directory"),
    target_dir: Path = typer.Option(..., help="Path to the target directory"),
    num_proc: Optional[int] = typer.Option(None, help="Number of processes to use"),
) -> None:
    target_dir.parent.mkdir(exist_ok=True, parents=True)
    ds = load_dataset("parquet", name="pl_judgements", data_dir=dataset_dir)
    num_shards = len(ds["train"].info.splits["train"].shard_lengths)
    text_extractor = SimplePlJudgementsParser()
    ds = (
        ds["train"]
        .select_columns(
            ["_id", "date", "type", "excerpt", "content"]
        )  # leave only most important columns
        .filter(lambda x: x["content"] is not None)
        .map(text_extractor, input_columns="content", num_proc=num_proc)
    )
    ds.save_to_disk(target_dir, num_shards=num_shards)


if __name__ == "__main__":
    typer.run(main)
