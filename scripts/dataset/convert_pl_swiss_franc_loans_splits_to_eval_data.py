import json
from pathlib import Path

import typer

from juddges.data.dataset_factory import get_dataset

PL_SWISS_FRANC_LOANS_DATASET = "data/datasets/pl/swiss_franc_loans"


def main(
    dataset_name_or_path: str = typer.Option(
        PL_SWISS_FRANC_LOANS_DATASET,
        help="Path to the dataset",
    ),
    output_dir: Path = typer.Option(..., help="Path to the output directory"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = get_dataset(dataset_name_or_path=dataset_name_or_path, split=None)
    results = []
    for gpt_item, annotated_item in zip(ds["test"], ds["annotated"], strict=True):
        results.append(
            {
                "answer": gpt_item,
                "gold": annotated_item,
            }
        )

    with open(output_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    typer.run(main)
