from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from huggingface_hub import DatasetCard, DatasetCardData, HfApi

load_dotenv()

DATASET_CARD_TEMPLATE_DIR = Path("data/datasets/pl/readme/instruct")

PL_CARD_DATA = DatasetCardData(
    language="pl",
    multilinguality="monolingual",
    size_categories="100K<n<1M",
    source_datasets=["JuDDGES/pl-court-raw"],
    pretty_name="Polish Court Judgments Instruct",
    tags=["polish court"],
    task_categories=["text-generation", "text2text-generation"],
)

EN_CARD_DATA = DatasetCardData(
    language="en",
    multilinguality="monolingual",
    size_categories="1K<n<10K",
    source_datasets=["JuDDGES/en-court-raw"],
    pretty_name="English Court Judgments Instruct",
    tags=["english court"],
    task_categories=["text-generation", "text2text-generation"],
)


def main(
    dataset_card_template_dir: Path = typer.Option(
        ...,
        help="Path to the directory with the dataset card. It should contain a README.md file and (optionally) a  README_files directory.",
    ),
    repo_id: str = typer.Option(...),
    language: str = typer.Option(...),
    branch: Optional[str] = typer.Option(None, help="Branch to push the dataset to"),
) -> None:
    assert any(file.name == "README.md" for file in dataset_card_template_dir.iterdir())
    assert any(file.name == "README_files" for file in dataset_card_template_dir.iterdir())

    card_data = PL_CARD_DATA if language == "pl" else EN_CARD_DATA
    card = DatasetCard.from_template(
        card_data,
        template_path=dataset_card_template_dir / "README.md",
    )
    card.push_to_hub(repo_id, revision=branch)

    api = HfApi()
    api.upload_folder(
        folder_path=dataset_card_template_dir / "README_files",
        path_in_repo="README_files",
        repo_id=repo_id,
        repo_type="dataset",
        revision=branch,
    )


if __name__ == "__main__":
    typer.run(main)
