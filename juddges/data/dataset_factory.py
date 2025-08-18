from pathlib import Path

from datasets import Dataset, load_dataset


def get_dataset(dataset_name_or_path: str | Path, split: str | None) -> Dataset:
    dataset_name_or_path = str(dataset_name_or_path)
    if dataset_name_or_path == "data/datasets/pl/swiss_franc_loans":
        return load_dataset(
            dataset_name_or_path,
            data_files={"train": "train.json", "test": "test.json", "annotated": "annotated.json"},
            split=split,
        )
    elif dataset_name_or_path == "data/datasets/pl/personal_rights":
        return load_dataset(
            dataset_name_or_path,
            data_files={"train": "train.json"},
            split=split,
        )
    elif dataset_name_or_path == "data/datasets/en/en_appealcourt_coded":
        return load_dataset(
            dataset_name_or_path,
            data_files={"test": "test.json", "annotated": "annotated.json"},
            split=split,
        )
    else:
        raise ValueError(f"Dataset {dataset_name_or_path} not supported.")
