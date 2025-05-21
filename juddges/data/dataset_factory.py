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
    elif dataset_name_or_path == "data/datasets/en/en_appealcourt_coded":
        return load_dataset(
            dataset_name_or_path,
            data_files={"test": "test.json", "annotated": "annotated.json"},
            split=split,
        )
    else:
        raise ValueError(f"Dataset {dataset_name_or_path} not supported.")


if __name__ == "__main__":
    from huggingface_hub import DatasetCardData

    ds = get_dataset("data/datasets/pl/swiss_franc_loans", split=None)

    card_data = DatasetCardData(
        configs=[
            {
                "config_name": "default",
                "data_files": [
                    {
                        "split": split,
                        "path": f"{split}.json",
                    }
                    for split in ds.keys()
                ],
            }
        ],
        dataset_info=ds["train"].info._to_yaml_dict(),
    )

    print(card_data)


    ds = get_dataset("data/datasets/en/en_appealcourt_coded", split=None)

    card_data = DatasetCardData(
        configs=[
            {
                "config_name": "default",
                "data_files": [
                    {
                        "split": split,
                        "path": f"{split}.json",
                    }
                    for split in ds.keys()
                ],
            }
        ],
        dataset_info=ds["test"].info._to_yaml_dict(),
    )

    print(card_data)