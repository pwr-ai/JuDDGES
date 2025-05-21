from pathlib import Path
from pprint import pformat

import typer
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file


def main(root_dir: Path = typer.Option(...)):
    files = list(root_dir.rglob("adapter_model.safetensors"))
    logger.info(f"Renaming files:\n{pformat(files)}")
    for file in files:
        rename_keys(file)


def rename_keys(file: Path) -> None:
    legacy_file = file.rename(file.with_stem("legacy_adapter_model.safetensors"))

    with safe_open(legacy_file, framework="pt", device="cpu") as f:
        tensors = {}
        for key in f.keys():
            *head, adapter_name, tail = key.split(".")
            assert adapter_name == "default"
            new_key = ".".join(head + [tail])
            tensors[new_key] = f.get_tensor(key)

    save_file(tensors, file)
    logger.info(f"Saved fixed file at {file}")


if __name__ == "__main__":
    typer.run(main)
