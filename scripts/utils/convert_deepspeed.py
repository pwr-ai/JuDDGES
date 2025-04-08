import os
import shutil
import subprocess
from pathlib import Path
from pprint import pformat

import typer
from loguru import logger
from tqdm import tqdm

SCRIPT_NAME = "zero_to_fp32.py"
CONVERTED_MODEL_PATTERN = "model*.safetensors"


def main(
    root_dir: Path = typer.Option(),
    adapter_only: bool = typer.Option(False, help="Only convert adapter"),
    remove: bool = typer.Option(False, help="Removes original deepspeed checkpoints"),
    remove_for_converted: bool = typer.Option(
        False, help="Removes only original deepspeed checkpoints for already converted models"
    ),
) -> None:
    checkpoint_dirs = [script_file.parent for script_file in root_dir.rglob(SCRIPT_NAME)]
    logger.info(f"Found {len(checkpoint_dirs)} checkpoints to convert:\n{pformat(checkpoint_dirs)}")
    for ckpt_dir in tqdm(checkpoint_dirs, desc="Converting checkpoints"):
        logger.info(f"Converting {ckpt_dir}")
        if list(ckpt_dir.glob(CONVERTED_MODEL_PATTERN)):
            logger.warning(f"Model already converted, skipping {ckpt_dir}")
            if remove_for_converted:
                logger.info(f"Removing deepspeed artifacts for {ckpt_dir}")
                remove_deepspeed_artifacts(ckpt_dir)
            continue
        else:
            convert(ckpt_dir)

        # deepspeed saves model as model.safetensors, need to rename it to adapter_model.safetensors
        if adapter_only:
            # there should be (almost) empty adapter_model.safetensors
            assert (ckpt_dir / "adapter_model.safetensors").exists()
            for model_file in ckpt_dir.glob("model*.safetensors"):
                model_file.rename(
                    model_file.with_stem(model_file.stem.replace("model", "adapter_model"))
                )

        if remove:
            remove_deepspeed_artifacts(ckpt_dir)


def convert(ckpt_dir: Path) -> None:
    script_file = ckpt_dir / SCRIPT_NAME
    step_dir = get_latest_step_dir(ckpt_dir)
    logger.info(f"Converting {step_dir}")
    cmd = [
        "python",
        str(script_file),
        str(ckpt_dir),  # checkpoint_dir
        str(ckpt_dir),  # output_dir
        "--safe_serialization",  # writes as safetensors file
        "--max_shard_size",
        "5GB",
        "--tag",
        step_dir.name,  # points to directory globalstep<step_num>
    ]
    env = os.environ.copy() | {"CUDA_VISIBLE_DEVICES": "-1"}
    subprocess.run(cmd, check=True, env=env)


def remove_deepspeed_artifacts(ckpt_dir: Path) -> None:
    step_dir = get_latest_step_dir(ckpt_dir)
    logger.info(f"Removing {step_dir}")
    shutil.rmtree(step_dir)

    for rng_file in ckpt_dir.glob("rng_state_*.pth"):
        os.remove(rng_file)

    os.remove(ckpt_dir / SCRIPT_NAME)
    os.remove(ckpt_dir / "latest")
    os.remove(ckpt_dir / "scheduler.pt")


def get_latest_step_dir(ckpt_dir: Path) -> Path:
    with open(ckpt_dir / "latest") as f:
        step_dirname = f.read().strip()
    return ckpt_dir / step_dirname


if __name__ == "__main__":
    typer.run(main)
