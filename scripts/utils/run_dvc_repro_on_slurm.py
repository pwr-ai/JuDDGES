import random
import re
import subprocess
import sys
import time
from pprint import pformat

import tabulate
import typer
from loguru import logger
from tqdm import tqdm

SLURM_SCRIPT = "./slurm/run_command.sh"
MIN_WAIT_TIME = 10
MAX_WAIT_TIME = 30


def main(
    stages: list[str] = typer.Argument(..., help="Stages to run"),
    num_gpus: int = typer.Option(..., "--num-gpus", help="Number of GPUs to use"),
    account: str = typer.Option(..., "--account", help="HPC account to use"),
    partition: str = typer.Option(..., "--partition", help="Partition to use"),
    glob: bool = typer.Option(False, "--glob", help="Use glob pattern to match stages"),
    dry: bool = typer.Option(False, "--dry", help="Dry run"),
):
    stages_to_run = get_stages_to_run(stages, glob)

    logger.info(f"Found {len(stages_to_run)} stages to run:\n{pformat(stages_to_run)}")
    if not typer.confirm("Do you want to run these stages?"):
        logger.info("Exiting...")
        typer.Abort()

    submission_info = []
    for stage in tqdm(stages_to_run, desc="Submitting"):
        if not dry:
            jitter = random.uniform(MIN_WAIT_TIME, MAX_WAIT_TIME)
            time.sleep(jitter)
        job_id, info = submit_stage_to_slurm(
            stage=stage,
            num_gpus=num_gpus,
            account=account,
            partition=partition,
            dry=dry,
        )
        submission_info.append([stage, job_id, info])

    print(
        tabulate.tabulate(
            submission_info,
            headers=["Stage", "Job ID", "Info"],
            tablefmt="grid",
        )
    )


def get_stages_to_run(stages: list[str], glob: bool) -> list[str]:
    """Run dvc repro --dry and extract stages that would run."""
    cmd = ["dvc", "repro", "-s", "--dry", "--ignore-errors"] + stages

    if glob:
        cmd.append("--glob")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        result.check_returncode()

        stages_to_run = []
        pattern = r"Running stage '([^']+)':"

        for line in result.stdout.splitlines():
            match = re.search(pattern, line)
            if match:
                stage_name = match.group(1)
                stages_to_run.append(stage_name)

        return stages_to_run

    except subprocess.CalledProcessError:
        logger.error(f"Error running fetching stages to run: {result.stderr}")
        sys.exit(1)


def submit_stage_to_slurm(
    stage: str,
    num_gpus: int,
    account: str,
    partition: str,
    dry: bool,
) -> tuple[str | None, str]:
    command = [
        "sbatch",
        "--account",
        account,
        "--job-name",
        f"juddges-{stage}",
        "--partition",
        partition,
        "--gpus-per-node",
        f"hopper:{num_gpus}",
        "--nodes",
        "1",
        SLURM_SCRIPT,
        "--run-cmd",
        f"dvc repro -s {stage}",
    ]

    logger.info(f"Submitting job for stage {stage} with command: {' '.join(command)}")

    if not dry:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Failed to submit job for stage {stage}: {result.stderr}")
            return None, result.stderr

        job_id = result.stdout.strip().split()[-1]
        logger.info(f"Submitted job {job_id} for stage {stage}")
        return job_id, ""
    else:
        return None, "Dry run"


if __name__ == "__main__":
    typer.run(main)
