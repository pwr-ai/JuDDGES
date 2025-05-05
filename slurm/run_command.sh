#!/bin/bash

#SBATCH --job-name=juddges_sft
#SBATCH --output=logs/%j-%x.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
# NOTE: You can reconfigure the above parameters to your needs in the sbatch call.
# NOTE: All env variables must be exported to be available after calling srun.
# NOTE: You may need to specify some NCCL args in .env file depending on your cluster configuration

echo "[$(date)] Running job $SLURM_JOB_ID on host: $(hostname)"

# =====Provide these user-specific env variables through .env file=====
if [ ! -f ./slurm/.env ]; then
    echo "Error: ./slurm/.env file not found"
    exit 1
fi
set -a # Enable automatic export of all variables
source ./slurm/.env
set +a # Disable automatic export after loading

# Validate required environment variables
required_vars=("WANDB_API_KEY" "HF_TOKEN" "SIF_IMAGE_PATH" "WORKDIR")
for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "Error: Required environment variable $var is not set"
        exit 1
    fi
done

# Validate SIF image exists
if [ ! -f "$SIF_IMAGE_PATH" ]; then
    echo "Error: SIF image not found at $SIF_IMAGE_PATH"
    exit 1
fi

# Validate WORKDIR exists
if [ ! -d "$WORKDIR" ]; then
    echo "Error: WORKDIR $WORKDIR does not exist"
    exit 1
fi

export WANDB_API_KEY
export HF_TOKEN
export SIF_IMAGE_PATH
export WORKDIR

export NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' '\n'))
export WORLD_SIZE=$(($SLURM_GPUS_ON_NODE * $SLURM_JOB_NUM_NODES))

# =====Parse command line arguments=====
run_cmd=""
while [ $# -gt 0 ]; do
    case "$1" in
        --run-cmd)
            run_cmd="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# check if parameter is provided
if [ -z "$run_cmd" ]; then
    echo "run_cmd parameter is required" >&2
    echo "Usage: $0 --run-cmd <command>" >&2
    echo "Example: $0 --run-cmd 'python my_script.py'" >&2
    exit 1
fi

# =====Run the script using apptainer image=====
export NUM_PROC=$SLURM_CPUS_PER_GPU
export PYTHONPATH="$PYTHONPATH."

cd $WORKDIR || {
    echo "Error: Failed to change to directory $WORKDIR"
    exit 1
}

echo "Running the following command with apptainer: $run_cmd"

srun --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    apptainer run \
        --fakeroot \
        --bind "$TMPDIR:$TMPDIR" \
        --bind "$WORKDIR:$WORKDIR" \
        --bind "$HOME:$HOME" \
        --nv \
        "$SIF_IMAGE_PATH" \
        bash -c "$run_cmd"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Command failed with exit code $EXIT_CODE"
fi
exit $EXIT_CODE
