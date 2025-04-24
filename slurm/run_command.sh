#!/bin/bash

#SBATCH --job-name=hallu
#SBATCH --partition=lem-gpu
#SBATCH --output=logs/%j-%x.log
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=hopper:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
# NOTE: You can reconfigure the above parameters to your needs in the sbatch call.
# NOTE: All env variables must be exported to be available after calling srun.
# NOTE: You may need to specify some NCCL args in .env file depending on your cluster configuration

echo "[$(date)] Running job on host: $(hostname)"

# =====Provide these user-specific env variables through .env file=====
if [ ! -f ./slurm/.env ]; then
    echo "Error: ./slurm/.env file not found"
    exit 1
fi
set -a # Enable automatic export of all variables
source ./slurm/.env
set +a # Disable automatic export after loading

export WANDB_API_KEY
export HF_TOKEN
export SIF_IMAGE_PATH
export CONDA_ENV_NAME

# use local tmpdir to prevent home folder filling up
export HF_HOME="$TMPDIR/.huggingface"
mkdir -p "$HF_HOME"

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
            shift
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
export NUM_PROC="$SLURM_CPUS_PER_GPU"
export PYTHONPATH="$PYTHONPATH:$PWD"

echo "Running the following command with apptainer: $run_cmd"

srun --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    conda run -n "$CONDA_ENV_NAME" "$run_cmd"

EXIT_CODE=$?
exit $EXIT_CODE
