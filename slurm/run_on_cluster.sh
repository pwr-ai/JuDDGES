#!/bin/bash

#SBATCH --job-name=juddges_sft
#SBATCH --output=logs/%j-%x.log
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
# NOTE: You can reconfigure the above parameters to your needs in the sbatch call.
# NOTE: All env variables must be exported to be available after calling srun.
# NOTE: You may need to specify some NCCL args in .env file depending on your cluster configuration

# =====Provide these user-specific env variables through .env file=====

if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found" >&2
    exit 1
fi

export WANDB_API_KEY
export HF_TOKEN
export SIF_IMAGE_PATH

export NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' '\n'))
export WORLD_SIZE=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES))

# =====Parse command line arguments=====
while [ $# -gt 0 ]; do
    case "$1" in
        -m|--model)
            model="$2"
            shift 2
            ;;
        -d|--dataset)
            dataset="$2"
            shift 2
            ;;
        *)
            echo "Invalid option: $1" >&2
            echo "Usage: $0 --model <model> --dataset <dataset>" >&2
            echo "   or: $0 -m <model> -d <dataset>" >&2
            echo "Example: $0 --model Unsloth-Llama-3-8B-Instruct --dataset pl-court-instruct" >&2
            exit 1
            ;;
    esac
done

# check if both parameters are provided
if [ -z "$model" ] || [ -z "$dataset" ]; then
    echo "Both model (--model) and dataset (--dataset) parameters are required" >&2
    echo "Usage: $0 --model <model> --dataset <dataset>" >&2
    echo "   or: $0 -m <model> -d <dataset>" >&2
    echo "Example: $0 --model Unsloth-Llama-3-8B-Instruct --dataset pl-court-instruct" >&2
    exit 1
fi

# =====Run the script using apptainer image=====
export NUM_PROC=$SLURM_CPUS_PER_GPU
export PYTHONPATH=$PYTHONPATH:.
export model
export dataset

export SFT_COMMAND="accelerate launch \
    --num_processes=$WORLD_SIZE \
    --num_machines=1 \
    --use-deepspeed \
    scripts/sft/fine_tune_deepspeed.py
        model=${model}
        dataset=${dataset}
"
srun --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    apptainer run \
        --fakeroot \
        --bind "$TMPDIR:$TMPDIR" \
        --nv \
        "$SIF_IMAGE_PATH" \
        bash -c "$SFT_COMMAND"

EXIT_CODE=$?
exit $EXIT_CODE
