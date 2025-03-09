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
export WORKDIR

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
        -s|--stage)
            stage="$2"
            shift 2
            ;;
        *)
            echo "Invalid option: $1" >&2
            echo "Usage: $0 --model <model> --dataset <dataset> --stage <stage>" >&2
            echo "   or: $0 -m <model> -d <dataset> -s <stage>" >&2
            echo "Example: $0 --model Unsloth-Llama-3-8B-Instruct --dataset pl-court-instruct --stage sft" >&2
            exit 1
            ;;
    esac
done

# check if all parameters are provided
if [ -z "$model" ] || [ -z "$dataset" ] || [ -z "$stage" ]; then
    echo "Model (--model), dataset (--dataset) and stage (--stage) parameters are required" >&2
    echo "Usage: $0 --model <model> --dataset <dataset> --stage <stage>" >&2
    echo "   or: $0 -m <model> -d <dataset> -s <stage>" >&2
    echo "Example: $0 --model Unsloth-Llama-3-8B-Instruct --dataset pl-court-instruct --stage sft" >&2
    exit 1
fi

# validate stage parameter
if [ "$stage" != "sft" ] && [ "$stage" != "predict" ]; then
    echo "Invalid stage parameter. Allowed values: sft, predict" >&2
    exit 1
fi

# =====Run the script using apptainer image=====
export NUM_PROC=$SLURM_CPUS_PER_GPU
export PYTHONPATH=$PYTHONPATH:.
export model
export dataset

cd $WORKDIR

if [ "$stage" = "sft" ]; then
    export COMMAND="accelerate launch \
        --num_processes=$WORLD_SIZE \
        --num_machines=1 \
        --use-deepspeed \
        --zero-stage 2 \
        --mixed_precision=bf16 \
        --dynamo_backend=no \
        scripts/sft/fine_tune_deepspeed.py \
            model=${model} \
            dataset=${dataset} \
    "
else
    export COMMAND="python scripts/sft/predict.py \
        model=${model} \
        dataset=${dataset} \
        random_seed=42
    "
fi

echo "Running the following command with apptainer: $COMMAND"

srun --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    apptainer run \
        --fakeroot \
        --bind "$TMPDIR:$TMPDIR" \
        --bind "$WORKDIR:$WORKDIR" \
        --bind "$HOME:$HOME" \
        --nv \
        "$SIF_IMAGE_PATH" \
        bash -c "$COMMAND"

EXIT_CODE=$?
exit $EXIT_CODE
