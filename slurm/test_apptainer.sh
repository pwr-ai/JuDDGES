#!/bin/bash

#SBATCH --job-name=juddges_sft
#SBATCH --output=logs/%j-%x.log
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
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

export HF_TOKEN
export SIF_IMAGE_PATH

export NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' '\n'))
export WORLD_SIZE=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES))

# =====Run the script using apptainer image=====
export PYTHONPATH=$PYTHONPATH:.

SCRIPT=$(cat << EOF
from juddges.utils import load_and_resolve_config
from juddges.models.factory import get_model
from juddges.config import LLMConfig

llm_config = LLMConfig(**load_and_resolve_config("configs/model/llama_3.1_8b_instruct.yaml"))
print(llm_config)

model_pack = get_model(llm_config)
print(model_pack.model)
EOF
)

srun --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    apptainer run \
        --fakeroot \
        --bind "$TMPDIR:$TMPDIR" \
        --nv \
        "$SIF_IMAGE_PATH" \
        bash -c "$SCRIPT"

EXIT_CODE=$?
exit $EXIT_CODE
