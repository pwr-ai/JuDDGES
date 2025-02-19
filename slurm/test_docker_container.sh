#!/bin/bash

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


COMMAND="cd /root/juddges && python -c '$SCRIPT'"

# place HF_TOKEN in slurm/.env file
docker run \
    --rm \
    -it \
    --gpus "device=$CUDA_VISIBLE_DEVICES" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --volume $HOME/.cache/huggingface:/root/.cache/huggingface \
    --volume $PWD:/root/juddges \
    --env-file .env \
    --env HF_HOME=/root/.cache/huggingface \
    juddges:latest \
        bash -c "$COMMAND"
