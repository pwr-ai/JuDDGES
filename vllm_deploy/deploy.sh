#!/usr/bin/env bash
# Usage: ./run_vllm.sh <GPU_INDEX>

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <gpu_index>"
  exit 1
fi

GPU_INDEX=$1

# Ensure .env exists
if [ ! -f ".env" ]; then
  echo ".env file not found! Please create one with HUGGING_FACE_HUB_TOKEN."
  exit 1
fi


# Run Docker container with specified GPU
docker run \
    --name vllm-local \
    --rm \
    --runtime nvidia \
    --gpus "device=${GPU_INDEX}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env-file vllm_deploy/.env \
    -p 127.0.0.1:8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-model-len 32000 \
    --enable-prefix-caching
