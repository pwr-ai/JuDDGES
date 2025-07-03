#!/usr/bin/env bash
# Usage: ./deploy.sh --llm <model_name> --gpu <gpu_index> --context <max_model_len>

# Parse named arguments
ADDITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -l|--llm)
      LLM="$2"
      shift 2
      ;;
    -g|--gpu)
      GPU_INDEX="$2"
      shift 2
      ;;
    -c|--context)
      CONTEXT="$2"
      shift 2
      ;;
    *)
      # Collect additional arguments for vLLM
      ADDITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Validate all parameters are provided
if [[ -z "$LLM" || -z "$GPU_INDEX" || -z "$CONTEXT" ]]; then
  echo "Error: All parameters are required"
  echo "Usage: $0 --llm <model_name> --gpu <gpu_index> --context <max_model_len>"
  exit 1
fi

# Validate integer types
if ! [[ "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
  echo "Error: GPU index must be an integer"
  exit 1
fi

if [[ "$GPU_INDEX" -gt 7 ]]; then
  echo "Error: GPU index must be at most 7"
  exit 1
fi

if ! [[ "$CONTEXT" =~ ^[0-9]+$ ]]; then
  echo "Error: Context length must be an integer"
  exit 1
fi

# Ensure .env exists in script directory
SCRIPT_DIR="$(dirname "$0")"
if [ ! -f "${SCRIPT_DIR}/.env" ]; then
  echo ".env file not found in ${SCRIPT_DIR}! Please create one with HUGGING_FACE_HUB_TOKEN."
  exit 1
fi

# Run Docker container with specified parameters
docker run \
    --name vllm-local \
    --rm \
    --runtime nvidia \
    --gpus "device=${GPU_INDEX}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env-file "${SCRIPT_DIR}/.env" \
    -p 127.0.0.1:8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model "${LLM}" \
    --max-model-len "${CONTEXT}" \
    --enable-prefix-caching \
    "${ADDITIONAL_ARGS[@]}"
