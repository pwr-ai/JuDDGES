FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

RUN pip install --no-cache-dir \
    "sentence-transformers>=3.3" "transformers>=4.51" \
    pyarrow pandas numpy "weaviate-client>=4.24" \
    loguru rich langchain-text-splitters pytest

# Brotli 1.1 breaks httpx2/huggingface_hub downloads (process() takes no keyword arguments)
RUN pip uninstall -y Brotli

WORKDIR /workspace
ENV PYTHONPATH=/workspace PYTHONUNBUFFERED=1
