FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /tmp

# hadolint ignore=DL3008, DL3013
RUN apt-get update \
    && apt-get install --no-install-recommends -y bash \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    zsh \
    gdebi-core \
    vim \
    libgl1 \
    libglib2.0-0 -y \
    less \
    git-lfs \
    poppler-utils \
    tesseract-ocr \
    fonts-freefont-ttf \
    && wget --progress=dot:giga https://github.com/quarto-dev/quarto-cli/releases/download/v1.5.17/quarto-1.5.17-linux-amd64.deb \
    && gdebi -n quarto-1.5.17-linux-amd64.deb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists \
    && rm -rf /tmp

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN chmod 1777 /tmp \
    && sh -c "$(wget --progress=dot:giga -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t https://github.com/denysdovhan/spaceship-prompt \
    -a 'SPACESHIP_PROMPT_ADD_NEWLINE="false"' \
    -a 'SPACESHIP_PROMPT_SEPARATE_LINE="false"' \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions

WORKDIR /app

COPY pyproject.toml .
COPY juddges/ ./juddges/
COPY scripts/ ./scripts/
COPY README.md .

RUN pip install uv && \
    uv pip install --system -e "." && \
    uv pip install --system --no-build-isolation \
        "accelerate>=1.2.1" \
        "bitsandbytes>=0.45.0" \
        "chardet>=5.2.0" \
        "deepdiff>=7.0.1" \
        "deepspeed>=0.15.4" \
        "flash-attn>=2.7.4.post1" \
        "langchain-community>=0.3.8" \
        "langchain-openai>=0.2.10" \
        "langchain-text-splitters>=0.3.2" \
        "lightning_fabric>=2.3.1" \
        "mpire>=2.10.0" \
        "openpyxl>=3.1.2" \
        "peft>=0.14.0" \
        "pyarrow>=15.0.0" \
        "pymongo>=4.3.3" \
        "pytz>=2024.1" \
        "PyYAML>=6.0.1" \
        "scikit-learn>=1.5.0" \
        "seaborn>=0.13.2" \
        "streamlit>=1.40.2" \
        "tenacity>=8.2.3" \
        "tensorboard>=2.16.2" \
        "torchmetrics>=1.4.0" \
        "torch_geometric>=2.5.3" \
        "trl>=0.12.2" \
        "umap-learn>=0.5.5" \
        "vllm>=0.6.4.post1" \
        "wandb>=0.19.0" \
        "xmltodict>=0.13.0" \
        "xlsxwriter>=3.2.0"
