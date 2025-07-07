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
    uv pip install --system -e . && \
    pip install flash-attn --no-build-isolation
