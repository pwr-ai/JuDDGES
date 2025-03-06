FROM python:3.12-slim

ARG USER_UID=1000
ARG USER_GID=1000

# Install git-lfs to handle hf-hub
RUN apt-get update && apt-get install -y git-lfs && git lfs install

# create home to keep default cache dirs
RUN groupadd --gid $USER_GID juddges_user && \
    useradd --uid $USER_UID --gid juddges_user --create-home juddges_user

USER juddges_user

ENV HOME=/home/juddges_user
ENV PYTHONPATH=/juddges

WORKDIR /juddges

# install requirements and copy the code
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt && rm requirements.txt

COPY juddges/ ./juddges
COPY scripts/dataset/pl_court_data_pipeline.py ./scripts/dataset/pl_court_data_pipeline.py
COPY nbs/ ./nbs

ENTRYPOINT ["python", "scripts/dataset/pl_court_data_pipeline.py"]
