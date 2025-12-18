FROM semitechnologies/transformers-inference:custom

ARG MODEL_NAME
ARG ENABLE_CUDA

RUN echo "MODEL_NAME: $MODEL_NAME"
RUN echo "ENABLE_CUDA: $ENABLE_CUDA"

RUN ./download.py
