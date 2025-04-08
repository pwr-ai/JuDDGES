FROM semitechnologies/transformers-inference:custom
ARG MODEL_NAME
RUN ./download.py
