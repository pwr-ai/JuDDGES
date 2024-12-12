FROM nvcr.io/nvidia/pytorch:24.11-py3

WORKDIR /judddges

RUN apt-get update -qq && apt-get install --yes -q make git

# Install requirements
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
COPY Makefile .
RUN make install

# Fix issues occurring when importing torchvision and transformers-engine
# as these are not necessary.
RUN pip uninstall torchvision transformers-engine --yes
