FROM nvcr.io/nvidia/pytorch:23.12-py3

WORKDIR /juddges

RUN apt-get update -qq && apt-get install --yes -q make git
# Install requirements
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
COPY Makefile .
RUN make install

# Fix issues occurring when importing torchvision and transformers-engine
# as these are not necessary.
RUN pip uninstall torchvision transformers-engine --yes

# Fix opencv issue (won't use it but import causes errors)
# see: https://github.com/opencv/opencv-python/issues/884#issuecomment-1806982912
RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2/ && pip install opencv-python-headless
