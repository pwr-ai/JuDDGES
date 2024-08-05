FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip --no-cache-dir install -r requirements.txt

COPY juddges juddges
COPY scripts scripts

ENV PYTHONPATH .

#ENTRYPOINT ["/bin/bash"]