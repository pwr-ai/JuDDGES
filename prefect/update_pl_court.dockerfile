FROM python:3.12-slim

WORKDIR /juddges
ENV PYTHONPATH=/juddges

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt && rm requirements.txt

COPY juddges/ ./juddges
COPY scripts/dataset/pl_court_data_pipeline.py ./scripts/dataset/pl_court_data_pipeline.py
COPY nbs/ ./nbs

ENTRYPOINT ["python", "scripts/dataset/pl_court_data_pipeline.py"]
