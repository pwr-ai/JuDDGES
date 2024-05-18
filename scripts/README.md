# Dataset preparation

Scripts for dataset preparation are located in `dataset` directory, and should be run from the root
of the repository.

### 1. Downloading the dataset

Dataset was downloaded from open API of [Polish Court Judgements](https://orzeczenia.ms.gov.pl/).
The following procedure will download data and store it in `MongoDB`.
Prior to downloading, make sure you have proper environment variable set in `.env` file:

```dotenv
MONGO_URI=<mongo_uri>
MONGO_DB_NAME="juddges"
```

1. Download judgements metadata - this will store metadata in the database:
    ```shell
    PYTHONPATH=. python scripts/dataset/download_pl_metadata.py
    ```

2. Download judgements text (XML content of judgements) - this will alter the database with content:
    ```shell
    PYTHONPATH=. python scripts/dataset/download_pl_additional_data.py --data-type content --n-jobs 10
    ```

3. Download additional details available for each judgement - this will alter the database with
   acquired details:
    ```shell
    PYTHONPATH=. python scripts/dataset/download_pl_additional_data.py --data-type details --n-jobs 10
    ```

4. Map id of courts and departments to court name:
    ```shell
    PYTHONPATH=. python scripts/dataset/map_court_dep_id_2_name.py --n-jobs 10
    ```
   _Remark_: File with mapping available at `data/datasets/pl/court_id_2_name.csv` was prepared based
   on data published on: https://orzeczenia.wroclaw.sa.gov.pl/indices


5. For further processing prepare local dataset dump in `parquet` file, version it with dvc and push
   to remote storage:
    ```shell
    PYTHONPATH=.  python scripts/dataset/dump_pl_dataset.py --file-name data/datasets/pl/raw/raw.parquet
    dvc add data/datasets/pl/raw/raw.parquet && dvc push 
    ```

6. Extract raw text from content XML and details of judgments not available through API, eventually
   ingest it to the database:
    ```shell
    PYTHONPATH=. python scripts/dataset/extract_pl_xml.py --dataset-dir data/datasets/pl/raw/ --target-dir data/datasets/pl/text --num-proc 6 --ingest
    ```