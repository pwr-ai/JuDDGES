# Dataset preparation

Scripts for dataset preparation are located in `dataset` directory, and should be run from the root
of the repository.

## 1. Building the dataset

Dataset was downloaded from open API of [Polish Court Judgements](https://orzeczenia.ms.gov.pl/).
The following procedure will download data and store it in `MongoDB`. Whenever script interacts with outside environment (storing data in `mongodb` or pushing files to `huggingface-hub`) it is run outisde `dvc`.
Prior to downloading, make sure you have proper environment variable set in `.env` file:

```dotenv
MONGO_URI=<mongo_uri_including_password>
MONGO_DB_NAME="datasets"
```

### Raw dataset

1. Download judgements metadata - this will store metadata in the database:
    ```shell
    PYTHONPATH=. python scripts/dataset/download_pl_metadata.py
    ```

2. Download judgements text (XML content of judgements) - this will alter the database with content:
    ```shell
    PYTHONPATH=. python scripts/dataset/download_pl_additional_data.py \
        --data-type content \
        --n-jobs 10
    ```

3. Download additional details available for each judgement - this will alter the database with
   acquired details:
    ```shell
    PYTHONPATH=. python scripts/dataset/download_pl_additional_data.py \
        --data-type details \
        --n-jobs 10
    ```

4. Map id of courts and departments to court name:
    ```shell
    PYTHONPATH=. python scripts/dataset/map_court_dep_id_2_name.py --n-jobs 10
    ```
   _Remark_: File with mapping available at `data/datasets/pl/court_id_2_name.csv` was prepared based
   on data published on: https://orzeczenia.wroclaw.sa.gov.pl/indices

5. Extract raw text from XML content and details of judgments not available through API:
    ```shell
    PYTHONPATH=. python scripts/dataset/extract_pl_xml.py --n-jobs 10
    ```

6. For further processing prepare local dataset dump in `parquet` file, version it with dvc and push
   to remote storage:
    ```shell
    PYTHONPATH=.  python scripts/dataset/dump_pl_dataset.py \
        --file-name data/datasets/pl/raw/raw.parquet
    dvc add data/datasets/pl/raw/raw.parquet && dvc push 
    ```
7. Generate dataset card for `pl-court-raw`
    ```shell
    dvc repro raw_dataset_readme && dvc push
    ```

9. Upload `pl-court-raw` dataset (with card) to huggingface
    ```shell
    PYTHONPATH=. python scripts/dataset/push_raw_dataset.py --repo-id "JuDDGES/pl-court-raw"
   ```

### Instruction dataset
10. Generate intruction dataset and upload it to huggingface (`pl-court-instruct`)
    ```shell
    NUM_JOBS=8 dvc repro build_instruct_dataset
    ```
    
11. Generate dataset card for `pl-court-instruct`
    ```shell
    dvc repro instruct_dataset_readme && dvc push
    ```
    
12. Upload `pl-court-instruct` dataset card to huggingface
   ```shell
   PYTHONPATH=. scripts/dataset/push_instruct_readme.py --repo-id JuDDGES/pl-court-instruct
   ```

### Graph dataset
13. Embed judgments with pre-trained lanuage model (documents arechunked and embeddings are computed per chunk)
    ```shell
    CUDA_VISIBLE_DEVICES=<device_number> dvc repro embed
    ```

14. Aggregate embeddings of chunks into embeddings of document
    ```shell
    NUM_PROC=4 dvc repro embed aggregate_embeddings
    ```

15. Eventually ingest data to `mongodb` (e.g. for vector search)
    ```shell
    PYTHONPATH=. python scripts/embed/ingest.py --embeddings-file <embeddgings>
    ```

16. Generate graph dataset
    ```shell
    dvc repro embed build_graph_dataset
    ```

17. Generate dataset card and upload it to huggingface (remember to be logged in to `huggingface` or set `HUGGING_FACE_HUB_TOKEN` env variable)
    ```shell
    PYTHONPATH=. python scripts/dataset/upload_graph_dataset.py \
        --root-dir <dir_to_dataset> \
        --repo-id JuDDGES/pl-court-graph \
        --commit-message <message>
    ```
