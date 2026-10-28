# Dataset preparation

Scripts for dataset preparation are located in `dataset` directory, and should be run from the root
of the repository.

## 1. Building the dataset

Dataset was downloaded from open API of [Polish Court Judgements](https://orzeczenia.ms.gov.pl/).
The original download pipeline (which stored intermediate data in MongoDB) has been removed from the
repository. The steps below start from the raw dataset dump in `parquet` format, versioned with dvc
under `data/datasets/pl/raw`. Whenever a script interacts with outside environment (e.g. pushing files
to `huggingface-hub`) it is run outside `dvc`.

### Raw dataset

1. Generate dataset card for `pl-court-raw`

    ```shell
    dvc repro raw_dataset_readme && dvc push
    ```

2. Upload `pl-court-raw` dataset (with card) to huggingface

    ```shell
    PYTHONPATH=. python scripts/dataset/push_raw_dataset.py --repo-id "JuDDGES/pl-court-raw" --commit-message <commit_message>
   ```

### Instruction dataset

3. Generate intruction dataset and upload it to huggingface (`pl-court-instruct`)

    ```shell
    NUM_JOBS=8 dvc repro build_instruct_dataset
    ```

4. Generate dataset card for `pl-court-instruct`

    ```shell
    dvc repro instruct_dataset_readme && dvc push
    ```

5. Upload `pl-court-instruct` dataset card to huggingface

   ```shell
   PYTHONPATH=. scripts/dataset/push_instruct_readme.py --repo-id JuDDGES/pl-court-instruct
   ```

### Graph dataset

6. Embed judgments with pre-trained language model (documents are chunked and embeddings are computed per chunk)

    ```shell
    CUDA_VISIBLE_DEVICES=<device_number> dvc repro embed
    ```

7. Aggregate embeddings of chunks into embeddings of document

    ```shell
    NUM_PROC=4 dvc repro embed aggregate_embeddings
    ```

8. Generate graph dataset

    ```shell
    dvc repro embed build_graph_dataset
    ```

9. Generate dataset card and upload it to huggingface (remember to be logged in to `huggingface` or set `HUGGING_FACE_HUB_TOKEN` env variable)

    ```shell
    PYTHONPATH=. python scripts/dataset/upload_graph_dataset.py \
        --root-dir <dir_to_dataset> \
        --repo-id JuDDGES/pl-court-graph \
        --commit-message <message>
    ```
