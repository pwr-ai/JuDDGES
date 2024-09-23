# Weaviate deployment

## Instruction
1. Prepare `.env` file with proper user names and API tokens
    ```bash
    cp example.env .env
    ```
2. Run containers through docker-compose
    ```bash
    docker compose up -d
    ```

## Remarks
* Persistent data will be stored inside mounted `./weaviate_data` path
* Deployment was tested on machine with 16 CPU, 64GB memory, and without GPU (vectors were computed outside weaviate instance, `t2v-transformers` used only for inference)
* see [scripts/embed/weaviate_example.py](../scripts/embed/weaviate_example.py) to see search example usage
