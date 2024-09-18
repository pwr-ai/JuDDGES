import math
import os
from pathlib import Path

import typer
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm

from juddges.data.vector_database import WeaviateJudgementsDatabase

load_dotenv()
WV_HOST = os.getenv("WV_URL", "localhost")
WV_PORT = os.getenv("WV_PORT", "8080")
WV_GRPC_PORT = os.getenv("WV_GRPC_PORT", "50051")
WV_API_KEY = os.getenv("WV_API_KEY", None)

BATCH_SIZE = 64


def main(
    embeddings_dir: Path = typer.Option(...),
    batch_size: int = typer.Option(BATCH_SIZE),
) -> None:
    embs = load_dataset(str(embeddings_dir))["train"]
    with WeaviateJudgementsDatabase(WV_HOST, WV_PORT, WV_GRPC_PORT, WV_API_KEY) as db:
        for batch in tqdm(
            embs.iter(batch_size=batch_size),
            total=math.ceil(len(embs) / batch_size),
        ):
            objects = [
                {
                    "properties": {
                        "judgment_id": batch["_id"][i],
                        "chunk_id": batch["chunk_id"][i],
                        "chunk_text": batch["text_chunk"][i],
                    },
                    "vector": batch["embedding"][i],
                }
                for i in range(batch_size)
            ]
            db.insert_batch(
                collection=db.judgement_chunks_collection,
                objects=objects,
            )


if __name__ == "__main__":
    typer.run(main)
