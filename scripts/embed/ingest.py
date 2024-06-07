import math
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import torch
from loguru import logger
from tqdm.auto import tqdm
import typer

from juddges.data.database import BatchDatabaseUpdate, BatchedDatabaseCursor, get_mongo_collection

load_dotenv()

BATCH_SIZE = 64


def main(
    mongo_uri: str = typer.Option(..., envvar="MONGO_URI"),
    batch_size: int = typer.Option(BATCH_SIZE),
    embeddings_file: Path = typer.Option(...),
):
    collection = get_mongo_collection(mongo_uri)
    query = {"embedding": {"$exists": False}}
    num_docs_to_update = collection.count_documents(query)
    logger.info(f"Found {num_docs_to_update} documents without embedding")

    cursor = collection.find(query, {"_id": 1}, batch_size=batch_size)
    batched_cursor = BatchedDatabaseCursor(cursor, batch_size=batch_size, prefetch=True)

    logger.info("Loading embeddings...")
    emb_getter = EmbeddingGetter(embeddings_file)
    ingest_embeddings = BatchDatabaseUpdate(mongo_uri, emb_getter)

    logger.info("Ingesting embeddings...")
    for batch in tqdm(iter(batched_cursor), total=math.ceil(num_docs_to_update / batch_size)):
        ingest_embeddings(batch)


class EmbeddingGetter:
    def __init__(self, embeddings_file: Path):
        self.embeddings = torch.load(embeddings_file)

    def __call__(self, doc: dict[str, Any]) -> dict[str, list[float] | None]:
        emb = self.embeddings.get(doc["_id"])
        if emb is not None:
            emb = emb.tolist()
        return {"embedding": emb}


if __name__ == "__main__":
    typer.run(main)
