import math
from pathlib import Path
from dotenv import load_dotenv
import torch
import tqdm
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
    cursor = collection.find(query, {"_id": 1})
    num_docs_to_update = collection.count_documents(query)
    batched_cursor = BatchedDatabaseCursor(cursor, batch_size=BATCH_SIZE, prefetch=True)

    embeddings = torch.load(embeddings_file)
    ingest_embeddings = BatchDatabaseUpdate(mongo_uri, lambda doc: embeddings.get(doc["_id"]))

    for batch in tqdm(batched_cursor, total=math.ceil(num_docs_to_update / batch_size)):
        ingest_embeddings(batch)


if __name__ == "__main__":
    typer.run(main)
