from pathlib import Path
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
import torch
import typer
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

MODEL = "sdadas/mmlw-roberta-large"
MAX_CHUNK_SIZE = 500
MIN_SPLIT_CHARS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(
    dataset_dir: Path = typer.Option(..., help="Path to the dataset directory"),
    model: str = typer.Option(MODEL, help="Name of the model from HF hub"),
    max_chunk_size: int = typer.Option(MAX_CHUNK_SIZE, help="Maximum number of chars in a chunk"),
    min_split_chars: int = typer.Option(
        MIN_SPLIT_CHARS,
        help="Minimum number of chars to keep a chunk",
    ),
    batch_size: int = typer.Option(..., help="Batch size for tokenization"),
    num_jobs: int = typer.Option(..., help="Number of parallel jobs to use"),
    device: str = typer.Option(DEVICE, help="Device to use for the model"),
) -> None:
    dataset = load_from_disk(dataset_dir)

    split_worker = TextSplitter(chunk_size=max_chunk_size, min_split_chars=min_split_chars)
    ds = dataset.select_columns(["_id", "text"]).map(
        split_worker,
        batched=True,
        num_proc=num_jobs,
        remove_columns=["_id", "text"],
    )
    logger.info(f"Dataset split into {ds.num_rows} chunks")

    model = SentenceTransformer(model).to(device)
    ds = ds.map(Encoder(model), batched=True, batch_size=batch_size, num_proc=None)
    ds.save_to_disk(dataset_dir.parent / "embeddings", num_shards=8)


class TextSplitter:
    def __init__(self, chunk_size: int, min_split_chars: int | None = None) -> None:
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
        self.min_split_chars = min_split_chars

    def __call__(self, txt: dict[str, Any]) -> dict[str, Any]:
        ids, chunks = [], []

        for id_, text in zip(txt["_id"], txt["text"]):
            current_chunks = self._split_text(text)
            chunks.extend(current_chunks)
            ids.extend([id_] * len(current_chunks))

        return {"_id": ids, "text_chunk": chunks}

    def _split_text(self, text: str) -> list[str]:
        chunks = self.splitter.split_text(text)

        if self.min_split_chars:
            chunks = [split for split in chunks if len(split) >= self.min_split_chars]

        return chunks


class Encoder:
    def __init__(self, model: SentenceTransformer) -> None:
        self.model = model

    def __call__(self, items: dict[str, Any]) -> Any:
        return {"embeddings": self.model.encode(items["text_chunk"])}


if __name__ == "__main__":
    typer.run(main)
