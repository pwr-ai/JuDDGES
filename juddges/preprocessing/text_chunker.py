from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    def __init__(
        self,
        chunk_size: int,
        min_split_chars: int | None = None,
        take_n_first_chunks: int | None = None,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
        self.min_split_chars = min_split_chars
        self.take_n_first_chunks = take_n_first_chunks

    def __call__(self, txt: dict[str, Any]) -> dict[str, Any]:
        ids, chunk_ids, chunks = [], [], []

        for id_, text in zip(txt["_id"], txt["text"]):
            current_chunks = self._split_text(text)

            if self.take_n_first_chunks:
                current_chunks = current_chunks[: self.take_n_first_chunks]

            chunks.extend(current_chunks)
            ids.extend([id_] * len(current_chunks))
            chunk_ids.extend(range(len(current_chunks)))

        return {"_id": ids, "chunk_id": chunk_ids, "text_chunk": chunks}

    def _split_text(self, text: str) -> list[str]:
        chunks = self.splitter.split_text(text)

        if self.min_split_chars:
            chunks = [split for split in chunks if len(split) >= self.min_split_chars]

        return chunks
