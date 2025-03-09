from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import PreTrainedTokenizer


class TextSplitter:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int | None = None,
        min_split_chars: int | None = None,
        take_n_first_chunks: int | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
    ) -> None:
        if tokenizer:
            self.splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)

        self.min_split_chars = min_split_chars
        self.take_n_first_chunks = take_n_first_chunks

    def __call__(self, txt: dict[str, Any]) -> dict[str, Any]:
        ids: list[str] = []
        chunk_ids: list[int] = []
        chunk_lens: list[int] = []
        chunks: list[str] = []

        for id_, text in zip(txt["judgement_id"], txt["text"]):
            current_chunks = self._split_text(text)

            if self.take_n_first_chunks:
                current_chunks = current_chunks[: self.take_n_first_chunks]

            chunks.extend(current_chunks)
            ids.extend([id_] * len(current_chunks))
            chunk_lens.extend([len(chunk) for chunk in current_chunks])
            chunk_ids.extend(range(len(current_chunks)))

        return {
            "judgement_id": ids,
            "chunk_id": chunk_ids,
            "chunk_len": chunk_lens,
            "chunk_text": chunks,
        }

    def _split_text(self, text: str) -> list[str]:
        chunks = self.splitter.split_text(text)

        if self.min_split_chars:
            chunks = [split for split in chunks if len(split) >= self.min_split_chars]

        return chunks
