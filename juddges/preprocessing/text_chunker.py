from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import PreTrainedTokenizer


class TextSplitter:
    CHUNK_ID_COL: str = "chunk_id"
    CHUNK_LEN_COL: str = "chunk_len"
    CHUNK_TEXT_COL: str = "chunk_text"

    def __init__(
        self,
        id_col: str,
        text_col: str,
        chunk_size: int,
        chunk_overlap: int | None = None,
        min_split_chars: int | None = None,
        take_n_first_chunks: int | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
    ) -> None:
        self.id_col = id_col
        self.text_col = text_col
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

        for id_, text in zip(txt[self.id_col], txt[self.text_col]):
            current_chunks = self._split_text(text)

            if self.take_n_first_chunks:
                current_chunks = current_chunks[: self.take_n_first_chunks]

            chunks.extend(current_chunks)
            ids.extend([id_] * len(current_chunks))
            chunk_lens.extend([len(chunk) for chunk in current_chunks])
            chunk_ids.extend(range(len(current_chunks)))

        return {
            self.id_col: ids,
            self.CHUNK_ID_COL: chunk_ids,
            self.CHUNK_LEN_COL: chunk_lens,
            self.CHUNK_TEXT_COL: chunks,
        }

    def _split_text(self, text: str) -> list[str]:
        chunks = self.splitter.split_text(text)

        if self.min_split_chars:
            chunks = [split for split in chunks if len(split) >= self.min_split_chars]

        return chunks
