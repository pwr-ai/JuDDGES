from typing import Any

import pytest

from juddges.preprocessing.text_chunker import TextChunker

TEXT = """
This is a first line of 37 char text.
This is a second line of 38 char text.
"""


@pytest.fixture
def input_data() -> dict[str, list[Any]]:
    return {
        "id": ["id1"],
        "text": [TEXT],
    }


def test_text_chunker_without_overlap(input_data: dict[str, list[Any]]) -> None:
    chunker = TextChunker(
        id_col="id",
        text_col="text",
        chunk_size=38,
        chunk_overlap=0,
    )
    chunks_data = chunker(input_data)

    target_data = {
        "id": ["id1", "id1", "id1"],
        chunker.CHUNK_ID_COL: [0, 1, 2],
        chunker.CHUNK_LEN_COL: [37, 32, 5],
        chunker.CHUNK_TEXT_COL: [
            "This is a first line of 37 char text.",
            "This is a second line of 38 char",
            "text.",
        ],
    }

    assert chunks_data == target_data


def test_text_chunker_with_overlap(input_data: dict[str, list[Any]]) -> None:
    chunker = TextChunker(
        id_col="id",
        text_col="text",
        chunk_size=38,
        chunk_overlap=10,
    )
    chunks_data = chunker(input_data)

    target_data = {
        "id": ["id1", "id1", "id1"],
        chunker.CHUNK_ID_COL: [0, 1, 2],
        chunker.CHUNK_LEN_COL: [37, 32, 13],
        chunker.CHUNK_TEXT_COL: [
            "This is a first line of 37 char text.",
            "This is a second line of 38 char",
            "38 char text.",
        ],
    }

    assert chunks_data == target_data
