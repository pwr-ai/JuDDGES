import re
from typing import Any, Dict, List

import streamlit as st
from pymongo.collection import Collection
from sentence_transformers import SentenceTransformer

from juddges.data.database import get_mongo_collection
from juddges.settings import TEXT_EMBEDDING_MODEL


@st.cache_resource
def get_judgments_collection(collection_name: str = "pl-court") -> Collection:
    """Get MongoDB collection for judgments."""
    return get_mongo_collection(collection_name=collection_name)


@st.cache_resource
def get_embedding_model() -> Any:
    """Get the sentence transformer model for text embeddings."""
    return SentenceTransformer(TEXT_EMBEDDING_MODEL)


def process_highlights(item: Dict[str, Any]) -> str:
    """Process and format highlighted text segments.

    Args:
        item: Document containing highlights and text

    Returns:
        Processed text with HTML highlight markup
    """
    if "highlights" not in item or "text" not in item:
        return item.get("text", "")

    full_text = item["text"]
    highlight_terms = {
        segment["value"]
        for highlight in item["highlights"]
        for segment in highlight["texts"]
        if segment["type"] == "hit"
    }

    for term in highlight_terms:
        escaped_term = term.replace("[", "\\[").replace("]", "\\]")
        pattern = f"({escaped_term})"
        replacement = "<span class='highlight'>\\1</span>"
        full_text = re.sub(pattern, replacement, full_text, flags=re.IGNORECASE)

    return full_text


def format_highlight_segments(highlight: Dict[str, List]) -> str:
    """Format individual highlight segments with HTML markup.

    Args:
        highlight: Dictionary containing highlight text segments

    Returns:
        Formatted text with highlight markup
    """
    text = ""
    for segment in highlight["texts"]:
        if segment["type"] == "hit":
            text += f"<span class='highlight'>{segment['value']}</span>"
        else:
            text += segment["value"]
    return text
