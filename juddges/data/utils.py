"""
Utility functions for Weaviate ingestion.
"""

import hashlib
from typing import Optional


def generate_deterministic_uuid(document_id: str, chunk_id: Optional[str] = None) -> str:
    """
    Generate a deterministic UUID for a document or chunk.

    This ensures that if ingestion is interrupted and restarted, the same object
    will receive the same UUID, allowing for proper deduplication.

    Args:
        document_id: The document ID (judgment_id)
        chunk_id: The chunk ID (if generating UUID for a chunk)

    Returns:
        A deterministic UUID as string
    """
    if chunk_id:
        # For chunks, combine document_id and chunk_id
        key = f"{document_id}_{chunk_id}"
    else:
        # For documents, use only document_id
        key = document_id

    # Create SHA-256 hash and use the first 32 chars (16 bytes) for UUID generation
    hash_obj = hashlib.sha256(key.encode("utf-8"))
    hex_digest = hash_obj.hexdigest()

    # Format as UUID (8-4-4-4-12 format)
    uuid = f"{hex_digest[:8]}-{hex_digest[8:12]}-{hex_digest[12:16]}-{hex_digest[16:20]}-{hex_digest[20:32]}"

    return uuid
