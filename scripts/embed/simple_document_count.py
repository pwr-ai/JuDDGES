#!/usr/bin/env python3
"""
Simple document count check for Weaviate database.
Returns raw counts without rich formatting for use in other scripts.
"""

import sys
from typing import Dict

from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase


def get_document_counts() -> Dict[str, int]:
    """
    Get document counts from both collections.

    Returns:
        Dict[str, int]: Dictionary with collection names as keys and document counts as values.
    """
    counts = {"legal_documents": 0, "document_chunks": 0}

    try:
        with WeaviateLegalDocumentsDatabase() as db:
            # Get legal documents count
            try:
                counts["legal_documents"] = db.get_collection_size(db.legal_documents_collection)
            except Exception as e:
                print(f"Error getting legal documents count: {e}", file=sys.stderr)

            # Get document chunks count
            try:
                counts["document_chunks"] = db.get_collection_size(db.document_chunks_collection)
            except Exception as e:
                print(f"Error getting document chunks count: {e}", file=sys.stderr)

    except Exception as e:
        print(f"Database connection error: {e}", file=sys.stderr)

    return counts


def main():
    """Main function to run the simple document count check."""
    counts = get_document_counts()

    print(f"legal_documents: {counts['legal_documents']}")
    print(f"document_chunks: {counts['document_chunks']}")
    print(f"total: {counts['legal_documents'] + counts['document_chunks']}")

    # Return appropriate exit code
    if counts["legal_documents"] == 0 and counts["document_chunks"] == 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
