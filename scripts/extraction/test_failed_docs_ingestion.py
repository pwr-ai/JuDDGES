#!/usr/bin/env python3
"""Test ingestion of the 4 failed documents with comma-separated keywords."""

import json
import sys
from pathlib import Path

import requests
import weaviate
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from juddges.extraction.gemini_chain import build_update_payload

# Failed document IDs from ingestion_report.json
FAILED_DOC_IDS = [
    "6209595f-d95d-5993-8291-94d8a5f89c74",
    "9b31e490-4679-539d-b0ba-3e7500507bee",
    "a0ba7b0e-23f5-52db-a01a-73f7641a1567",
    "7834d067-87ec-5531-96f1-373a59811d29",
]


def load_extracted_data(extraction_file: Path) -> dict:
    """Load extracted data from JSONL file."""
    extractions = {}
    with open(extraction_file) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                doc_id = data.get("document_id")
                if doc_id in FAILED_DOC_IDS:
                    extractions[doc_id] = data
    return extractions


def get_weaviate_document(weaviate_host: str, weaviate_port: int, doc_id: str) -> dict | None:
    """Fetch document from Weaviate REST API."""
    weaviate_uuid = weaviate.util.generate_uuid5(doc_id)
    url = f"http://{weaviate_host}:{weaviate_port}/v1/objects/LegalDocuments/{weaviate_uuid}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Document {doc_id} not found in Weaviate: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching document {doc_id}: {e}")
        return None


def update_weaviate_document(
    weaviate_host: str,
    weaviate_port: int,
    doc_id: str,
    extracted_data: dict,
    overwrite: bool = False,
) -> bool:
    """Update a single document in Weaviate."""
    weaviate_uuid = weaviate.util.generate_uuid5(doc_id)
    url = f"http://{weaviate_host}:{weaviate_port}/v1/objects/LegalDocuments/{weaviate_uuid}"

    # Get existing document
    existing_doc = get_weaviate_document(weaviate_host, weaviate_port, doc_id)
    if not existing_doc:
        logger.error(f"Cannot update {doc_id} - document not found in Weaviate")
        return False

    existing_properties = existing_doc.get("properties", {})

    # Build update payload
    payload = build_update_payload(
        extracted_data=extracted_data.get("extracted_data", {}),
        existing_properties=existing_properties,
        overwrite_existing=overwrite,
    )

    if not payload:
        logger.info(f"No updates needed for {doc_id}")
        return True

    logger.info(f"Updating {doc_id} with payload: {json.dumps(payload, indent=2)}")

    # Send PATCH request
    try:
        response = requests.patch(url, json=payload, timeout=30)

        if response.status_code == 204:
            logger.info(f"✓ Successfully updated {doc_id}")
            return True
        else:
            logger.error(
                f"✗ Failed to update {doc_id}: {response.status_code} - {response.text}"
            )
            return False

    except Exception as e:
        logger.error(f"✗ Exception updating {doc_id}: {e}")
        return False


def main():
    """Main function to test ingestion of failed documents."""
    weaviate_host = "localhost"
    weaviate_port = 8084

    # Load extracted data
    extraction_file = Path("data/extraction_results/sample_judgment_extraction_fixed.jsonl")
    if not extraction_file.exists():
        logger.error(f"Extraction file not found: {extraction_file}")
        return

    extractions = load_extracted_data(extraction_file)
    logger.info(f"Loaded {len(extractions)} failed documents from extraction file")

    # Test each document
    success_count = 0
    for doc_id in FAILED_DOC_IDS:
        if doc_id not in extractions:
            logger.warning(f"Document {doc_id} not found in extraction file")
            continue

        logger.info(f"\n{'='*80}")
        logger.info(f"Testing document: {doc_id}")
        logger.info(f"{'='*80}")

        extracted_data = extractions[doc_id]

        # Show keywords value
        keywords = extracted_data.get("extracted_data", {}).get("keywords")
        logger.info(f"Keywords type: {type(keywords)}")
        logger.info(f"Keywords value: {keywords}")

        # Attempt update
        if update_weaviate_document(weaviate_host, weaviate_port, doc_id, extracted_data):
            success_count += 1

    logger.info(f"\n{'='*80}")
    logger.info(f"Results: {success_count}/{len(FAILED_DOC_IDS)} documents successfully updated")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
