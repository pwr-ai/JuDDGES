#!/usr/bin/env python3
"""
Example script demonstrating how to map HuggingFace dataset records to Weaviate documents.

This script shows how to:
1. Build an index for fast lookups
2. Retrieve dataset records by document_id or document_number
3. Join Weaviate documents with dataset data
4. Update Weaviate documents with raw_content from the dataset
"""

from loguru import logger

from juddges.data.dataset_mapper import DatasetToWeaviateMapper
from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase


def main():
    """Demonstrate dataset to Weaviate mapping."""
    # Initialize Weaviate connection
    logger.info("Connecting to Weaviate...")
    db = WeaviateLegalDocumentsDatabase(
        host="localhost",
        port=8222,
        grpc_port=8085,
    )

    # Initialize mapper with a dataset
    logger.info("Initializing mapper with pl-court-raw dataset...")
    mapper = DatasetToWeaviateMapper(
        db=db,
        dataset_name="juddges/pl-court-raw",
    )

    # Build index for fast lookups
    logger.info("Building dataset index...")
    mapper.build_index(
        id_field="judgment_id",  # Primary ID field in dataset
        secondary_id_field="docket_number",  # Secondary field for fallback
    )

    # Example 1: Get a dataset record by document_id
    logger.info("\n=== Example 1: Get dataset record by document_id ===")
    sample_doc = mapper.dataset[0]
    doc_id = sample_doc.get("judgment_id")
    logger.info(f"Looking up dataset record for document_id: {doc_id}")

    dataset_record = mapper.get_dataset_record(document_id=doc_id)
    if dataset_record:
        logger.info(f"Found dataset record with fields: {list(dataset_record.keys())}")
        logger.info(f"Raw text preview: {dataset_record.get('text', '')[:200]}...")
    else:
        logger.warning("Dataset record not found")

    # Example 2: Get a Weaviate document by document_id
    logger.info("\n=== Example 2: Get Weaviate document by document_id ===")
    weaviate_doc = mapper.get_weaviate_document(document_id=doc_id)
    if weaviate_doc:
        logger.info(f"Found Weaviate document with fields: {list(weaviate_doc.keys())}")
        logger.info(f"Has raw_content: {bool(weaviate_doc.get('raw_content'))}")
    else:
        logger.warning("Weaviate document not found")

    # Example 3: Find documents missing raw_content
    logger.info("\n=== Example 3: Find documents missing raw_content ===")
    missing_raw_content = mapper.get_missing_raw_content_documents()
    logger.info(f"Found {len(missing_raw_content)} documents missing raw_content")

    if missing_raw_content:
        logger.info(f"First missing document ID: {missing_raw_content[0].get('document_id')}")

    # Example 4: Update Weaviate documents with raw_content from dataset
    logger.info("\n=== Example 4: Update raw_content from dataset (dry run) ===")
    logger.info(
        "To update Weaviate documents with raw_content, uncomment the following line:"
    )
    logger.info("# updated_count = mapper.update_raw_content_from_dataset(")
    logger.info("#     raw_content_field='text',  # Field in dataset with raw text")
    logger.info("#     batch_size=100,")
    logger.info("#     document_id_field='judgment_id'")
    logger.info("# )")

    # Example 5: Join Weaviate documents with dataset fields
    logger.info("\n=== Example 5: Join Weaviate with dataset ===")
    # Get a few Weaviate documents
    collection = db.legal_documents_collection
    response = collection.query.fetch_objects(limit=3)
    weaviate_docs = [obj.properties for obj in response.objects]

    # Enrich with dataset fields
    enriched_docs = mapper.join_dataset_to_weaviate(
        weaviate_documents=weaviate_docs,
        dataset_fields=["text", "excerpt", "judgment_date"],  # Specific fields only
    )

    logger.info(f"Enriched {len(enriched_docs)} documents")
    if enriched_docs:
        logger.info(f"Enriched fields: {[k for k in enriched_docs[0].keys() if k.startswith('dataset_')]}")

    logger.info("\n=== Done ===")


if __name__ == "__main__":
    main()
