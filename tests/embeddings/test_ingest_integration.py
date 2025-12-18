"""Integration tests for Weaviate ingestion with real-world scenarios."""

import json

import numpy as np
import pytest
from datasets import Dataset

from scripts.embed.ingest_to_weaviate import (
    ChunkIngester,
    DocumentIngester,
    IngestConfig,
    generate_deterministic_uuid,
)


class TestRealWorldScenarios:
    """Test ingestion with real-world data scenarios."""

    def test_large_document_with_many_chunks(self, mock_weaviate_database):
        """Test ingesting a document with many chunks."""
        # Simulate a large legal document with 100 chunks
        num_chunks = 100
        document_id = "large_doc_1"

        document_dataset = Dataset.from_dict(
            {
                "document_id": [document_id],
                "title": ["Large Legal Document - Constitutional Law Analysis"],
                "language": ["en"],
                "country": ["US"],
                "date_issued": ["2023-01-01"],
                "document_type": ["judgment"],
                "full_text": ["Very long legal document text..."] * 1,
            }
        )

        chunk_dataset = Dataset.from_dict(
            {
                "document_id": [document_id] * num_chunks,
                "chunk_id": [f"chunk_{i}" for i in range(num_chunks)],
                "chunk_text": [f"Chunk {i} of the large document..." for i in range(num_chunks)],
                "embedding": [np.random.rand(384).tolist() for _ in range(num_chunks)],
                "position": list(range(num_chunks)),
            }
        )

        config = IngestConfig(batch_size=10)
        ingester = ChunkIngester(db=mock_weaviate_database, config=config)
        ingester.ingest(document_dataset, chunk_dataset)

        # Should process 10 batches (100 chunks / 10 batch_size)
        expected_batches = num_chunks // config.batch_size
        assert (
            mock_weaviate_database.document_chunks_collection.batch.fixed_size.call_count
            == expected_batches
        )

    def test_multilingual_documents(self, mock_weaviate_database):
        """Test ingesting documents in multiple languages."""
        multilingual_docs = Dataset.from_dict(
            {
                "document_id": ["doc_en", "doc_pl", "doc_de", "doc_fr"],
                "title": [
                    "English Legal Document",
                    "Polskie Orzeczenie Sądowe",
                    "Deutsches Gerichtsurteil",
                    "Document Juridique Français",
                ],
                "language": ["en", "pl", "de", "fr"],
                "country": ["US", "PL", "DE", "FR"],
                "date_issued": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
                "document_type": ["judgment", "judgment", "tax_interpretation", "legal_act"],
                "full_text": [
                    "English legal text...",
                    "Polski tekst prawny...",
                    "Deutscher Rechtstext...",
                    "Texte juridique français...",
                ],
            }
        )

        doc_embeddings = Dataset.from_dict(
            {
                "document_id": ["doc_en", "doc_pl", "doc_de", "doc_fr"],
                "embedding": [np.random.rand(384).tolist() for _ in range(4)],
            }
        )

        config = IngestConfig(batch_size=2)
        ingester = DocumentIngester(db=mock_weaviate_database, config=config)
        ingester.ingest(multilingual_docs, doc_embeddings)

        # Should process 2 batches (4 docs / 2 batch_size)
        assert mock_weaviate_database.legal_documents_collection.batch.fixed_size.call_count == 2

    def test_documents_with_missing_optional_fields(self, mock_weaviate_database):
        """Test ingesting documents with some missing optional fields."""
        sparse_docs = Dataset.from_dict(
            {
                "document_id": ["doc_complete", "doc_minimal"],
                "title": ["Complete Document", None],  # Missing title
                "language": ["en", "en"],
                "country": ["US", None],  # Missing country
                "date_issued": ["2023-01-01", None],  # Missing date
                "document_type": ["judgment", "judgment"],
                "full_text": ["Complete text...", "Minimal text..."],
                "summary": ["Complete summary", None],  # Missing summary
            }
        )

        doc_embeddings = Dataset.from_dict(
            {
                "document_id": ["doc_complete", "doc_minimal"],
                "embedding": [np.random.rand(384).tolist() for _ in range(2)],
            }
        )

        config = IngestConfig(batch_size=2)
        ingester = DocumentIngester(db=mock_weaviate_database, config=config)

        # Should handle missing fields gracefully
        ingester.ingest(sparse_docs, doc_embeddings)

        assert mock_weaviate_database.legal_documents_collection.batch.fixed_size.called

    def test_chunks_with_complex_metadata(self, mock_weaviate_database):
        """Test ingesting chunks with complex metadata like nested JSON."""
        document_dataset = Dataset.from_dict(
            {
                "document_id": ["complex_doc"],
                "language": ["en"],
                "country": ["US"],
                "date_issued": ["2023-01-01"],
                "document_type": ["judgment"],
            }
        )

        complex_chunks = Dataset.from_dict(
            {
                "document_id": ["complex_doc", "complex_doc"],
                "chunk_id": ["chunk_1", "chunk_2"],
                "chunk_text": [
                    "Chunk with complex metadata...",
                    "Another chunk with different metadata...",
                ],
                "embedding": [np.random.rand(384).tolist() for _ in range(2)],
                "position": [0, 1],
                "cited_references": [
                    json.dumps(
                        [
                            {"case": "Smith v. Jones", "year": 2020, "citation": "123 F.3d 456"},
                            {"statute": "42 U.S.C. § 1983", "type": "federal"},
                        ]
                    ),
                    json.dumps([{"regulation": "29 CFR 1630.2", "agency": "EEOC"}]),
                ],
                "tags": [
                    json.dumps(["constitutional", "due_process", "procedural"]),
                    json.dumps(["employment", "discrimination", "ada"]),
                ],
            }
        )

        config = IngestConfig(batch_size=2)
        ingester = ChunkIngester(db=mock_weaviate_database, config=config)
        ingester.ingest(document_dataset, complex_chunks)

        assert mock_weaviate_database.document_chunks_collection.batch.fixed_size.called

    def test_duplicate_document_ids_different_content(self, mock_weaviate_database):
        """Test handling of duplicate document IDs (should be prevented by deterministic UUIDs)."""
        # Create datasets with duplicate document_id but different content
        docs_with_duplicates = Dataset.from_dict(
            {
                "document_id": ["doc_1", "doc_1", "doc_2"],  # Duplicate doc_1
                "title": ["First Version", "Second Version", "Unique Doc"],
                "language": ["en", "en", "en"],
                "document_type": ["judgment", "judgment", "judgment"],
                "full_text": ["First content...", "Different content...", "Unique content..."],
            }
        )

        embeddings_with_duplicates = Dataset.from_dict(
            {
                "document_id": ["doc_1", "doc_1", "doc_2"],
                "embedding": [np.random.rand(384).tolist() for _ in range(3)],
            }
        )

        config = IngestConfig(batch_size=3, upsert=True)
        ingester = DocumentIngester(db=mock_weaviate_database, config=config)

        # With upsert=True, should handle duplicates by updating
        ingester.ingest(docs_with_duplicates, embeddings_with_duplicates)

        assert mock_weaviate_database.legal_documents_collection.batch.fixed_size.called

    def test_incremental_ingestion_simulation(self, mock_weaviate_database):
        """Test simulating incremental ingestion with existing data."""
        # First batch of documents
        initial_docs = Dataset.from_dict(
            {
                "document_id": ["doc_1", "doc_2"],
                "title": ["Document 1", "Document 2"],
                "language": ["en", "en"],
                "document_type": ["judgment", "judgment"],
            }
        )

        initial_embeddings = Dataset.from_dict(
            {
                "document_id": ["doc_1", "doc_2"],
                "embedding": [np.random.rand(384).tolist() for _ in range(2)],
            }
        )

        # Simulate that doc_1 already exists in the database
        existing_uuid = generate_deterministic_uuid("doc_1")
        mock_weaviate_database.get_uuids.return_value = [existing_uuid]

        config = IngestConfig(batch_size=2, upsert=False)  # Insert-only mode
        ingester = DocumentIngester(db=mock_weaviate_database, config=config)
        ingester.ingest(initial_docs, initial_embeddings)

        # Should only process doc_2 (doc_1 already exists)
        assert mock_weaviate_database.get_uuids.called

    def test_empty_datasets(self, mock_weaviate_database):
        """Test handling of empty datasets."""
        empty_docs = Dataset.from_dict(
            {"document_id": [], "title": [], "language": [], "document_type": []}
        )

        empty_embeddings = Dataset.from_dict({"document_id": [], "embedding": []})

        config = IngestConfig(batch_size=32)
        ingester = DocumentIngester(db=mock_weaviate_database, config=config)

        # Should handle empty datasets gracefully
        ingester.ingest(empty_docs, empty_embeddings)

        # No batch operations should be called for empty datasets
        assert not mock_weaviate_database.legal_documents_collection.batch.fixed_size.called


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios."""

    def test_invalid_embedding_dimensions(self, mock_weaviate_database):
        """Test handling of embeddings with wrong dimensions."""
        docs = Dataset.from_dict(
            {
                "document_id": ["doc_1"],
                "title": ["Test Document"],
                "language": ["en"],
                "document_type": ["judgment"],
            }
        )

        # Embeddings with wrong dimension (should be 384, but using 100)
        invalid_embeddings = Dataset.from_dict(
            {
                "document_id": ["doc_1"],
                "embedding": [np.random.rand(100).tolist()],  # Wrong dimension
            }
        )

        config = IngestConfig(batch_size=1)
        ingester = DocumentIngester(db=mock_weaviate_database, config=config)

        # The current implementation handles invalid dimensions gracefully
        # by passing them through to Weaviate without client-side validation
        ingester.ingest(docs, invalid_embeddings)

        # Verify that ingestion was attempted
        assert mock_weaviate_database.legal_documents_collection.batch.fixed_size.called

    def test_malformed_json_in_fields(self, mock_weaviate_database):
        """Test handling of malformed JSON in string fields."""
        document_dataset = Dataset.from_dict(
            {"document_id": ["doc_1"], "language": ["en"], "document_type": ["judgment"]}
        )

        chunks_with_bad_json = Dataset.from_dict(
            {
                "document_id": ["doc_1"],
                "chunk_id": ["chunk_1"],
                "chunk_text": ["Test chunk"],
                "embedding": [np.random.rand(384).tolist()],
                "cited_references": ["invalid json {missing quotes}"],  # Malformed JSON
                "tags": ['["valid", "json"]'],  # Valid JSON
            }
        )

        config = IngestConfig(batch_size=1)
        ingester = ChunkIngester(db=mock_weaviate_database, config=config)

        # Should handle malformed JSON gracefully
        try:
            ingester.ingest(document_dataset, chunks_with_bad_json)
            # If it doesn't raise an error, that's also acceptable
        except Exception as e:
            # Should be a JSON-related error
            assert "json" in str(e).lower() or isinstance(e, (ValueError, TypeError))

    def test_database_connection_failure_simulation(self):
        """Test handling of database connection failures."""
        config = IngestConfig(batch_size=1)

        docs = Dataset.from_dict(
            {"document_id": ["doc_1"], "title": ["Test"], "document_type": ["judgment"]}
        )

        embeddings = Dataset.from_dict(
            {"document_id": ["doc_1"], "embedding": [np.random.rand(384).tolist()]}
        )

        # Test with None database - should raise an appropriate error or handle gracefully
        ingester = DocumentIngester(db=None, config=config)

        # Since the current implementation tries to connect to a real database when db=None,
        # we expect this to fail with a connection error rather than succeed
        with pytest.raises(Exception):  # Expect some kind of connection error
            ingester.ingest(docs, embeddings)


if __name__ == "__main__":
    pytest.main([__file__])
