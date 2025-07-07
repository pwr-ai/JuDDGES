"""
Tests for the simplified streaming ingester.
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from juddges.data.stream_ingester import (
    ProcessedDocTracker,
    SimpleChunker,
    StreamingIngester,
    TextChunk,
)


class TestSimpleChunker:
    """Test the SimpleChunker class."""

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        chunker = SimpleChunker(chunk_size=100, overlap=20)
        text = "This is a test document. " * 10  # ~250 characters
        chunks = chunker.chunk_text(text, "doc1")

        assert len(chunks) >= 2
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert all(chunk.document_id == "doc1" for chunk in chunks)
        assert chunks[0].position == 0
        assert chunks[1].position == 1

    def test_chunk_text_short(self):
        """Test chunking very short text."""
        chunker = SimpleChunker(chunk_size=100, overlap=20, min_chunk_size=50)
        text = "Short text"
        chunks = chunker.chunk_text(text, "doc1")

        assert len(chunks) == 0  # Too short

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunker = SimpleChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_text("", "doc1")

        assert len(chunks) == 0


class TestProcessedDocTracker:
    """Test the ProcessedDocTracker class."""

    def test_tracker_initialization(self):
        """Test tracker database initialization."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        ProcessedDocTracker(db_path)

        # Check that database is created
        assert Path(db_path).exists()

        # Check that table exists
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='processed_documents'"
            )
            assert cursor.fetchone() is not None

        # Cleanup
        Path(db_path).unlink()

    def test_mark_and_check_processed(self):
        """Test marking and checking processed documents."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        tracker = ProcessedDocTracker(db_path)

        # Initially not processed
        assert not tracker.is_processed("doc1")

        # Mark as processed
        tracker.mark_processed("doc1", chunks_count=5, success=True)

        # Should be marked as processed
        assert tracker.is_processed("doc1")

        # Mark another as failed
        tracker.mark_processed("doc2", chunks_count=0, success=False)

        # Failed document should not be marked as processed
        assert not tracker.is_processed("doc2")

        # Get stats
        stats = tracker.get_stats()
        assert stats["total"] == 2
        assert stats["successful"] == 1
        assert stats["failed"] == 1

        # Cleanup
        Path(db_path).unlink()


class TestStreamingIngester:
    """Test the StreamingIngester class."""

    @patch("juddges.data.stream_ingester.weaviate.connect_to_local")
    @patch("juddges.data.stream_ingester.SentenceTransformer")
    def test_ingester_initialization(self, mock_transformer, mock_weaviate):
        """Test ingester initialization."""
        # Mock the Weaviate client
        mock_client_instance = Mock()
        mock_client_instance.collections.list_all.return_value = {}
        mock_weaviate.return_value = mock_client_instance

        # Mock the sentence transformer
        mock_transformer_instance = Mock()
        mock_transformer.return_value = mock_transformer_instance

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        ingester = StreamingIngester(
            weaviate_url="http://localhost:8080", embedding_model="test-model", tracker_db=db_path
        )

        assert ingester.weaviate_client == mock_client_instance
        assert ingester.transformers["base"] == mock_transformer_instance
        assert isinstance(ingester.chunker, SimpleChunker)
        assert isinstance(ingester.tracker, ProcessedDocTracker)

        # Cleanup
        Path(db_path).unlink()

    @patch("juddges.data.stream_ingester.weaviate.connect_to_local")
    @patch("juddges.data.stream_ingester.SentenceTransformer")
    def test_generate_embeddings(self, mock_transformer, mock_weaviate):
        """Test embedding generation."""
        # Mock the Weaviate client
        mock_client_instance = Mock()
        mock_client_instance.collections.list_all.return_value = {}
        mock_weaviate.return_value = mock_client_instance

        # Mock the sentence transformer
        mock_transformer_instance = Mock()
        mock_transformer.return_value = mock_transformer_instance

        # Mock embedding generation
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_transformer_instance.encode.return_value = mock_embeddings

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        ingester = StreamingIngester(
            weaviate_url="http://localhost:8080", embedding_model="test-model", tracker_db=db_path
        )

        texts = ["text1", "text2"]
        embeddings = ingester._generate_embeddings(texts)

        # Should return embeddings for each vector model
        assert "base" in embeddings
        assert embeddings["base"] == mock_embeddings.tolist()
        mock_transformer_instance.encode.assert_called()

        # Cleanup
        Path(db_path).unlink()

    @patch("juddges.data.stream_ingester.weaviate.connect_to_local")
    @patch("juddges.data.stream_ingester.SentenceTransformer")
    def test_aggregate_embeddings(self, mock_transformer, mock_weaviate):
        """Test embedding aggregation."""
        # Mock the Weaviate client
        mock_client_instance = Mock()
        mock_client_instance.collections.list_all.return_value = {}
        mock_weaviate.return_value = mock_client_instance

        # Mock the sentence transformer
        mock_transformer_instance = Mock()
        mock_transformer_instance.get_sentence_embedding_dimension.return_value = 768
        mock_transformer.return_value = mock_transformer_instance

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        ingester = StreamingIngester(
            weaviate_url="http://localhost:8080", embedding_model="test-model", tracker_db=db_path
        )

        embeddings_dict = {"base": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
        aggregated = ingester._aggregate_embeddings(embeddings_dict)

        # Should be mean of embeddings for base vector
        expected = {"base": [2.5, 3.5, 4.5]}
        assert aggregated == expected

        # Test empty embeddings
        empty_aggregated = ingester._aggregate_embeddings({"base": []})
        assert len(empty_aggregated["base"]) == 768  # Default size

        # Cleanup
        Path(db_path).unlink()

    def test_generate_uuid(self):
        """Test UUID generation."""
        # This test doesn't need mocking as it's a pure function
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        with (
            patch("juddges.data.stream_ingester.weaviate.connect_to_local") as mock_weaviate,
            patch("juddges.data.stream_ingester.SentenceTransformer"),
        ):
            # Mock the weaviate client
            mock_client = Mock()
            mock_client.collections.list_all.return_value = {}
            mock_weaviate.return_value = mock_client

            ingester = StreamingIngester(tracker_db=db_path)

            # Test deterministic UUID generation
            uuid1 = ingester._generate_uuid("test_id", "test_text")
            uuid2 = ingester._generate_uuid("test_id", "test_text")
            uuid3 = ingester._generate_uuid("different_id", "different_text")

            # Same input should produce same UUID
            assert uuid1 == uuid2

            # Different input should produce different UUID
            assert uuid1 != uuid3

            # Should be valid UUID format
            assert len(uuid1) == 36
            assert uuid1.count("-") == 4

        # Cleanup
        Path(db_path).unlink()


# Integration test (requires actual Weaviate instance)
@pytest.mark.integration
class TestStreamingIngesterIntegration:
    """Integration tests for StreamingIngester (requires Weaviate)."""

    def test_full_integration(self):
        """Test full integration with mock dataset."""
        # This would require a running Weaviate instance
        # Skip in unit tests, run separately for integration testing
        pytest.skip("Integration test requires running Weaviate instance")
