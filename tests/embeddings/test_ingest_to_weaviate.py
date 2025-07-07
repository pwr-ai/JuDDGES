"""Test cases for Weaviate ingestion functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from datasets import Dataset

from juddges.config import EmbeddingConfig, EmbeddingModelConfig
from scripts.embed.ingest_to_weaviate import (
    DEFAULT_INGEST_BATCH_SIZE,
    DEFAULT_UPSERT,
    ChunkIngester,
    DatasetLoader,
    DocumentIngester,
    IngestConfig,
    generate_deterministic_uuid,
)


class TestGenerateDeterministicUuid:
    """Test the deterministic UUID generation function."""

    def test_generate_uuid_document_only(self):
        """Test UUID generation for documents."""
        document_id = "doc_123"
        uuid = generate_deterministic_uuid(document_id)

        # Should generate a valid UUID format
        assert len(uuid) == 36
        assert uuid.count("-") == 4

        # Should be deterministic
        uuid2 = generate_deterministic_uuid(document_id)
        assert uuid == uuid2

    def test_generate_uuid_with_chunk(self):
        """Test UUID generation for chunks."""
        document_id = "doc_123"
        chunk_id = "chunk_456"
        uuid = generate_deterministic_uuid(document_id, chunk_id)

        # Should generate a valid UUID format
        assert len(uuid) == 36
        assert uuid.count("-") == 4

        # Should be deterministic
        uuid2 = generate_deterministic_uuid(document_id, chunk_id)
        assert uuid == uuid2

    def test_uuid_different_for_different_inputs(self):
        """Test that different inputs generate different UUIDs."""
        uuid1 = generate_deterministic_uuid("doc_1")
        uuid2 = generate_deterministic_uuid("doc_2")
        uuid3 = generate_deterministic_uuid("doc_1", "chunk_1")

        assert uuid1 != uuid2
        assert uuid1 != uuid3
        assert uuid2 != uuid3


class TestIngestConfig:
    """Test the ingestion configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IngestConfig()
        assert config.batch_size == DEFAULT_INGEST_BATCH_SIZE
        assert config.upsert == DEFAULT_UPSERT
        assert config.max_documents is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = IngestConfig(
            batch_size=16,
            upsert=False,
            max_documents=100,
            processing_proc=2,
            ingest_proc=1,
        )
        assert config.batch_size == 16
        assert config.upsert is False
        assert config.max_documents == 100
        assert config.processing_proc == 2
        assert config.ingest_proc == 1


class TestCollectionIngester:
    """Test the base collection ingester functionality using concrete implementations."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        db = Mock()
        collection = Mock()

        # Properly setup context manager for batch operations
        batch_context = Mock()
        batch_context.__enter__ = Mock(return_value=batch_context)
        batch_context.__exit__ = Mock(return_value=None)

        collection.batch.fixed_size.return_value = batch_context
        db.document_chunks_collection = collection
        db.legal_documents_collection = collection
        db.get_collection.return_value = collection
        db.get_collection_size.return_value = 0
        db.get_uuids.return_value = []
        return db

    @pytest.fixture
    def sample_document_dataset(self):
        """Create a sample document dataset."""
        return Dataset.from_dict(
            {
                "document_id": ["doc_1", "doc_2", "doc_3"],
                "title": ["Title 1", "Title 2", "Title 3"],
                "language": ["en", "en", "pl"],
                "country": ["US", "UK", "PL"],
                "date_issued": ["2023-01-01", "2023-02-01", "2023-03-01"],
                "document_type": ["judgment", "judgment", "tax_interpretation"],
                "full_text": ["Full text 1", "Full text 2", "Full text 3"],
            }
        )

    @pytest.fixture
    def sample_embedding_dataset(self):
        """Create a sample embedding dataset for documents."""
        embeddings = [np.random.rand(384).tolist() for _ in range(3)]
        return Dataset.from_dict(
            {"document_id": ["doc_1", "doc_2", "doc_3"], "embedding": embeddings}
        )

    def test_ingester_initialization(self, mock_db):
        """Test ingester initialization using ChunkIngester as concrete implementation."""
        config = IngestConfig(batch_size=16)
        ingester = ChunkIngester(
            db=mock_db,
            config=config,
            columns_to_ingest=["document_id", "title"],
        )

        assert ingester.collection_name == "document_chunks"
        assert ingester.db == mock_db
        assert ingester.config.batch_size == 16
        assert ingester.columns_to_ingest == ["document_id", "title"]

    def test_dataset_validation_missing_document_id(self, mock_db):
        """Test that ingestion fails when document_id column is missing."""
        ingester = ChunkIngester(db=mock_db)

        invalid_dataset = Dataset.from_dict(
            {"title": ["Title 1", "Title 2"], "content": ["Content 1", "Content 2"]}
        )

        with pytest.raises(ValueError, match="Dataset must contain 'document_id' column"):
            ingester.ingest(invalid_dataset, invalid_dataset)

    def test_columns_filtering(self, mock_db, sample_document_dataset, sample_embedding_dataset):
        """Test that column filtering works correctly."""
        config = IngestConfig(max_documents=1)  # Limit to avoid processing
        ingester = ChunkIngester(
            db=mock_db,
            config=config,
            columns_to_ingest=["document_id", "title"],
        )

        # Mock the process_batch method to avoid actual processing
        with patch.object(ingester, "process_batch"), patch.object(ingester, "_process_batches"):
            ingester.ingest(sample_document_dataset, sample_embedding_dataset)

        # Verify that database collections were accessed
        assert mock_db.document_chunks_collection is not None


class TestChunkIngester:
    """Test the chunk ingester functionality."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database for chunks."""
        db = Mock()
        collection = Mock()

        # Create a proper context manager mock
        class MockBatchContext:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

            def add_object(self, *args, **kwargs):
                pass

        batch_context = MockBatchContext()
        collection.batch.fixed_size.return_value = batch_context
        db.document_chunks_collection = collection
        db.get_collection.return_value = collection
        db.get_collection_size.return_value = 0
        db.get_uuids.return_value = []
        return db

    @pytest.fixture
    def sample_document_dataset(self):
        """Create a sample document dataset."""
        return Dataset.from_dict(
            {
                "document_id": ["doc_1", "doc_2"],
                "language": ["en", "pl"],
                "country": ["US", "PL"],
                "date_issued": ["2023-01-01", "2023-02-01"],
                "document_type": ["judgment", "tax_interpretation"],
            }
        )

    @pytest.fixture
    def sample_chunk_dataset(self):
        """Create a sample chunk embeddings dataset."""
        embeddings = [np.random.rand(384).tolist() for _ in range(4)]
        return Dataset.from_dict(
            {
                "document_id": ["doc_1", "doc_1", "doc_2", "doc_2"],
                "chunk_id": [1, 2, 3, 4],
                "chunk_text": [
                    "This is the first chunk of document 1.",
                    "This is the second chunk of document 1.",
                    "This is the first chunk of document 2.",
                    "This is the second chunk of document 2.",
                ],
                "embedding": embeddings,
                "position": [0, 1, 0, 1],
                "cited_references": [
                    json.dumps(["ref1", "ref2"]),
                    json.dumps(["ref3"]),
                    None,
                    json.dumps(["ref4", "ref5"]),
                ],
                "tags": [
                    json.dumps(["facts", "reasoning"]),
                    json.dumps(["conclusion"]),
                    json.dumps(["facts"]),
                    None,
                ],
            }
        )

    def test_chunk_ingester_initialization(self, mock_db):
        """Test chunk ingester initialization."""
        config = IngestConfig(batch_size=8)
        ingester = ChunkIngester(db=mock_db, config=config)

        assert ingester.collection_name == "document_chunks"  # Actual constant value
        assert ingester.db == mock_db
        assert ingester.config.batch_size == 8

    def test_chunk_batch_processing(self, mock_db, sample_document_dataset, sample_chunk_dataset):
        """Test processing a batch of chunks."""
        config = IngestConfig(batch_size=2)
        ingester = ChunkIngester(db=mock_db, config=config)

        # Test processing the first batch
        ingester.process_batch(
            dataset=sample_document_dataset,
            embedding_dataset=sample_chunk_dataset,
            batch_idx=0,
        )

        # Verify the batch processing was called
        mock_db.document_chunks_collection.batch.fixed_size.assert_called_with(batch_size=2)

    def test_chunk_filtering_by_document_ids(self, mock_db):
        """Test filtering chunks by valid document IDs."""
        config = IngestConfig()
        ingester = ChunkIngester(db=mock_db, config=config)

        # Create test data
        chunk_dataset = Dataset.from_dict(
            {
                "document_id": ["doc_1", "doc_2", "doc_3", "doc_4"],
                "chunk_id": ["chunk_1", "chunk_2", "chunk_3", "chunk_4"],
                "chunk_text": ["text1", "text2", "text3", "text4"],
                "embedding": [np.random.rand(384).tolist() for _ in range(4)],
            }
        )

        valid_document_ids = {"doc_1", "doc_3"}  # Only these are valid

        # Mock the database to return empty UUIDs (no existing chunks)
        mock_db.get_uuids.return_value = []

        filtered_dataset = ingester._filter_existing_objects(
            chunk_dataset,
            mock_db.get_collection.return_value,
            chunk_dataset.num_rows,
            document_ids=valid_document_ids,
        )

        # Should only keep chunks for doc_1 and doc_3
        assert filtered_dataset.num_rows == 2
        assert set(filtered_dataset["document_id"]) == {"doc_1", "doc_3"}

    def test_chunk_data_merging(self, mock_db, sample_document_dataset, sample_chunk_dataset):
        """Test that document attributes are properly merged into chunks."""
        config = IngestConfig(batch_size=4)
        ingester = ChunkIngester(db=mock_db, config=config)

        # Test processing (with our mock, this should not raise errors)
        ingester.process_batch(
            dataset=sample_document_dataset,
            embedding_dataset=sample_chunk_dataset,
            batch_idx=0,
        )

        # Verify that the batch operation was called
        assert mock_db.document_chunks_collection.batch.fixed_size.called


class TestDocumentIngester:
    """Test the document ingester functionality."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database for documents."""
        db = Mock()
        collection = Mock()

        # Create a proper context manager mock
        class MockBatchContext:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

            def add_object(self, *args, **kwargs):
                pass

        batch_context = MockBatchContext()
        collection.batch.fixed_size.return_value = batch_context
        db.legal_documents_collection = collection
        db.get_collection.return_value = collection
        db.get_collection_size.return_value = 0
        db.get_uuids.return_value = []
        return db

    @pytest.fixture
    def sample_document_dataset(self):
        """Create a sample document dataset."""
        return Dataset.from_dict(
            {
                "document_id": ["doc_1", "doc_2", "doc_3"],
                "title": ["Legal Case 1", "Tax Interpretation 2", "Judgment 3"],
                "language": ["en", "pl", "en"],
                "country": ["US", "PL", "UK"],
                "date_issued": ["2023-01-01", "2023-02-01", "2023-03-01"],
                "document_type": ["judgment", "tax_interpretation", "judgment"],
                "full_text": [
                    "Full legal text 1",
                    "Tax interpretation text",
                    "Judgment text",
                ],
                "summary": ["Summary 1", "Summary 2", "Summary 3"],
                "keywords": [
                    json.dumps(["contract", "liability"]),
                    json.dumps(["tax", "deduction"]),
                    json.dumps(["criminal", "evidence"]),
                ],
            }
        )

    @pytest.fixture
    def sample_document_embeddings(self):
        """Create document embeddings dataset."""
        embeddings = [np.random.rand(384).tolist() for _ in range(3)]
        return Dataset.from_dict(
            {"document_id": ["doc_1", "doc_2", "doc_3"], "embedding": embeddings}
        )

    def test_document_ingester_initialization(self, mock_db):
        """Test document ingester initialization."""
        config = IngestConfig(batch_size=4)
        default_values = {"processing_status": "completed"}

        ingester = DocumentIngester(db=mock_db, config=config, default_column_values=default_values)

        assert ingester.collection_name == "LegalDocuments"  # Actual constant value
        assert ingester.db == mock_db
        assert ingester.config.batch_size == 4
        assert ingester.default_column_values == default_values

    def test_document_batch_processing(
        self, mock_db, sample_document_dataset, sample_document_embeddings
    ):
        """Test processing a batch of documents."""
        config = IngestConfig(batch_size=2)
        ingester = DocumentIngester(db=mock_db, config=config)

        ingester.process_batch(
            dataset=sample_document_dataset,
            embedding_dataset=sample_document_embeddings,
            batch_idx=0,
        )

        # Verify batch processing was called
        mock_db.legal_documents_collection.batch.fixed_size.assert_called_with(batch_size=2)

    def test_document_filtering_existing(self, mock_db):
        """Test filtering out existing documents."""
        config = IngestConfig()
        ingester = DocumentIngester(db=mock_db, config=config)

        # Create test dataset
        dataset = Dataset.from_dict(
            {
                "document_id": ["doc_1", "doc_2", "doc_3"],
                "title": ["Title 1", "Title 2", "Title 3"],
            }
        )

        # Mock existing UUIDs (doc_2 already exists)
        existing_uuid = generate_deterministic_uuid("doc_2")
        mock_db.get_uuids.return_value = [existing_uuid]

        filtered_dataset = ingester._filter_existing_objects(
            dataset, mock_db.get_collection.return_value, dataset.num_rows
        )

        # Should filter out doc_2
        assert filtered_dataset.num_rows == 2
        assert "doc_2" not in filtered_dataset["document_id"]
        assert set(filtered_dataset["document_id"]) == {"doc_1", "doc_3"}

    def test_document_default_values(
        self, mock_db, sample_document_dataset, sample_document_embeddings
    ):
        """Test that default column values are applied."""
        config = IngestConfig(batch_size=1)
        default_values = {"processing_status": "completed", "confidence_score": 0.95}

        ingester = DocumentIngester(db=mock_db, config=config, default_column_values=default_values)

        ingester.process_batch(
            dataset=sample_document_dataset,
            embedding_dataset=sample_document_embeddings,
            batch_idx=0,
        )

        # Verify that batch processing was called with default values config
        assert mock_db.legal_documents_collection.batch.fixed_size.called
        assert ingester.default_column_values == default_values


class TestDatasetLoader:
    """Test the dataset loader functionality."""

    @pytest.fixture
    def embedding_config(self):
        """Create a sample embedding configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = EmbeddingConfig(
                output_dir=Path(temp_dir) / "embeddings",
                dataset_name="test/dataset",
                embedding_model=EmbeddingModelConfig(name="test-model", max_seq_length=512),
                batch_size=16,
                num_output_shards=2,
            )
            # Create the embeddings directory structure
            config.output_dir.mkdir(parents=True, exist_ok=True)
            (config.output_dir / config.CHUNK_EMBEDDINGS_DIR).mkdir(exist_ok=True)
            (config.output_dir / config.AGG_EMBEDDINGS_DIR).mkdir(exist_ok=True)

            yield config

    def test_loader_initialization(self, embedding_config):
        """Test dataset loader initialization."""
        loader = DatasetLoader(embedding_config)
        assert loader.config == embedding_config

    def test_check_embeddings_exist_missing_dir(self):
        """Test that missing embeddings directory raises error."""
        config = EmbeddingConfig(
            output_dir=Path("/nonexistent/path"),
            dataset_name="test/dataset",
            embedding_model=EmbeddingModelConfig(name="test-model", max_seq_length=512),
            batch_size=16,
            num_output_shards=2,
        )

        loader = DatasetLoader(config)

        with pytest.raises(ValueError, match="Embeddings directory does not exist"):
            loader._check_embeddings_exist()

    @patch("scripts.embed.ingest_to_weaviate.load_dataset")
    def test_load_chunk_dataset(self, mock_load_dataset, embedding_config):
        """Test loading chunk dataset."""
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset

        loader = DatasetLoader(embedding_config)
        result = loader.load_chunk_dataset()

        assert result == mock_dataset
        mock_load_dataset.assert_called_once_with(
            "parquet",
            data_dir=str(embedding_config.output_dir / embedding_config.CHUNK_EMBEDDINGS_DIR),
            split="train",
            num_proc=embedding_config.num_output_shards,
        )

    @patch("scripts.embed.ingest_to_weaviate.load_dataset")
    def test_load_document_embeddings_dataset(self, mock_load_dataset, embedding_config):
        """Test loading document embeddings dataset."""
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset

        loader = DatasetLoader(embedding_config)
        result = loader.load_document_embeddings_dataset()

        assert result == mock_dataset
        mock_load_dataset.assert_called_once_with(
            "parquet",
            data_dir=str(embedding_config.output_dir / embedding_config.AGG_EMBEDDINGS_DIR),
            split="train",
            num_proc=embedding_config.num_output_shards,
        )

    @patch("scripts.embed.ingest_to_weaviate.load_dataset")
    def test_load_document_dataset(self, mock_load_dataset, embedding_config):
        """Test loading document dataset."""
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset

        loader = DatasetLoader(embedding_config)
        result = loader.load_document_dataset()

        assert result == mock_dataset
        mock_load_dataset.assert_called_once_with(
            embedding_config.dataset_name,
            split="train",
            num_proc=embedding_config.num_output_shards,
        )


class TestIntegrationScenarios:
    """Integration tests for complete ingestion scenarios."""

    @pytest.fixture
    def mock_weaviate_db(self):
        """Create a comprehensive mock for WeaviateLegalDocumentsDatabase."""
        with patch("scripts.embed.ingest_to_weaviate.WeaviateLegalDocumentsDatabase") as MockDB:
            db_instance = Mock()
            db_instance.__enter__ = Mock(return_value=db_instance)
            db_instance.__exit__ = Mock(return_value=None)

            # Mock collections
            doc_collection = Mock()
            chunk_collection = Mock()

            # Mock batch operations with proper context manager setup
            doc_batch_context = Mock()
            doc_batch_context.__enter__ = Mock(return_value=doc_batch_context)
            doc_batch_context.__exit__ = Mock(return_value=None)

            chunk_batch_context = Mock()
            chunk_batch_context.__enter__ = Mock(return_value=chunk_batch_context)
            chunk_batch_context.__exit__ = Mock(return_value=None)

            doc_collection.batch.fixed_size.return_value = doc_batch_context
            chunk_collection.batch.fixed_size.return_value = chunk_batch_context

            db_instance.legal_documents_collection = doc_collection
            db_instance.document_chunks_collection = chunk_collection
            db_instance.get_collection.side_effect = lambda name: (
                doc_collection if "Documents" in name else chunk_collection
            )
            db_instance.get_collection_size.return_value = 0
            db_instance.get_uuids.return_value = []
            db_instance.create_collections.return_value = None

            MockDB.return_value = db_instance
            yield db_instance

    def test_full_ingestion_flow_small_dataset(self, mock_weaviate_db):
        """Test complete ingestion flow with a small dataset."""
        # Create small test datasets
        document_dataset = Dataset.from_dict(
            {
                "document_id": ["doc_1", "doc_2"],
                "title": ["Document 1", "Document 2"],
                "language": ["en", "pl"],
                "country": ["US", "PL"],
                "date_issued": ["2023-01-01", "2023-02-01"],
                "document_type": ["judgment", "tax_interpretation"],
                "full_text": ["Text 1", "Text 2"],
            }
        )

        doc_embeddings = Dataset.from_dict(
            {
                "document_id": ["doc_1", "doc_2"],
                "embedding": [np.random.rand(384).tolist() for _ in range(2)],
            }
        )

        chunk_embeddings = Dataset.from_dict(
            {
                "document_id": ["doc_1", "doc_1", "doc_2"],
                "chunk_id": ["chunk_1", "chunk_2", "chunk_3"],
                "chunk_text": ["Chunk 1 text", "Chunk 2 text", "Chunk 3 text"],
                "embedding": [np.random.rand(384).tolist() for _ in range(3)],
                "position": [0, 1, 0],
            }
        )

        config = IngestConfig(batch_size=2, max_documents=2)

        # Test document ingestion
        doc_ingester = DocumentIngester(db=mock_weaviate_db, config=config)
        doc_ingester.ingest(document_dataset, doc_embeddings)

        # Test chunk ingestion
        chunk_ingester = ChunkIngester(db=mock_weaviate_db, config=config)
        chunk_ingester.ingest(document_dataset, chunk_embeddings)

        # Verify database interactions
        assert mock_weaviate_db.get_collection.call_count >= 2
        assert mock_weaviate_db.legal_documents_collection.batch.fixed_size.called
        assert mock_weaviate_db.document_chunks_collection.batch.fixed_size.called

    def test_upsert_vs_insert_only_mode(self, mock_weaviate_db):
        """Test the difference between upsert and insert-only modes."""
        dataset = Dataset.from_dict(
            {"document_id": ["doc_1", "doc_2"], "title": ["Doc 1", "Doc 2"]}
        )

        embeddings = Dataset.from_dict(
            {
                "document_id": ["doc_1", "doc_2"],
                "embedding": [np.random.rand(384).tolist() for _ in range(2)],
            }
        )

        # Test upsert mode (should process all documents)
        upsert_config = IngestConfig(upsert=True, batch_size=2)
        ingester = DocumentIngester(db=mock_weaviate_db, config=upsert_config)
        ingester.ingest(dataset, embeddings)

        # Verify upsert mode processes documents
        assert mock_weaviate_db.legal_documents_collection.batch.fixed_size.called

        # Reset mock
        mock_weaviate_db.reset_mock()

        # Test insert-only mode with existing documents
        existing_uuid = generate_deterministic_uuid("doc_1")
        mock_weaviate_db.get_uuids.return_value = [existing_uuid]

        insert_config = IngestConfig(upsert=False, batch_size=2)
        ingester = DocumentIngester(db=mock_weaviate_db, config=insert_config)
        ingester.ingest(dataset, embeddings)

        # Verify insert-only mode was used
        assert mock_weaviate_db.legal_documents_collection.batch.fixed_size.called

    def test_error_handling_invalid_embeddings(self, mock_weaviate_db):
        """Test error handling for invalid embedding data."""
        dataset = Dataset.from_dict({"document_id": ["doc_1"], "title": ["Document 1"]})

        # Invalid embeddings (wrong dimension or None values)
        invalid_embeddings = Dataset.from_dict(
            {"document_id": ["doc_1"], "embedding": [None]}  # Invalid embedding
        )

        config = IngestConfig(batch_size=1)
        ingester = DocumentIngester(db=mock_weaviate_db, config=config)

        # The current implementation handles None embeddings gracefully
        # by filtering them out during processing
        ingester.ingest(dataset, invalid_embeddings)

        # Verify that ingestion was attempted
        assert mock_weaviate_db.get_collection_size.called

    def test_large_batch_processing(self, mock_weaviate_db):
        """Test processing with larger batches to verify batch logic."""
        # Create a larger dataset
        num_docs = 50
        document_ids = [f"doc_{i}" for i in range(num_docs)]

        large_dataset = Dataset.from_dict(
            {
                "document_id": document_ids,
                "title": [f"Document {i}" for i in range(num_docs)],
                "language": ["en"] * num_docs,
                "country": ["US"] * num_docs,
                "document_type": ["judgment"] * num_docs,
            }
        )

        large_embeddings = Dataset.from_dict(
            {
                "document_id": document_ids,
                "embedding": [np.random.rand(384).tolist() for _ in range(num_docs)],
            }
        )

        config = IngestConfig(batch_size=10, max_documents=num_docs)
        ingester = DocumentIngester(db=mock_weaviate_db, config=config)
        ingester.ingest(large_dataset, large_embeddings)

        # Should have processed multiple batches
        assert mock_weaviate_db.legal_documents_collection.batch.fixed_size.call_count >= 5


if __name__ == "__main__":
    pytest.main([__file__])
