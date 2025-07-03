"""Test cases for the DatasetLoader class."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from datasets import Dataset

from juddges.config import EmbeddingConfig, EmbeddingModelConfig
from scripts.embed.ingest_to_weaviate import DatasetLoader


class TestDatasetLoader:
    """Test the DatasetLoader functionality."""

    @pytest.fixture
    def embedding_config_with_files(self):
        """Create embedding config with actual parquet files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = EmbeddingConfig(
                output_dir=Path(temp_dir) / "embeddings",
                dataset_name="test/legal-dataset",
                embedding_model=EmbeddingModelConfig(
                    name="sentence-transformers/all-MiniLM-L6-v2", max_seq_length=384
                ),
                batch_size=16,
                num_output_shards=1,
            )

            # Create directory structure
            config.output_dir.mkdir(parents=True, exist_ok=True)
            chunks_dir = config.output_dir / config.CHUNK_EMBEDDINGS_DIR
            agg_dir = config.output_dir / config.AGG_EMBEDDINGS_DIR
            chunks_dir.mkdir(exist_ok=True)
            agg_dir.mkdir(exist_ok=True)

            # Create sample chunk embeddings
            chunk_data = Dataset.from_dict(
                {
                    "document_id": ["doc_1", "doc_2", "doc_3"],
                    "chunk_id": ["chunk_1", "chunk_2", "chunk_3"],
                    "chunk_text": [
                        "First chunk of legal document...",
                        "Second chunk with legal reasoning...",
                        "Third chunk with conclusion...",
                    ],
                    "embedding": [np.random.rand(384).tolist() for _ in range(3)],
                    "position": [0, 1, 2],
                }
            )
            chunk_data.to_parquet(str(chunks_dir / "data.parquet"))

            # Create sample aggregated document embeddings
            agg_data = Dataset.from_dict(
                {
                    "document_id": ["doc_1", "doc_2"],
                    "embedding": [np.random.rand(384).tolist() for _ in range(2)],
                }
            )
            agg_data.to_parquet(str(agg_dir / "data.parquet"))

            yield config

    def test_loader_initialization(self, sample_embedding_config):
        """Test DatasetLoader initialization."""
        loader = DatasetLoader(sample_embedding_config)
        assert loader.config == sample_embedding_config

    def test_check_embeddings_exist_valid_path(self, embedding_config_with_files):
        """Test checking embeddings directory when it exists."""
        loader = DatasetLoader(embedding_config_with_files)
        # Should not raise an exception
        loader._check_embeddings_exist()

    def test_check_embeddings_exist_missing_path(self):
        """Test checking embeddings directory when it doesn't exist."""
        config = EmbeddingConfig(
            output_dir=Path("/nonexistent/path/embeddings"),
            dataset_name="test/dataset",
            embedding_model=EmbeddingModelConfig(name="test-model", max_seq_length=512),
            batch_size=16,
            num_output_shards=2,
        )

        loader = DatasetLoader(config)

        with pytest.raises(ValueError, match="Embeddings directory does not exist"):
            loader._check_embeddings_exist()

    @patch("scripts.embed.ingest_to_weaviate.load_dataset")
    def test_load_chunk_dataset_success(self, mock_load_dataset, embedding_config_with_files):
        """Test successful loading of chunk dataset."""
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset

        loader = DatasetLoader(embedding_config_with_files)
        result = loader.load_chunk_dataset()

        assert result == mock_dataset
        mock_load_dataset.assert_called_once_with(
            "parquet",
            data_dir=str(
                embedding_config_with_files.output_dir
                / embedding_config_with_files.CHUNK_EMBEDDINGS_DIR
            ),
            split="train",
            num_proc=embedding_config_with_files.num_output_shards,
        )

    @patch("scripts.embed.ingest_to_weaviate.load_dataset")
    def test_load_document_embeddings_dataset_success(
        self, mock_load_dataset, embedding_config_with_files
    ):
        """Test successful loading of document embeddings dataset."""
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset

        loader = DatasetLoader(embedding_config_with_files)
        result = loader.load_document_embeddings_dataset()

        assert result == mock_dataset
        mock_load_dataset.assert_called_once_with(
            "parquet",
            data_dir=str(
                embedding_config_with_files.output_dir
                / embedding_config_with_files.AGG_EMBEDDINGS_DIR
            ),
            split="train",
            num_proc=embedding_config_with_files.num_output_shards,
        )

    @patch("scripts.embed.ingest_to_weaviate.load_dataset")
    def test_load_document_dataset_success(self, mock_load_dataset, embedding_config_with_files):
        """Test successful loading of document dataset."""
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset

        loader = DatasetLoader(embedding_config_with_files)
        result = loader.load_document_dataset()

        assert result == mock_dataset
        mock_load_dataset.assert_called_once_with(
            embedding_config_with_files.dataset_name,
            split="train",
            num_proc=embedding_config_with_files.num_output_shards,
        )

    def test_load_all_datasets_integration(self, embedding_config_with_files):
        """Test loading all three types of datasets in sequence."""
        loader = DatasetLoader(embedding_config_with_files)

        with patch("scripts.embed.ingest_to_weaviate.load_dataset") as mock_load:
            # Mock returns for different dataset types
            chunk_dataset = Mock()
            chunk_dataset.num_rows = 3

            doc_embeddings = Mock()
            doc_embeddings.num_rows = 2

            doc_dataset = Mock()
            doc_dataset.num_rows = 2

            mock_load.side_effect = [chunk_dataset, doc_embeddings, doc_dataset]

            # Load all datasets
            chunks = loader.load_chunk_dataset()
            doc_embs = loader.load_document_embeddings_dataset()
            docs = loader.load_document_dataset()

            # Verify correct datasets returned
            assert chunks == chunk_dataset
            assert doc_embs == doc_embeddings
            assert docs == doc_dataset

            # Verify correct number of calls
            assert mock_load.call_count == 3

    def test_different_shard_configurations(self, embedding_config_with_files):
        """Test loading with different num_output_shards configurations."""
        # Test with single shard
        embedding_config_with_files.num_output_shards = 1
        loader = DatasetLoader(embedding_config_with_files)

        with patch("scripts.embed.ingest_to_weaviate.load_dataset") as mock_load:
            mock_load.return_value = Mock()
            loader.load_chunk_dataset()

            # Verify num_proc parameter
            call_args = mock_load.call_args
            assert call_args[1]["num_proc"] == 1

        # Test with multiple shards
        embedding_config_with_files.num_output_shards = 4

        with patch("scripts.embed.ingest_to_weaviate.load_dataset") as mock_load:
            mock_load.return_value = Mock()
            loader.load_document_dataset()

            # Verify num_proc parameter
            call_args = mock_load.call_args
            assert call_args[1]["num_proc"] == 4

    @patch("scripts.embed.ingest_to_weaviate.load_dataset")
    def test_load_dataset_with_exception(self, mock_load_dataset, embedding_config_with_files):
        """Test handling of exceptions during dataset loading."""
        mock_load_dataset.side_effect = Exception("Failed to load dataset")

        loader = DatasetLoader(embedding_config_with_files)

        with pytest.raises(Exception, match="Failed to load dataset"):
            loader.load_chunk_dataset()

    def test_path_construction(self, embedding_config_with_files):
        """Test that dataset paths are constructed correctly."""
        loader = DatasetLoader(embedding_config_with_files)

        # Check that paths are built correctly
        expected_chunk_path = (
            embedding_config_with_files.output_dir
            / embedding_config_with_files.CHUNK_EMBEDDINGS_DIR
        )
        expected_agg_path = (
            embedding_config_with_files.output_dir / embedding_config_with_files.AGG_EMBEDDINGS_DIR
        )

        with patch("scripts.embed.ingest_to_weaviate.load_dataset") as mock_load:
            mock_load.return_value = Mock()

            # Test chunk dataset path
            loader.load_chunk_dataset()
            chunk_call_args = mock_load.call_args
            assert chunk_call_args[1]["data_dir"] == str(expected_chunk_path)

            # Test aggregated embeddings path
            loader.load_document_embeddings_dataset()
            agg_call_args = mock_load.call_args
            assert agg_call_args[1]["data_dir"] == str(expected_agg_path)


class TestDatasetLoaderWithRealFiles:
    """Test DatasetLoader with actual parquet files (slower tests)."""

    def test_real_file_loading(self):
        """Test loading actual parquet files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = EmbeddingConfig(
                output_dir=Path(temp_dir) / "embeddings",
                dataset_name="test/dataset",
                embedding_model=EmbeddingModelConfig(name="test-model", max_seq_length=384),
                batch_size=16,
                num_output_shards=1,
            )

            # Create directories
            config.output_dir.mkdir(parents=True, exist_ok=True)
            chunks_dir = config.output_dir / config.CHUNK_EMBEDDINGS_DIR
            agg_dir = config.output_dir / config.AGG_EMBEDDINGS_DIR
            chunks_dir.mkdir(exist_ok=True)
            agg_dir.mkdir(exist_ok=True)

            # Create real chunk data
            chunk_data = Dataset.from_dict(
                {
                    "document_id": ["legal_doc_1", "legal_doc_2"],
                    "chunk_id": ["chunk_1", "chunk_2"],
                    "chunk_text": [
                        "This case involves a contract dispute between...",
                        "The court finds that the defendant breached...",
                    ],
                    "embedding": [np.random.rand(384).tolist(), np.random.rand(384).tolist()],
                    "position": [0, 1],
                    "cited_references": [
                        '["Smith v. Jones (2020)", "Contract Law § 123"]',
                        '["Brown v. Green (2019)", "Civil Code § 456"]',
                    ],
                }
            )

            # Create real aggregated data
            agg_data = Dataset.from_dict(
                {
                    "document_id": ["legal_doc_1", "legal_doc_2"],
                    "embedding": [np.random.rand(384).tolist(), np.random.rand(384).tolist()],
                }
            )

            # Save to parquet
            chunk_data.to_parquet(str(chunks_dir / "chunks.parquet"))
            agg_data.to_parquet(str(agg_dir / "documents.parquet"))

            # Test loading
            loader = DatasetLoader(config)

            # This will try to load actual files, but we'll mock the HuggingFace dataset name
            with patch("scripts.embed.ingest_to_weaviate.load_dataset") as mock_load:
                # Mock only the document dataset call (which uses dataset_name)
                def side_effect(*args, **kwargs):
                    if args[0] == "test/dataset":
                        # Return mock for HuggingFace dataset
                        return Mock()
                    else:
                        # For parquet files, call the real function
                        from datasets import load_dataset

                        return load_dataset(*args, **kwargs)

                mock_load.side_effect = side_effect

                # Load chunk dataset (real parquet file)
                chunks = loader.load_chunk_dataset()
                assert chunks.num_rows == 2
                assert "document_id" in chunks.column_names
                assert "chunk_text" in chunks.column_names
                assert "embedding" in chunks.column_names

                # Load document embeddings (real parquet file)
                doc_embeddings = loader.load_document_embeddings_dataset()
                assert doc_embeddings.num_rows == 2
                assert "document_id" in doc_embeddings.column_names
                assert "embedding" in doc_embeddings.column_names

                # Load document dataset (mocked HuggingFace dataset)
                docs = loader.load_document_dataset()
                assert docs is not None  # Mock object


if __name__ == "__main__":
    pytest.main([__file__])
