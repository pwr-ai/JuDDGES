"""Shared test fixtures for Weaviate ingestion tests."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from datasets import Dataset

from juddges.config import EmbeddingConfig, EmbeddingModelConfig


@pytest.fixture
def sample_embedding_config():
    """Create a sample embedding configuration for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = EmbeddingConfig(
            output_dir=Path(temp_dir) / "embeddings",
            dataset_name="test/legal-dataset",
            embedding_model=EmbeddingModelConfig(
                name="sentence-transformers/all-MiniLM-L6-v2", max_seq_length=384
            ),
            batch_size=16,
            num_output_shards=2,
            max_documents=None,
            ingest_batch_size=32,
            upsert=True,
        )

        # Create the directory structure
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / config.CHUNK_EMBEDDINGS_DIR).mkdir(exist_ok=True)
        (config.output_dir / config.AGG_EMBEDDINGS_DIR).mkdir(exist_ok=True)

        # Create dummy parquet files to simulate embeddings
        dummy_chunk_data = Dataset.from_dict(
            {
                "document_id": ["doc_1", "doc_2"],
                "chunk_id": ["chunk_1", "chunk_2"],
                "chunk_text": ["Sample chunk 1", "Sample chunk 2"],
                "embedding": [np.random.rand(384).tolist() for _ in range(2)],
            }
        )

        dummy_doc_data = Dataset.from_dict(
            {
                "document_id": ["doc_1", "doc_2"],
                "embedding": [np.random.rand(384).tolist() for _ in range(2)],
            }
        )

        # Save as parquet files
        chunk_path = config.output_dir / config.CHUNK_EMBEDDINGS_DIR / "data.parquet"
        doc_path = config.output_dir / config.AGG_EMBEDDINGS_DIR / "data.parquet"

        dummy_chunk_data.to_parquet(str(chunk_path))
        dummy_doc_data.to_parquet(str(doc_path))

        yield config


class MockBatchContext:
    """Mock context manager for Weaviate batch operations."""

    def __init__(self):
        self.add_object = Mock()
        self.add_data_object = Mock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockDatabase:
    """Mock database class with proper context manager support."""

    def __init__(self):
        # Mock collections
        self.legal_documents_collection = Mock()
        self.document_chunks_collection = Mock()

        # Set up batch context managers
        self.legal_documents_collection.batch.fixed_size.return_value = MockBatchContext()
        self.document_chunks_collection.batch.fixed_size.return_value = MockBatchContext()

        # Other methods
        self.get_collection_size = Mock(return_value=0)
        self.get_uuids = Mock(return_value=[])
        self.create_collections = Mock(return_value=None)

        # Collection getter
        def get_collection(name):
            if "Documents" in name:
                return self.legal_documents_collection
            else:
                return self.document_chunks_collection

        self.get_collection = Mock(side_effect=get_collection)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def reset_mock(self):
        """Reset all mock objects."""
        self.legal_documents_collection.reset_mock()
        self.document_chunks_collection.reset_mock()
        self.get_collection_size.reset_mock()
        self.get_uuids.reset_mock()
        self.create_collections.reset_mock()
        self.get_collection.reset_mock()

        # Re-setup batch context managers
        self.legal_documents_collection.batch.fixed_size.return_value = MockBatchContext()
        self.document_chunks_collection.batch.fixed_size.return_value = MockBatchContext()

        # Reset return values
        self.get_collection_size.return_value = 0
        self.get_uuids.return_value = []
        self.create_collections.return_value = None

        # Re-setup collection getter
        def get_collection(name):
            if "Documents" in name:
                return self.legal_documents_collection
            else:
                return self.document_chunks_collection

        self.get_collection.side_effect = get_collection


@pytest.fixture
def mock_weaviate_database():
    """Create a comprehensive mock for WeaviateLegalDocumentsDatabase."""
    return MockDatabase()


@pytest.fixture
def sample_legal_documents():
    """Create sample legal document data for testing."""
    return Dataset.from_dict(
        {
            "document_id": ["doc_1", "doc_2", "doc_3", "doc_4"],
            "title": [
                "Contract Law Case - Smith v. Jones",
                "Tax Interpretation - Deduction Guidelines",
                "Criminal Procedure - Evidence Rules",
                "Administrative Law - License Requirements",
            ],
            "language": ["en", "en", "en", "en"],
            "country": ["US", "US", "UK", "CA"],
            "date_issued": ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05"],
            "document_type": ["judgment", "tax_interpretation", "judgment", "legal_act"],
            "full_text": [
                "This is a contract law case involving breach of contract...",
                "Tax interpretation regarding business expense deductions...",
                "Criminal procedure case establishing evidence admissibility rules...",
                "Administrative regulation for professional license requirements...",
            ],
            "summary": [
                "Contract breach case establishing precedent for damages",
                "Guidelines for allowable business deductions under current tax code",
                "Rules for evidence admissibility in criminal proceedings",
                "Professional licensing requirements and procedures",
            ],
            "keywords": [
                '["contract", "breach", "damages", "civil"]',
                '["tax", "deduction", "business", "expenses"]',
                '["criminal", "evidence", "procedure", "admissibility"]',
                '["administrative", "license", "professional", "regulation"]',
            ],
            "issuing_body": [
                '{"court": "Superior Court", "jurisdiction": "State"}',
                '{"agency": "Tax Authority", "department": "Business Division"}',
                '{"court": "High Court", "jurisdiction": "Federal"}',
                '{"agency": "Professional Licensing Board", "jurisdiction": "Provincial"}',
            ],
        }
    )


@pytest.fixture
def sample_document_embeddings():
    """Create sample document embeddings for testing."""
    embedding_dim = 384
    return Dataset.from_dict(
        {
            "document_id": ["doc_1", "doc_2", "doc_3", "doc_4"],
            "embedding": [np.random.rand(embedding_dim).tolist() for _ in range(4)],
        }
    )


@pytest.fixture
def sample_document_chunks():
    """Create sample document chunks with embeddings for testing."""
    embedding_dim = 384
    return Dataset.from_dict(
        {
            "document_id": ["doc_1", "doc_1", "doc_2", "doc_2", "doc_3", "doc_4"],
            "chunk_id": ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5", "chunk_6"],
            "chunk_text": [
                "This is the first chunk of document 1 containing facts and background.",
                "This is the second chunk of document 1 with legal reasoning and conclusion.",
                "First chunk of tax interpretation document with regulation overview.",
                "Second chunk of tax interpretation with specific deduction guidelines.",
                "Criminal procedure chunk discussing evidence admissibility standards.",
                "Administrative law chunk outlining licensing procedure requirements.",
            ],
            "embedding": [np.random.rand(embedding_dim).tolist() for _ in range(6)],
            "position": [0, 1, 0, 1, 0, 0],
            "cited_references": [
                '["Smith v. Brown (2020)", "Civil Code Sec. 1234"]',
                '["Precedent Case XYZ", "Contract Law Manual"]',
                '["Tax Code Section 501", "Revenue Ruling 2023-01"]',
                '["Business Expense Guidelines", "IRS Publication 535"]',
                '["Evidence Code 352", "People v. Smith (2019)"]',
                '["Professional Licensing Act", "Administrative Code 15.2"]',
            ],
            "tags": [
                '["facts", "background", "contract"]',
                '["reasoning", "conclusion", "damages"]',
                '["regulation", "overview", "tax"]',
                '["guidelines", "deduction", "business"]',
                '["evidence", "criminal", "procedure"]',
                '["licensing", "administrative", "professional"]',
            ],
        }
    )


@pytest.fixture
def embedding_vectors():
    """Generate consistent embedding vectors for testing."""

    def _generate_embeddings(count: int, dim: int = 384):
        """Generate deterministic embeddings for consistent testing."""
        np.random.seed(42)  # Fixed seed for reproducible tests
        return [np.random.rand(dim).tolist() for _ in range(count)]

    return _generate_embeddings
