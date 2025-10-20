"""
Fixtures and mocks for testing documentation code examples.

This module provides mock implementations of external services and dependencies
to allow documentation examples to run in isolated test environments.
"""

from typing import Any
from unittest.mock import MagicMock


class MockWeaviateClient:
    """Mock Weaviate client for testing."""

    def __init__(self, url: str = "http://localhost:8080", **kwargs):
        self.url = url
        self.collections = MockCollections()
        self.is_connected = True

    def connect(self):
        """Mock connect method."""
        self.is_connected = True
        return self

    def close(self):
        """Mock close method."""
        self.is_connected = False

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class MockCollections:
    """Mock Weaviate collections."""

    def __init__(self):
        self._collections = {}

    def create(self, name: str, **kwargs):
        """Mock create collection."""
        self._collections[name] = MockCollection(name)
        return self._collections[name]

    def get(self, name: str):
        """Mock get collection."""
        if name not in self._collections:
            self._collections[name] = MockCollection(name)
        return self._collections[name]

    def delete(self, name: str):
        """Mock delete collection."""
        if name in self._collections:
            del self._collections[name]

    def list_all(self):
        """Mock list all collections."""
        return list(self._collections.keys())


class MockCollection:
    """Mock Weaviate collection."""

    def __init__(self, name: str):
        self.name = name
        self._data = []

    def data(self):
        """Mock data operations."""
        return MockDataOperations(self._data)

    def query(self):
        """Mock query operations."""
        return MockQueryOperations(self._data)

    def aggregate(self):
        """Mock aggregate operations."""
        return MockAggregateOperations(self._data)


class MockDataOperations:
    """Mock data operations."""

    def __init__(self, data: list):
        self._data = data

    def insert(self, properties: dict, **kwargs):
        """Mock insert."""
        self._data.append(properties)
        return {"id": f"mock-id-{len(self._data)}"}

    def insert_many(self, objects: list, **kwargs):
        """Mock batch insert."""
        for obj in objects:
            self._data.append(obj)
        return {"inserted": len(objects)}


class MockQueryOperations:
    """Mock query operations."""

    def __init__(self, data: list):
        self._data = data

    def fetch_objects(self, limit: int = 10, **kwargs):
        """Mock fetch objects."""
        return MockQueryResult(self._data[:limit])

    def near_text(self, query: str, **kwargs):
        """Mock near_text query."""
        return MockQueryBuilder(self._data)

    def bm25(self, query: str, **kwargs):
        """Mock BM25 query."""
        return MockQueryBuilder(self._data)

    def hybrid(self, query: str, **kwargs):
        """Mock hybrid query."""
        return MockQueryBuilder(self._data)


class MockQueryBuilder:
    """Mock query builder."""

    def __init__(self, data: list):
        self._data = data
        self._limit = 10

    def with_limit(self, limit: int):
        """Mock limit."""
        self._limit = limit
        return self

    def with_additional(self, properties: list):
        """Mock additional properties."""
        return self

    def do(self):
        """Execute query."""
        return MockQueryResult(self._data[: self._limit])


class MockQueryResult:
    """Mock query result."""

    def __init__(self, objects: list):
        self.objects = objects


class MockAggregateOperations:
    """Mock aggregate operations."""

    def __init__(self, data: list):
        self._data = data

    def over_all(self, **kwargs):
        """Mock aggregate over all."""
        return {"total_count": len(self._data)}


class MockGeminiAPI:
    """Mock Gemini API for testing."""

    def __init__(self, api_key: str = "mock-api-key"):
        self.api_key = api_key

    def generate_content(self, prompt: str, **kwargs):
        """Mock content generation."""
        return MockGeminiResponse(
            text=f"Mock response for: {prompt[:50]}...",
            finish_reason="STOP",
        )

    def generate_content_async(self, prompt: str, **kwargs):
        """Mock async content generation."""
        import asyncio

        return asyncio.Future().set_result(
            MockGeminiResponse(
                text=f"Mock response for: {prompt[:50]}...",
                finish_reason="STOP",
            )
        )


class MockGeminiResponse:
    """Mock Gemini API response."""

    def __init__(self, text: str, finish_reason: str):
        self.text = text
        self.finish_reason = finish_reason
        self.candidates = [MockCandidate(text)]

    @property
    def parts(self):
        """Mock parts."""
        return [MockPart(self.text)]


class MockCandidate:
    """Mock Gemini candidate."""

    def __init__(self, text: str):
        self.text = text
        self.content = MockContent(text)


class MockContent:
    """Mock Gemini content."""

    def __init__(self, text: str):
        self.text = text
        self.parts = [MockPart(text)]


class MockPart:
    """Mock Gemini part."""

    def __init__(self, text: str):
        self.text = text


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.model_max_length = 512

    def __call__(self, text: str | list[str], **kwargs):
        """Tokenize text."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        # Simple tokenization: split on whitespace
        tokens = []
        for t in texts:
            tokens.append(t.split())

        return MockTokenizerOutput(tokens)

    def encode(self, text: str, **kwargs):
        """Encode text to token IDs."""
        # Simple encoding: map each word to a number
        words = text.split()
        return list(range(len(words)))

    def decode(self, token_ids: list[int], **kwargs):
        """Decode token IDs to text."""
        # Simple decoding: map numbers back to placeholder text
        return " ".join([f"token_{i}" for i in token_ids])

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """Mock from_pretrained."""
        return cls()


class MockTokenizerOutput:
    """Mock tokenizer output."""

    def __init__(self, tokens: list[list[str]]):
        self.tokens = tokens
        self.input_ids = [[hash(t) % 50000 for t in token_list] for token_list in tokens]

    def __getitem__(self, key):
        """Get item."""
        if key == "input_ids":
            return self.input_ids
        return getattr(self, key)


class MockDataset:
    """Mock HuggingFace dataset."""

    def __init__(self, data: dict):
        self.data = data
        self.column_names = list(data.keys())

    def map(self, function, **kwargs):
        """Mock map operation."""
        # For testing, just return self
        return self

    def __getitem__(self, idx):
        """Get item by index."""
        if isinstance(idx, int):
            return {key: values[idx] for key, values in self.data.items()}
        return self.data

    def __len__(self):
        """Get dataset length."""
        if self.data:
            return len(next(iter(self.data.values())))
        return 0

    def save_to_disk(self, path: str):
        """Mock save to disk."""
        pass

    @classmethod
    def load_from_disk(cls, path: str):
        """Mock load from disk."""
        return cls(
            {
                "judgment_id": ["doc1", "doc2"],
                "full_text": ["Sample text 1", "Sample text 2"],
            }
        )


def mock_load_dataset(dataset_name: str, **kwargs):
    """Mock datasets.load_dataset function."""
    return MockDataset(
        {
            "judgment_id": ["doc1", "doc2", "doc3"],
            "full_text": [
                "Sample legal document text. " * 50,
                "Another sample document. " * 50,
                "Third document content. " * 50,
            ],
        }
    )


class MockAutoTokenizer:
    """Mock AutoTokenizer class."""

    @staticmethod
    def from_pretrained(model_name: str, **kwargs):
        """Mock from_pretrained."""
        return MockTokenizer()


# Monkey-patching helper functions
def patch_external_dependencies(monkeypatch):
    """Patch external dependencies for isolated testing."""

    # Patch Weaviate
    try:
        import weaviate

        monkeypatch.setattr(weaviate, "Client", MockWeaviateClient)
    except ImportError:
        pass

    # Patch datasets
    try:
        import datasets

        monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)
    except ImportError:
        pass

    # Patch transformers
    try:
        from transformers import AutoTokenizer

        monkeypatch.setattr(AutoTokenizer, "from_pretrained", MockAutoTokenizer.from_pretrained)
    except ImportError:
        pass
