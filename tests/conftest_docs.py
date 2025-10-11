"""
Pytest plugin for testing documentation code examples.

This module provides pytest fixtures and hooks for extracting and testing
Python code examples from markdown documentation files.
"""

import ast
import re
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--docs-path",
        action="store",
        default="docs/",
        help="Path to documentation directory",
    )
    parser.addoption(
        "--skip-external-deps",
        action="store_true",
        default=False,
        help="Skip examples requiring external dependencies",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "docs: Mark test as documentation example test")
    config.addinivalue_line(
        "markers", "requires_weaviate: Mark test as requiring Weaviate instance"
    )
    config.addinivalue_line("markers", "requires_gemini: Mark test as requiring Gemini API key")
    config.addinivalue_line("markers", "requires_gpu: Mark test as requiring GPU")
    config.addinivalue_line(
        "markers", "requires_network: Mark test as requiring network access"
    )


@pytest.fixture(scope="session")
def doctest_config() -> dict:
    """Load doctest configuration."""
    config_path = Path(".doctest.yaml")
    if config_path.exists():
        return yaml.safe_load(config_path.read_text())
    return {}


@pytest.fixture(scope="session")
def docs_path(request) -> Path:
    """Get documentation path."""
    path = request.config.getoption("--docs-path")
    return Path(path)


@pytest.fixture
def mock_weaviate_client():
    """Provide mock Weaviate client."""
    from tests.docs.fixtures import MockWeaviateClient

    return MockWeaviateClient()


@pytest.fixture
def mock_gemini_api():
    """Provide mock Gemini API."""
    from tests.docs.fixtures import MockGeminiAPI

    return MockGeminiAPI()


@pytest.fixture
def sample_dataset():
    """Provide sample dataset for testing."""
    return {
        "judgment_id": ["doc1", "doc2", "doc3"],
        "full_text": [
            "This is a sample legal document with sufficient length for testing purposes. " * 20,
            "Another sample document with different content for testing. " * 20,
            "Third document with unique content for comprehensive testing. " * 20,
        ],
    }


@pytest.fixture
def mock_tokenizer():
    """Provide mock tokenizer."""
    from tests.docs.fixtures import MockTokenizer

    return MockTokenizer()


class MarkdownCodeBlock:
    """Represents a code block from markdown."""

    def __init__(
        self,
        file_path: Path,
        block_number: int,
        line_number: int,
        code: str,
        language: str,
    ):
        self.file_path = file_path
        self.block_number = block_number
        self.line_number = line_number
        self.code = code
        self.language = language

    @property
    def id(self) -> str:
        """Generate unique ID for this code block."""
        return f"{self.file_path.name}::{self.block_number}"

    def has_marker(self, marker: str) -> bool:
        """Check if code contains a marker."""
        return marker in self.code

    def extract_imports(self) -> list[str]:
        """Extract import statements from code."""
        try:
            tree = ast.parse(self.code)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            return imports
        except SyntaxError:
            return []


def extract_code_blocks(file_path: Path) -> Iterator[MarkdownCodeBlock]:
    """Extract Python code blocks from a markdown file."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return

    # Pattern to match fenced code blocks
    pattern = r"```(python|python3)\n(.*?)```"
    matches = re.finditer(pattern, content, re.DOTALL)

    for block_num, match in enumerate(matches, 1):
        language = match.group(1)
        code = match.group(2).strip()

        # Find line number
        line_number = content[: match.start()].count("\n") + 1

        yield MarkdownCodeBlock(
            file_path=file_path,
            block_number=block_num,
            line_number=line_number,
            code=code,
            language=language,
        )


def should_skip_example(code_block: MarkdownCodeBlock, config: dict) -> tuple[bool, str]:
    """Determine if a code example should be skipped."""
    # Check for skip markers
    skip_markers = config.get("annotations", {}).get("skip", [])
    for marker in skip_markers:
        if code_block.has_marker(marker):
            return True, f"Contains skip marker: {marker}"

    # Check for demonstration markers
    demo_markers = config.get("code_blocks", {}).get("demo_markers", [])
    for marker in demo_markers:
        if marker in code_block.code:
            return True, f"Demonstration code: {marker}"

    # Check for placeholder patterns
    placeholder_patterns = config.get("code_blocks", {}).get("placeholder_patterns", [])
    for pattern in placeholder_patterns:
        if re.search(pattern, code_block.code):
            return True, f"Contains placeholder: {pattern}"

    return False, ""


def pytest_generate_tests(metafunc):
    """Dynamically generate tests for code examples."""
    if "markdown_code_example" in metafunc.fixturenames:
        # Load configuration
        config_path = Path(".doctest.yaml")
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text())
        else:
            config = {}

        # Get documentation path
        docs_path = Path(config.get("scan_paths", ["docs/"])[0])

        # Collect all code examples
        code_examples = []
        for md_file in docs_path.rglob("*.md"):
            # Skip excluded paths
            exclude_paths = config.get("exclude_paths", [])
            if any(exclude in str(md_file) for exclude in exclude_paths):
                continue

            for code_block in extract_code_blocks(md_file):
                # Check if should skip
                should_skip, reason = should_skip_example(code_block, config)

                code_examples.append((code_block, should_skip, reason))

        # Generate test IDs
        ids = [example[0].id for example in code_examples]

        # Parametrize the test
        metafunc.parametrize(
            "markdown_code_example,should_skip,skip_reason",
            code_examples,
            ids=ids,
        )
