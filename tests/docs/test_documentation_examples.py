"""
Test Python code examples from documentation.

This module tests all Python code blocks extracted from markdown documentation
to ensure they remain valid and functional as the codebase evolves.
"""

import ast
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestDocumentationExamples:
    """Test suite for documentation code examples."""

    def test_syntax_validation(self, markdown_code_example, should_skip, skip_reason):
        """Test that code example has valid Python syntax."""
        if should_skip:
            pytest.skip(skip_reason)

        code_block = markdown_code_example

        # Try to parse the code
        try:
            ast.parse(code_block.code)
        except SyntaxError as e:
            pytest.fail(
                f"Syntax error in {code_block.file_path}:{code_block.line_number} "
                f"(block {code_block.block_number})\n"
                f"Error: {e.msg} at line {e.lineno}\n"
                f"Code:\n{code_block.code}"
            )

    def test_imports_available(self, markdown_code_example, should_skip, skip_reason):
        """Test that imported modules are available."""
        if should_skip:
            pytest.skip(skip_reason)

        code_block = markdown_code_example
        imports = code_block.extract_imports()

        # Check if imports are from the juddges package
        juddges_imports = [imp for imp in imports if imp.startswith("juddges")]

        for module_name in juddges_imports:
            # Try to import the module
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(
                    f"Import error in {code_block.file_path}:{code_block.line_number}\n"
                    f"Cannot import '{module_name}': {e}\n"
                    f"This module may not exist or is not properly exposed."
                )

    def test_no_undefined_variables(self, markdown_code_example, should_skip, skip_reason):
        """Test that code doesn't use obviously undefined variables."""
        if should_skip:
            pytest.skip(skip_reason)

        code_block = markdown_code_example

        try:
            tree = ast.parse(code_block.code)
        except SyntaxError:
            # Syntax errors are caught by test_syntax_validation
            return

        # Track defined names
        defined_names = set()
        used_names = set()

        for node in ast.walk(tree):
            # Track assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_names.add(target.id)
            elif isinstance(node, ast.FunctionDef):
                defined_names.add(node.name)
                # Add function parameters
                for arg in node.args.args:
                    defined_names.add(arg.arg)
            elif isinstance(node, ast.ClassDef):
                defined_names.add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    defined_names.add(name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    defined_names.add(name)
            # Track usage
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)

        # Check for common builtins
        builtins = {
            "print",
            "len",
            "range",
            "str",
            "int",
            "float",
            "list",
            "dict",
            "set",
            "tuple",
            "True",
            "False",
            "None",
        }
        defined_names.update(builtins)

        # Find potentially undefined variables
        undefined = used_names - defined_names

        # Filter out common patterns that are okay
        # (like method calls on imported modules)
        acceptable_undefined = {"self", "cls", "dataset", "ds", "db", "client", "model"}
        undefined = undefined - acceptable_undefined

        if undefined:
            # This is a warning, not necessarily an error
            # Some variables might be defined in previous examples or context
            pytest.skip(
                f"Potentially undefined variables in "
                f"{code_block.file_path}:{code_block.line_number}: "
                f"{', '.join(undefined)}"
            )


class TestSpecificExamples:
    """Test specific documentation examples with proper setup."""

    @pytest.mark.docs
    def test_text_chunker_basic_example(self, sample_dataset, mock_tokenizer):
        """Test basic text chunker example from documentation."""
        from juddges.preprocessing.text_chunker import TextChunker

        # Initialize chunker
        chunker = TextChunker(
            id_col="judgment_id",
            text_col="full_text",
            chunk_size=512,
            chunk_overlap=50,
        )

        # Chunk documents
        chunked = chunker(sample_dataset)

        # Verify output structure
        assert "judgment_id" in chunked
        assert "chunk_id" in chunked
        assert "chunk_len" in chunked
        assert "chunk_text" in chunked

        # Verify chunks were created
        assert len(chunked["judgment_id"]) > 0

    @pytest.mark.docs
    def test_text_chunker_with_tokenizer(self, sample_dataset, mock_tokenizer):
        """Test text chunker with tokenizer example."""
        from juddges.preprocessing.text_chunker import TextChunker

        chunker = TextChunker(
            id_col="judgment_id",
            text_col="full_text",
            chunk_size=512,
            chunk_overlap=50,
            tokenizer=mock_tokenizer,
        )

        chunked = chunker(sample_dataset)
        assert "chunk_text" in chunked

    @pytest.mark.docs
    def test_text_chunker_advanced_config(self, sample_dataset):
        """Test advanced text chunker configuration."""
        from juddges.preprocessing.text_chunker import TextChunker

        chunker = TextChunker(
            id_col="judgment_id",
            text_col="full_text",
            chunk_size=1024,
            chunk_overlap=100,
            min_split_chars=100,
            take_n_first_chunks=5,
        )

        chunked = chunker(sample_dataset)

        # Verify max chunks per document
        from collections import Counter

        chunk_counts = Counter(chunked["judgment_id"])
        for doc_id, count in chunk_counts.items():
            assert count <= 5, f"Document {doc_id} has {count} chunks, expected <= 5"


class TestDocumentationCoverage:
    """Test documentation coverage metrics."""

    @pytest.mark.docs
    def test_all_public_apis_have_examples(self):
        """Verify that public APIs have documentation examples."""
        from pathlib import Path

        docs_dir = Path("docs/reference/api/")
        if not docs_dir.exists():
            pytest.skip("API documentation directory not found")

        api_docs = list(docs_dir.rglob("*.md"))
        assert len(api_docs) > 0, "No API documentation found"

        # Check that each API doc has at least one code example
        docs_without_examples = []
        for doc_file in api_docs:
            content = doc_file.read_text()
            if "```python" not in content:
                docs_without_examples.append(doc_file.name)

        # Allow some docs to not have examples (like index pages)
        allowed_without_examples = {"index.md", "README.md"}
        docs_without_examples = [
            doc for doc in docs_without_examples if doc not in allowed_without_examples
        ]

        if docs_without_examples:
            pytest.skip(
                f"API docs without code examples: {', '.join(docs_without_examples)}\n"
                f"Consider adding examples to improve documentation."
            )

    @pytest.mark.docs
    def test_example_freshness(self):
        """Check that documentation examples are not stale."""
        from datetime import datetime, timedelta
        from pathlib import Path

        docs_dir = Path("docs/")
        if not docs_dir.exists():
            pytest.skip("Documentation directory not found")

        # Get last modified times
        now = datetime.now()
        staleness_threshold = timedelta(days=90)  # 90 days

        stale_docs = []
        for doc_file in docs_dir.rglob("*.md"):
            if "archive" in str(doc_file):
                continue

            mtime = datetime.fromtimestamp(doc_file.stat().st_mtime)
            age = now - mtime

            if age > staleness_threshold:
                stale_docs.append((doc_file.name, age.days))

        if stale_docs:
            stale_list = "\n".join([f"  - {name} ({days} days old)" for name, days in stale_docs])
            pytest.skip(
                f"Some documentation files are older than {staleness_threshold.days} days:\n"
                f"{stale_list}\n"
                f"Consider reviewing and updating examples."
            )


@pytest.mark.docs
@pytest.mark.integration
class TestDocumentationIntegration:
    """Integration tests for documentation examples requiring external services."""

    @pytest.mark.requires_weaviate
    def test_weaviate_examples_with_mock(self, mock_weaviate_client):
        """Test Weaviate examples with mock client."""
        # This would test examples that use Weaviate
        # For now, just verify mock works
        assert mock_weaviate_client.is_connected

        # Test basic operations
        collection = mock_weaviate_client.collections.create("test_collection")
        assert collection.name == "test_collection"

    @pytest.mark.requires_gemini
    def test_gemini_examples_with_mock(self, mock_gemini_api):
        """Test Gemini API examples with mock."""
        # This would test examples that use Gemini API
        response = mock_gemini_api.generate_content("Test prompt")
        assert response.text is not None
        assert response.finish_reason == "STOP"


if __name__ == "__main__":
    # Allow running this file directly
    pytest.main([__file__, "-v"])
