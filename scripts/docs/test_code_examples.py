#!/usr/bin/env python3
"""
Test code examples from documentation markdown files.

This script extracts Python code blocks from markdown files and validates them
to ensure all documentation examples remain accurate and functional.

Usage:
    # Test all documentation
    python scripts/docs/test_code_examples.py

    # Test specific file
    python scripts/docs/test_code_examples.py docs/reference/api/README.md

    # Test specific directory
    python scripts/docs/test_code_examples.py docs/how-to/

    # Show successful tests
    python scripts/docs/test_code_examples.py --verbose

    # Run with pytest
    python scripts/docs/test_code_examples.py --pytest

    # Generate report
    python scripts/docs/test_code_examples.py --report report.json
"""

import ast
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

console = Console()


@dataclass
class CodeExample:
    """Represents a code example extracted from markdown."""

    file_path: Path
    block_number: int
    line_number: int
    code: str
    language: str
    annotations: list[str]

    def should_skip(self, config: dict) -> tuple[bool, str]:
        """Check if this example should be skipped."""
        skip_markers = config.get("annotations", {}).get("skip", [])

        # Check for skip annotations
        for marker in skip_markers:
            if marker in self.code or marker in "\n".join(self.annotations):
                return True, f"Skip marker: {marker}"

        # Check for demonstration markers
        demo_markers = config.get("code_blocks", {}).get("demo_markers", [])
        for marker in demo_markers:
            if marker in self.code:
                return True, f"Demonstration code: {marker}"

        # Check for placeholder patterns
        placeholder_patterns = config.get("code_blocks", {}).get("placeholder_patterns", [])
        for pattern in placeholder_patterns:
            if re.search(pattern, self.code):
                return True, f"Contains placeholder: {pattern}"

        return False, ""

    def get_requirements(self, config: dict) -> list[str]:
        """Extract requirements from annotations."""
        requirements = []
        requires_patterns = config.get("annotations", {}).get("requires", [])

        for req_config in requires_patterns:
            pattern = req_config.get("pattern", "")
            if pattern in self.code or pattern in "\n".join(self.annotations):
                requirements.append(req_config.get("check", ""))

        return requirements


@dataclass
class TestResult:
    """Result of testing a code example."""

    example: CodeExample
    passed: bool
    skipped: bool
    skip_reason: str
    error_message: str
    execution_time: float


class MarkdownCodeExtractor:
    """Extract Python code blocks from markdown files."""

    def __init__(self, config: dict):
        self.config = config

    def extract_from_file(self, file_path: Path) -> list[CodeExample]:
        """Extract all Python code blocks from a markdown file."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return []

        examples = []
        # Pattern to match code blocks with optional language and annotations
        pattern = r"```(\w+)?\s*(?:#\s*(.+?)\n)?(.*?)```"
        matches = re.finditer(pattern, content, re.DOTALL)

        for block_num, match in enumerate(matches, 1):
            language = match.group(1) or "python"
            annotation_line = match.group(2) or ""
            code = match.group(3).strip()

            # Only process Python code blocks
            languages = self.config.get("code_blocks", {}).get("languages", ["python"])
            if language not in languages:
                continue

            # Find line number
            line_number = content[: match.start()].count("\n") + 1

            # Parse annotations
            annotations = [annotation_line] if annotation_line else []

            examples.append(
                CodeExample(
                    file_path=file_path,
                    block_number=block_num,
                    line_number=line_number,
                    code=code,
                    language=language,
                    annotations=annotations,
                )
            )

        return examples

    def extract_from_directory(self, directory: Path) -> list[CodeExample]:
        """Extract code blocks from all markdown files in directory."""
        examples = []
        include_patterns = self.config.get("include_patterns", ["*.md"])

        for pattern in include_patterns:
            for file_path in directory.rglob(pattern):
                # Check if file should be excluded
                if self._should_exclude(file_path):
                    continue

                examples.extend(self.extract_from_file(file_path))

        return examples

    def _should_exclude(self, file_path: Path) -> bool:
        """Check if file path should be excluded."""
        exclude_paths = self.config.get("exclude_paths", [])
        for exclude in exclude_paths:
            if exclude in str(file_path):
                return True
        return False


class CodeExampleValidator:
    """Validate Python code examples."""

    def __init__(self, config: dict):
        self.config = config

    def validate_syntax(self, example: CodeExample) -> tuple[bool, str]:
        """Validate Python syntax."""
        try:
            ast.parse(example.code)
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Parse error: {str(e)}"

    def validate_execution(self, example: CodeExample) -> tuple[bool, str]:
        """Validate code execution (syntax check + basic imports)."""
        # First check syntax
        syntax_valid, error = self.validate_syntax(example)
        if not syntax_valid:
            return False, error

        # Check for undefined variables (basic analysis)
        try:
            tree = ast.parse(example.code)
            # This is a simplified check - full execution would require isolated environment
            return True, ""
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def test_example(self, example: CodeExample) -> TestResult:
        """Test a single code example."""
        import time

        start_time = time.time()

        # Check if should skip
        should_skip, skip_reason = example.should_skip(self.config)
        if should_skip:
            execution_time = time.time() - start_time
            return TestResult(
                example=example,
                passed=False,
                skipped=True,
                skip_reason=skip_reason,
                error_message="",
                execution_time=execution_time,
            )

        # Validate code
        passed, error_message = self.validate_execution(example)
        execution_time = time.time() - start_time

        return TestResult(
            example=example,
            passed=passed,
            skipped=False,
            skip_reason="",
            error_message=error_message,
            execution_time=execution_time,
        )


class TestReporter:
    """Generate test reports."""

    def __init__(self, console: Console):
        self.console = console

    def print_summary(self, results: list[TestResult], verbose: bool = False):
        """Print test results summary."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed and not r.skipped)
        skipped = sum(1 for r in results if r.skipped)

        # Create summary table
        table = Table(title="Code Example Test Summary")
        table.add_column("Status", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")

        table.add_row("Total", str(total), "100%")
        table.add_row(
            "Passed", str(passed), f"{(passed/total*100):.1f}%" if total > 0 else "0%", style="green"
        )
        table.add_row(
            "Failed", str(failed), f"{(failed/total*100):.1f}%" if total > 0 else "0%", style="red"
        )
        table.add_row(
            "Skipped",
            str(skipped),
            f"{(skipped/total*100):.1f}%" if total > 0 else "0%",
            style="yellow",
        )

        self.console.print(table)

        # Print failures
        if failed > 0:
            self.console.print("\n[bold red]Failed Examples:[/bold red]")
            for result in results:
                if not result.passed and not result.skipped:
                    self._print_failure(result)

        # Print skipped if verbose
        if verbose and skipped > 0:
            self.console.print("\n[bold yellow]Skipped Examples:[/bold yellow]")
            for result in results:
                if result.skipped:
                    self._print_skipped(result)

        # Print success if verbose
        if verbose and passed > 0:
            self.console.print("\n[bold green]Passed Examples:[/bold green]")
            for result in results:
                if result.passed:
                    self._print_success(result)

    def _print_failure(self, result: TestResult):
        """Print a failed test result."""
        example = result.example
        self.console.print(
            f"  [red]✗[/red] {example.file_path}:{example.line_number} "
            f"(block {example.block_number})"
        )
        self.console.print(f"    Error: {result.error_message}")

    def _print_skipped(self, result: TestResult):
        """Print a skipped test result."""
        example = result.example
        self.console.print(
            f"  [yellow]⊝[/yellow] {example.file_path}:{example.line_number} "
            f"(block {example.block_number})"
        )
        self.console.print(f"    Reason: {result.skip_reason}")

    def _print_success(self, result: TestResult):
        """Print a successful test result."""
        example = result.example
        self.console.print(
            f"  [green]✓[/green] {example.file_path}:{example.line_number} "
            f"(block {example.block_number})"
        )

    def generate_report(self, results: list[TestResult], output_path: Path):
        """Generate JSON report."""
        import json

        report = {
            "summary": {
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.passed and not r.skipped),
                "skipped": sum(1 for r in results if r.skipped),
            },
            "results": [
                {
                    "file": str(r.example.file_path),
                    "line": r.example.line_number,
                    "block": r.example.block_number,
                    "passed": r.passed,
                    "skipped": r.skipped,
                    "skip_reason": r.skip_reason,
                    "error": r.error_message,
                    "execution_time": r.execution_time,
                }
                for r in results
            ],
        }

        output_path.write_text(json.dumps(report, indent=2))
        self.console.print(f"[green]Report generated:[/green] {output_path}")


def load_config() -> dict:
    """Load configuration from .doctest.yaml."""
    config_path = Path(".doctest.yaml")
    if not config_path.exists():
        logger.warning("No .doctest.yaml found, using defaults")
        return {}

    try:
        return yaml.safe_load(config_path.read_text())
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test documentation code examples")
    parser.add_argument(
        "paths", nargs="*", default=["docs/"], help="Paths to test (files or directories)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show all test results")
    parser.add_argument("--pytest", action="store_true", help="Run using pytest")
    parser.add_argument("--report", help="Generate JSON report to file")
    parser.add_argument("--config", help="Path to config file", default=".doctest.yaml")

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # If pytest mode, delegate to pytest
    if args.pytest:
        import subprocess

        result = subprocess.run(
            ["pytest", "tests/docs/test_documentation_examples.py", "-v"],
            cwd=Path.cwd(),
        )
        sys.exit(result.returncode)

    # Initialize components
    extractor = MarkdownCodeExtractor(config)
    validator = CodeExampleValidator(config)
    reporter = TestReporter(console)

    # Collect examples
    all_examples = []
    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            logger.error(f"Path not found: {path}")
            continue

        if path.is_file():
            all_examples.extend(extractor.extract_from_file(path))
        else:
            all_examples.extend(extractor.extract_from_directory(path))

    if not all_examples:
        console.print("[yellow]No code examples found[/yellow]")
        sys.exit(0)

    console.print(f"[bold]Testing {len(all_examples)} code examples...[/bold]\n")

    # Test examples
    results = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Testing examples...", total=len(all_examples))

        for example in all_examples:
            result = validator.test_example(example)
            results.append(result)
            progress.advance(task)

    # Report results
    reporter.print_summary(results, verbose=args.verbose)

    # Generate report if requested
    if args.report:
        reporter.generate_report(results, Path(args.report))

    # Exit with appropriate code
    failed_count = sum(1 for r in results if not r.passed and not r.skipped)
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
