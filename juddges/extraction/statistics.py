"""Statistics calculation and reporting for extraction and ingestion operations.

This module provides utilities for calculating field coverage, success rates,
and generating summary reports for extraction and Weaviate ingestion operations.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console

console = Console()


@dataclass
class FieldCoverage:
    """Statistics for a single field's coverage."""

    populated: int = 0
    empty: int = 0

    @property
    def total(self) -> int:
        """Total number of documents for this field."""
        return self.populated + self.empty

    @property
    def coverage_rate(self) -> float:
        """Coverage rate as percentage (0-100)."""
        if self.total == 0:
            return 0.0
        return (self.populated / self.total) * 100


@dataclass
class ExtractionStatistics:
    """Statistics for an extraction run."""

    total_documents: int
    successful_extractions: int
    failed_extractions: int
    field_coverage: Dict[str, FieldCoverage] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage (0-100)."""
        if self.total_documents == 0:
            return 0.0
        return (self.successful_extractions / self.total_documents) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_documents": self.total_documents,
            "successful_extractions": self.successful_extractions,
            "failed_extractions": self.failed_extractions,
            "success_rate": f"{self.success_rate:.1f}%",
            "field_coverage": {
                field_name: {
                    "populated": coverage.populated,
                    "empty": coverage.empty,
                    "total": coverage.total,
                    "coverage_rate": f"{coverage.coverage_rate:.1f}%",
                }
                for field_name, coverage in self.field_coverage.items()
            },
        }


@dataclass
class IngestionStatistics:
    """Statistics for a Weaviate ingestion operation."""

    total_documents: int
    successful_updates: int
    failed_updates: int
    skipped_documents: int
    duration_seconds: float
    timestamp: str
    errors: List[Dict[str, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage (0-100)."""
        total = self.successful_updates + self.failed_updates
        if total == 0:
            return 0.0
        return (self.successful_updates / total) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_documents": self.total_documents,
            "successful_updates": self.successful_updates,
            "failed_updates": self.failed_updates,
            "skipped_documents": self.skipped_documents,
            "duration_seconds": round(self.duration_seconds, 2),
            "timestamp": self.timestamp,
            "success_rate": f"{self.success_rate:.1f}%",
            "errors": self.errors,
        }


def calculate_field_coverage(extraction_results: List[Dict[str, Any]]) -> Dict[str, FieldCoverage]:
    """Calculate field coverage statistics from extraction results.

    Args:
        extraction_results: List of extraction results with 'extraction_status' and 'extracted_data'

    Returns:
        Dictionary mapping field names to FieldCoverage statistics
    """
    field_coverage = {}

    for result in extraction_results:
        if result.get("extraction_status") != "success":
            continue

        extracted_data = result.get("extracted_data") or {}
        if not isinstance(extracted_data, dict):
            continue

        for field_name, value in extracted_data.items():
            if field_name not in field_coverage:
                field_coverage[field_name] = FieldCoverage()

            # Check if field is populated
            if _is_field_populated(value):
                field_coverage[field_name].populated += 1
            else:
                field_coverage[field_name].empty += 1

    return field_coverage


def _is_field_populated(value: Any) -> bool:
    """Check if a field value is considered populated.

    Args:
        value: Field value to check

    Returns:
        True if field has meaningful content, False otherwise
    """
    if value is None or value == "":
        return False

    if isinstance(value, str):
        return bool(value.strip())
    elif isinstance(value, (list, dict)):
        return bool(value)
    else:
        # Numbers, booleans, etc. are considered populated if not None
        return True


def generate_extraction_summary(
    documents: List[Dict[str, Any]],
    extraction_results: List[Dict[str, Any]],
) -> ExtractionStatistics:
    """Generate extraction statistics summary.

    Args:
        documents: Original documents
        extraction_results: Extraction results

    Returns:
        ExtractionStatistics object with complete summary
    """
    successful = sum(1 for r in extraction_results if r.get("extraction_status") == "success")
    failed = len(extraction_results) - successful
    field_coverage = calculate_field_coverage(extraction_results)

    return ExtractionStatistics(
        total_documents=len(documents),
        successful_extractions=successful,
        failed_extractions=failed,
        field_coverage=field_coverage,
    )


def save_extraction_results(
    documents: List[Dict[str, Any]],
    extraction_results: List[Dict[str, Any]],
    output_dir: Path,
) -> ExtractionStatistics:
    """Save extraction results and generate statistics.

    Args:
        documents: Original documents with full_text
        extraction_results: Extraction results
        output_dir: Output directory

    Returns:
        ExtractionStatistics with summary information
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full_text documents
    full_text_file = output_dir / "sample_documents_full_text.jsonl"
    with open(full_text_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Save extraction results
    extracted_file = output_dir / "sample_documents_extracted.jsonl"
    with open(extracted_file, "w", encoding="utf-8") as f:
        for result in extraction_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Generate and save summary
    stats = generate_extraction_summary(documents, extraction_results)
    summary_file = output_dir / "extraction_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2)

    return stats


def display_extraction_results(stats: ExtractionStatistics):
    """Display extraction results in formatted console output.

    Args:
        stats: ExtractionStatistics to display
    """
    console.print("\n[bold cyan]Extraction Summary[/bold cyan]")
    console.print(f"Total documents: {stats.total_documents}")
    console.print(f"[green]Successful: {stats.successful_extractions}[/green]")
    console.print(f"[red]Failed: {stats.failed_extractions}[/red]")
    console.print(f"Success rate: {stats.success_rate:.1f}%")

    console.print("\n[bold cyan]Field Coverage[/bold cyan]")
    for field_name in sorted(stats.field_coverage.keys()):
        coverage = stats.field_coverage[field_name]
        console.print(
            f"  {field_name}: {coverage.coverage_rate:.1f}% ({coverage.populated}/{coverage.total})"
        )


def save_ingestion_report(stats: IngestionStatistics, output_dir: Path):
    """Save ingestion report to JSON file.

    Args:
        stats: IngestionStatistics to save
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "ingestion_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2)


def display_ingestion_results(stats: IngestionStatistics):
    """Display ingestion results in formatted console output.

    Args:
        stats: IngestionStatistics to display
    """
    console.print("\n" + "=" * 60)
    console.print("[bold green]✓ Weaviate Ingestion Complete![/bold green]")
    console.print("=" * 60)

    console.print("\n[cyan]Statistics:[/cyan]")
    console.print(f"  • Total documents: {stats.total_documents}")
    console.print(f"  • Successful updates: [green]{stats.successful_updates}[/green]")
    console.print(f"  • Failed updates: [red]{stats.failed_updates}[/red]")
    console.print(f"  • Skipped (failed extractions): [yellow]{stats.skipped_documents}[/yellow]")
    console.print(f"  • Duration: {stats.duration_seconds:.1f} seconds")

    if stats.successful_updates > 0:
        console.print(f"  • Success rate: [green]{stats.success_rate:.1f}%[/green]")

    if stats.errors:
        console.print(f"\n[red]Errors ({len(stats.errors)}):[/red]")
        for error in stats.errors[:5]:  # Show first 5 errors
            console.print(f"  • {error.get('document_id', 'unknown')}: {error.get('error', 'Unknown error')}")
        if len(stats.errors) > 5:
            console.print(f"  ... and {len(stats.errors) - 5} more errors")
