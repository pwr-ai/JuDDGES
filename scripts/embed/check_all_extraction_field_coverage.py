#!/usr/bin/env python3
"""Check coverage of all extraction-related fields in Weaviate LegalDocuments collection.

This script analyzes how many documents have populated values for each field
that is mapped from the Gemini extraction schema to Weaviate.

Uses the REST client directly to avoid import issues with the main Weaviate database class.

Fields checked (from juddges/extraction/field_mapping.py):
- Direct TEXT: title, summary, thesis, factual_state, legal_state
- TEXT_ARRAY: keywords
- TEXT (JSON): outcome, legal_references, legal_concepts, parties,
               legal_analysis, judgment_specific, tax_interpretation_specific
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(env_path, override=True)

console = Console()

# All fields mapped from extraction schema to Weaviate
# Based on juddges/extraction/field_mapping.py
EXTRACTION_FIELDS = [
    # Direct TEXT fields
    "title",
    "summary",
    "thesis",
    "factual_state",
    "legal_state",
    # TEXT_ARRAY field
    "keywords",
    # TEXT (JSON) fields
    "outcome",
    "legal_references",
    "legal_concepts",
    "parties",
    "legal_analysis",
    "judgment_specific",
    "tax_interpretation_specific",
]

# Fields that are always expected to be populated (from original data, not extraction)
CORE_FIELDS = [
    "document_id",
    "document_number",
    "document_type",
]


@dataclass
class FieldCoverageStats:
    """Statistics for a single field."""

    field_name: str
    populated_count: int = 0
    empty_count: int = 0
    total_checked: int = 0

    @property
    def coverage_percentage(self) -> float:
        if self.total_checked == 0:
            return 0.0
        return (self.populated_count / self.total_checked) * 100


@dataclass
class CoverageReport:
    """Complete coverage report for all fields."""

    total_documents: int = 0
    documents_checked: int = 0
    field_stats: dict[str, FieldCoverageStats] = field(default_factory=dict)
    check_duration_seconds: float = 0.0

    def add_field(self, field_name: str) -> None:
        self.field_stats[field_name] = FieldCoverageStats(field_name=field_name)

    def record_populated(self, field_name: str) -> None:
        if field_name in self.field_stats:
            self.field_stats[field_name].populated_count += 1
            self.field_stats[field_name].total_checked += 1

    def record_empty(self, field_name: str) -> None:
        if field_name in self.field_stats:
            self.field_stats[field_name].empty_count += 1
            self.field_stats[field_name].total_checked += 1


def is_field_populated(value: Any) -> bool:
    """Check if a field value is considered populated (non-empty).

    Args:
        value: The field value to check

    Returns:
        True if the field has meaningful content, False otherwise
    """
    if value is None:
        return False

    if isinstance(value, str):
        # Check for non-empty string (strip whitespace)
        stripped = value.strip()
        # Also check for empty JSON representations
        if stripped in ("", "null", "[]", "{}", '""'):
            return False
        return True

    if isinstance(value, list):
        # Check for non-empty list with at least one non-empty item
        return any(item and str(item).strip() for item in value)

    if isinstance(value, dict):
        # Check for non-empty dict with at least one non-null value
        return any(v is not None and v != "" for v in value.values())

    # For other types (numbers, booleans), consider them populated
    return True


class WeaviateRestClient:
    """Simple REST client for Weaviate operations."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8084,
        api_key: str | None = None,
        timeout: int = 60,
    ):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"

        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    @classmethod
    def from_env(cls) -> "WeaviateRestClient":
        """Create client from environment variables."""
        return cls(
            host=os.getenv("WEAVIATE_HOST", "localhost"),
            port=int(os.getenv("WEAVIATE_PORT", "8084")),
            api_key=os.getenv("WEAVIATE_API_KEY"),
        )

    def get_total_count(self) -> int:
        """Get total document count using aggregate query."""
        query = """
        {
            Aggregate {
                LegalDocuments {
                    meta {
                        count
                    }
                }
            }
        }
        """
        response = requests.post(
            f"{self.base_url}/v1/graphql",
            headers=self.headers,
            json={"query": query},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        try:
            return data["data"]["Aggregate"]["LegalDocuments"][0]["meta"]["count"]
        except (KeyError, IndexError, TypeError):
            logger.error(f"Unexpected response format: {data}")
            return 0

    def fetch_documents_batch(
        self,
        properties: list[str],
        limit: int = 500,
        cursor: str | None = None,
    ) -> tuple[list[dict], str | None]:
        """Fetch a batch of documents using cursor-based pagination.

        Args:
            properties: List of property names to fetch
            limit: Maximum documents per batch
            cursor: Cursor for pagination (None for first batch)

        Returns:
            Tuple of (list of documents, next cursor or None)
        """
        # Build properties string for GraphQL
        props_str = " ".join(properties)

        # Build cursor part of query
        if cursor:
            after_clause = f', after: "{cursor}"'
        else:
            after_clause = ""

        query = f"""
        {{
            Get {{
                LegalDocuments(limit: {limit}{after_clause}) {{
                    {props_str}
                    _additional {{
                        id
                    }}
                }}
            }}
        }}
        """

        response = requests.post(
            f"{self.base_url}/v1/graphql",
            headers=self.headers,
            json={"query": query},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        # Check for errors
        if "errors" in data:
            logger.error(f"GraphQL errors: {data['errors']}")
            return [], None

        try:
            documents = data["data"]["Get"]["LegalDocuments"]
            if not documents:
                return [], None

            # Get the last document's ID for cursor
            next_cursor = documents[-1]["_additional"]["id"]

            # Extract properties (remove _additional from results)
            cleaned_docs = []
            for doc in documents:
                cleaned = {k: v for k, v in doc.items() if k != "_additional"}
                cleaned_docs.append(cleaned)

            return cleaned_docs, next_cursor

        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Unexpected response format: {e}")
            return [], None


def check_field_coverage(
    client: WeaviateRestClient,
    fields_to_check: list[str],
    batch_size: int = 500,
    limit: int | None = None,
) -> CoverageReport:
    """Check field coverage across all documents in the collection.

    Args:
        client: Weaviate REST client
        fields_to_check: List of field names to check
        batch_size: Number of documents to fetch per batch
        limit: Optional limit on total documents to check (None = all)

    Returns:
        CoverageReport with statistics for each field
    """
    report = CoverageReport()

    # Initialize field stats
    for field_name in fields_to_check:
        report.add_field(field_name)

    # Get total count
    logger.info("Getting total document count...")
    report.total_documents = client.get_total_count()

    if limit:
        docs_to_check = min(limit, report.total_documents)
    else:
        docs_to_check = report.total_documents

    logger.info(f"Total documents in collection: {report.total_documents:,}")
    logger.info(f"Documents to check: {docs_to_check:,}")
    logger.info(f"Fields to check: {len(fields_to_check)}")

    start_time = time.time()

    # Use cursor-based pagination for efficiency
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed:,}/{task.total:,})"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Checking field coverage...", total=docs_to_check)

        cursor = None
        documents_processed = 0

        while documents_processed < docs_to_check:
            try:
                # Calculate batch size for this iteration
                remaining = docs_to_check - documents_processed
                current_batch_size = min(batch_size, remaining)

                # Fetch batch using cursor
                documents, next_cursor = client.fetch_documents_batch(
                    properties=fields_to_check,
                    limit=current_batch_size,
                    cursor=cursor,
                )

                if not documents:
                    logger.info("No more documents to process")
                    break

                # Process each document in the batch
                for doc in documents:
                    # Check each field
                    for field_name in fields_to_check:
                        value = doc.get(field_name)

                        if is_field_populated(value):
                            report.record_populated(field_name)
                        else:
                            report.record_empty(field_name)

                    documents_processed += 1
                    report.documents_checked += 1

                # Update cursor for next batch
                cursor = next_cursor

                progress.update(task, completed=documents_processed)

                # Log progress every 10k documents
                if documents_processed % 10000 == 0:
                    logger.info(f"Processed {documents_processed:,} documents...")

            except Exception as e:
                logger.error(f"Error fetching documents: {e}")
                break

    report.check_duration_seconds = time.time() - start_time

    return report


def print_coverage_report(report: CoverageReport) -> None:
    """Print a formatted coverage report to the console.

    Args:
        report: The coverage report to display
    """
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Weaviate Extraction Field Coverage Report[/bold cyan]",
            border_style="cyan",
        )
    )

    # Summary stats
    console.print("\n[bold]Collection Statistics:[/bold]")
    console.print(f"  Total documents in collection: {report.total_documents:,}")
    console.print(f"  Documents checked: {report.documents_checked:,}")
    console.print(f"  Check duration: {report.check_duration_seconds:.1f} seconds")
    console.print(f"  Fields analyzed: {len(report.field_stats)}")

    # Create coverage table
    table = Table(
        title="\nField Coverage Analysis",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Field Name", style="cyan", width=30)
    table.add_column("Populated", justify="right", style="green")
    table.add_column("Empty", justify="right", style="red")
    table.add_column("Coverage %", justify="right")
    table.add_column("Status", justify="center")

    # Sort by coverage percentage (descending)
    sorted_stats = sorted(
        report.field_stats.values(),
        key=lambda x: x.coverage_percentage,
        reverse=True,
    )

    for stats in sorted_stats:
        # Determine status emoji based on coverage
        coverage = stats.coverage_percentage
        if coverage >= 80:
            status = "[green]High[/green]"
            coverage_style = "green"
        elif coverage >= 50:
            status = "[yellow]Medium[/yellow]"
            coverage_style = "yellow"
        elif coverage > 0:
            status = "[red]Low[/red]"
            coverage_style = "red"
        else:
            status = "[dim]None[/dim]"
            coverage_style = "dim"

        table.add_row(
            stats.field_name,
            f"{stats.populated_count:,}",
            f"{stats.empty_count:,}",
            f"[{coverage_style}]{coverage:.2f}%[/{coverage_style}]",
            status,
        )

    console.print(table)

    # Summary by category
    high_coverage = [s for s in sorted_stats if s.coverage_percentage >= 80]
    medium_coverage = [s for s in sorted_stats if 50 <= s.coverage_percentage < 80]
    low_coverage = [s for s in sorted_stats if 0 < s.coverage_percentage < 50]
    no_coverage = [s for s in sorted_stats if s.coverage_percentage == 0]

    console.print("\n[bold]Coverage Summary:[/bold]")
    console.print(f"  [green]High coverage (>=80%):[/green] {len(high_coverage)} fields")
    console.print(f"  [yellow]Medium coverage (50-79%):[/yellow] {len(medium_coverage)} fields")
    console.print(f"  [red]Low coverage (<50%):[/red] {len(low_coverage)} fields")
    console.print(f"  [dim]No coverage (0%):[/dim] {len(no_coverage)} fields")

    if no_coverage:
        console.print("\n[bold red]Fields with NO data:[/bold red]")
        for stats in no_coverage:
            console.print(f"  - {stats.field_name}")

    # Calculate overall extraction coverage
    extraction_fields_with_data = sum(1 for s in sorted_stats if s.coverage_percentage > 0)
    console.print(
        f"\n[bold]Overall:[/bold] {extraction_fields_with_data}/{len(sorted_stats)} "
        f"extraction fields have at least some data"
    )


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Check extraction field coverage in Weaviate")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to check (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for fetching documents (default: 500)",
    )
    parser.add_argument(
        "--include-core",
        action="store_true",
        help="Also check core fields (document_id, document_number, etc.)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Weaviate host (default: from WEAVIATE_HOST env var or localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Weaviate port (default: from WEAVIATE_PORT env var or 8084)",
    )

    args = parser.parse_args()

    # Determine which fields to check
    fields_to_check = EXTRACTION_FIELDS.copy()
    if args.include_core:
        fields_to_check = CORE_FIELDS + fields_to_check

    console.print(
        Panel.fit(
            "Weaviate Extraction Field Coverage Check",
            style="bold blue",
            border_style="bright_blue",
        )
    )

    # Create client
    if args.host or args.port:
        client = WeaviateRestClient(
            host=args.host or os.getenv("WEAVIATE_HOST", "localhost"),
            port=args.port or int(os.getenv("WEAVIATE_PORT", "8084")),
            api_key=os.getenv("WEAVIATE_API_KEY"),
        )
    else:
        client = WeaviateRestClient.from_env()

    console.print(f"\n[cyan]Configuration:[/cyan]")
    console.print(f"  - Weaviate: {client.host}:{client.port}")
    console.print(f"  - Fields to check: {len(fields_to_check)}")
    console.print(f"  - Batch size: {args.batch_size}")
    console.print(f"  - Document limit: {args.limit or 'all'}")

    logger.info("Connecting to Weaviate...")

    try:
        # Test connection
        total = client.get_total_count()
        if total == 0:
            console.print("[yellow]Warning: No documents found or connection failed[/yellow]")

        console.print(f"[green]Connected to Weaviate ({total:,} documents)[/green]\n")

        report = check_field_coverage(
            client=client,
            fields_to_check=fields_to_check,
            batch_size=args.batch_size,
            limit=args.limit,
        )

        print_coverage_report(report)

    except requests.exceptions.ConnectionError as e:
        console.print(f"\n[bold red]Connection Error: Could not connect to Weaviate at {client.base_url}[/bold red]")
        console.print("[yellow]Make sure Weaviate is running and accessible.[/yellow]")
        logger.error(f"Connection error: {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        logger.exception("Coverage check failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
