"""Analyze field coverage in Weaviate LegalDocuments collection."""

from collections import defaultdict
from typing import Dict, List

from loguru import logger
from rich.console import Console
from rich.table import Table

from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase

console = Console()


def analyze_field_coverage(db: WeaviateLegalDocumentsDatabase, sample_size: int = 1000) -> Dict:
    """Analyze which fields are frequently empty in the database.

    Args:
        db: Weaviate database connection
        sample_size: Number of documents to sample for analysis

    Returns:
        Dict with field coverage statistics
    """
    collection = db.legal_documents_collection

    # Get total count
    total_response = collection.aggregate.over_all(total_count=True)
    total_count = total_response.total_count

    logger.info(f"Total documents in collection: {total_count}")

    # Fetch a sample of documents with all properties
    response = collection.query.fetch_objects(limit=min(sample_size, total_count))

    documents = [obj.properties for obj in response.objects]

    if not documents:
        logger.warning("No documents found in collection")
        return {}

    # Count non-empty fields
    field_counts = defaultdict(int)
    field_types = {}

    for doc in documents:
        for field, value in doc.items():
            # Check if field has meaningful content
            if value is not None:
                if isinstance(value, str):
                    if value.strip():  # Non-empty string
                        field_counts[field] += 1
                elif isinstance(value, list):
                    if value:  # Non-empty list
                        field_counts[field] += 1
                elif isinstance(value, (int, float)):
                    field_counts[field] += 1
                else:
                    field_counts[field] += 1

            # Track field type
            if field not in field_types:
                field_types[field] = type(value).__name__ if value is not None else "None"

    # Calculate coverage percentages
    sample_count = len(documents)
    field_stats = {}

    for field in sorted(field_counts.keys()):
        count = field_counts[field]
        coverage = (count / sample_count) * 100
        empty_count = sample_count - count

        field_stats[field] = {
            "populated_count": count,
            "empty_count": empty_count,
            "coverage_percentage": round(coverage, 2),
            "type": field_types.get(field, "unknown"),
        }

    # Add fields that are completely empty
    all_fields = set()
    for doc in documents:
        all_fields.update(doc.keys())

    for field in all_fields:
        if field not in field_stats:
            field_stats[field] = {
                "populated_count": 0,
                "empty_count": sample_count,
                "coverage_percentage": 0.0,
                "type": field_types.get(field, "unknown"),
            }

    return {
        "total_documents": total_count,
        "sample_size": sample_count,
        "field_stats": field_stats,
    }


def identify_llm_candidate_fields(field_stats: Dict) -> List[Dict]:
    """Identify fields that could be populated using LLMs.

    Args:
        field_stats: Field statistics from analyze_field_coverage

    Returns:
        List of candidate fields with their statistics
    """
    # Fields that are good candidates for LLM generation
    # Based on their semantic nature and low coverage
    llm_candidate_criteria = {
        "summary": "Generate concise summary of legal document",
        "thesis": "Extract main legal thesis/principle",
        "keywords": "Generate relevant keywords/tags",
        "outcome": "Summarize the outcome/decision",
        "legal_concepts": "Extract legal concepts discussed",
        "legal_references": "Extract and structure legal citations",
        "parties": "Extract and structure party information",
        "tags": "Generate semantic tags for searchability",
        "legal_analysis": "Generate structured legal analysis",
        "structured_content": "Create structured representation",
    }

    candidates = []

    for field, description in llm_candidate_criteria.items():
        if field in field_stats["field_stats"]:
            stats = field_stats["field_stats"][field]
            candidates.append({
                "field": field,
                "description": description,
                "coverage_percentage": stats["coverage_percentage"],
                "empty_count": stats["empty_count"],
                "populated_count": stats["populated_count"],
                "type": stats["type"],
            })

    # Sort by coverage (lowest first = most empty = highest priority)
    candidates.sort(key=lambda x: x["coverage_percentage"])

    return candidates


def print_results(field_stats: Dict, llm_candidates: List[Dict]):
    """Print analysis results in formatted tables."""

    # Print overall statistics
    console.print("\n[bold cyan]Overall Statistics[/bold cyan]")
    console.print(f"Total documents: {field_stats['total_documents']:,}")
    console.print(f"Sample size analyzed: {field_stats['sample_size']:,}")

    # Print all fields coverage table
    console.print("\n[bold cyan]All Fields Coverage[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan", width=30)
    table.add_column("Type", style="blue", width=15)
    table.add_column("Populated", justify="right", style="green")
    table.add_column("Empty", justify="right", style="red")
    table.add_column("Coverage %", justify="right", style="yellow")

    # Sort by coverage percentage
    sorted_fields = sorted(
        field_stats["field_stats"].items(),
        key=lambda x: x[1]["coverage_percentage"]
    )

    for field, stats in sorted_fields:
        coverage_color = "green" if stats["coverage_percentage"] > 50 else "yellow" if stats["coverage_percentage"] > 10 else "red"
        table.add_row(
            field,
            stats["type"],
            str(stats["populated_count"]),
            str(stats["empty_count"]),
            f"[{coverage_color}]{stats['coverage_percentage']:.1f}%[/{coverage_color}]",
        )

    console.print(table)

    # Print LLM candidate fields
    console.print("\n[bold cyan]LLM Generation Candidate Fields[/bold cyan]")
    console.print("(Fields that could be populated using LLMs, sorted by priority)\n")

    llm_table = Table(show_header=True, header_style="bold magenta")
    llm_table.add_column("Priority", justify="right", style="cyan", width=8)
    llm_table.add_column("Field", style="cyan", width=25)
    llm_table.add_column("Description", style="white", width=45)
    llm_table.add_column("Coverage %", justify="right", style="yellow", width=12)
    llm_table.add_column("Empty", justify="right", style="red", width=10)

    for idx, candidate in enumerate(llm_candidates, 1):
        priority_color = "red" if idx <= 3 else "yellow" if idx <= 6 else "green"
        llm_table.add_row(
            f"[{priority_color}]{idx}[/{priority_color}]",
            candidate["field"],
            candidate["description"],
            f"{candidate['coverage_percentage']:.1f}%",
            str(candidate["empty_count"]),
        )

    console.print(llm_table)


def main():
    """Main execution function."""
    logger.info("Connecting to Weaviate...")

    # Initialize database connection using context manager
    with WeaviateLegalDocumentsDatabase() as db:
        try:
            logger.info("Analyzing field coverage...")
            field_stats = analyze_field_coverage(db, sample_size=1000)

            logger.info("Identifying LLM candidate fields...")
            llm_candidates = identify_llm_candidate_fields(field_stats)

            print_results(field_stats, llm_candidates)

            # Print markdown table for documentation
            console.print("\n[bold cyan]Markdown Table for Documentation:[/bold cyan]\n")
            console.print("| Priority | Field | Description | Coverage % | Empty Count | LLM Generation Feasibility |")
            console.print("|----------|-------|-------------|------------|-------------|---------------------------|")
            for idx, candidate in enumerate(llm_candidates, 1):
                feasibility = "High" if idx <= 3 else "Medium" if idx <= 6 else "Low"
                console.print(
                    f"| {idx} | `{candidate['field']}` | {candidate['description']} | "
                    f"{candidate['coverage_percentage']:.1f}% | {candidate['empty_count']} | {feasibility} |"
                )

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise


if __name__ == "__main__":
    main()
