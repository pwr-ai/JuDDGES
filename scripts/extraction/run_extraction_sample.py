"""Run Gemini extraction on random sample of legal documents.

This script:
1. Randomly samples documents from Weaviate
2. Extracts information using the comprehensive schema
3. Saves full_text and extracted results to separate files
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any

# Set Weaviate connection environment variables BEFORE importing modules
# Override docker service name with localhost for running on host
if not os.getenv("WEAVIATE_HOST") or os.getenv("WEAVIATE_HOST") == "weaviate":
    os.environ["WEAVIATE_HOST"] = "127.0.0.1"

if not os.getenv("WEAVIATE_PORT"):
    os.environ["WEAVIATE_PORT"] = "8084"

if not os.getenv("WEAVIATE_GRPC_PORT"):
    os.environ["WEAVIATE_GRPC_PORT"] = "50051"

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from langfuse.langchain import CallbackHandler

from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase
from juddges.extraction import GeminiExtractionChain
from juddges.extraction.gemini_chain import DocumentType, ExtractionSchema

console = Console()


def create_comprehensive_schema() -> ExtractionSchema:
    """Create comprehensive extraction schema based on Weaviate properties."""
    return ExtractionSchema(
        fields={
            # Core document identification
            "document_number": "string, official case/document reference number",
            "document_type": "string, type: 'judgment', 'tax_interpretation', or 'legal_act'",
            "title": "string, document title or generated descriptive title",
            "date_issued": "date ISO 8601 (YYYY-MM-DD), date document was issued",

            # High-priority augmentation fields
            "summary": "string, concise 3-5 sentence summary covering: (1) document type and issuing body, (2) main legal issue, (3) key facts, (4) decision/outcome, (5) primary legal basis",
            "thesis": "string, main legal principle or rule established by the document in 1-3 sentences",
            "keywords": "List[string], 5-15 relevant legal keywords covering legal domains, institutions, and specific concepts",

            # Outcome
            "outcome": """JSON object: {
                "decision_type": "string (uwzględniono/oddalono/uchylono/pozytywne/negatywne)",
                "decision_summary": "string, brief summary of decision",
                "awarded_amounts": [{"type": "string", "amount": number, "currency": "string", "recipient": "string"}],
                "legal_effect": "string, practical legal consequence"
            }""",

            # Legal content
            "legal_references": """JSON array: [{
                "type": "string (statute/article/case_law/regulation)",
                "title": "string, title of legal source",
                "article": "string, specific article/section",
                "jurisdiction": "string, jurisdiction (Poland/EU/etc)",
                "citation": "string, full citation as in document",
                "context": "string, how reference is used"
            }]""",

            "legal_concepts": """JSON array: [{
                "concept_name": "string, name of legal concept",
                "legal_area": "string, area of law",
                "definition_context": "string, how concept is used",
                "relevance": "string (primary/secondary/mentioned)"
            }]""",

            "parties": """JSON array: [{
                "party_type": "string (plaintiff/defendant/applicant/etc)",
                "party_name": "string, name (anonymized if needed)",
                "party_category": "string (natural_person/company/public_entity)",
                "representation": "string, legal representative if available"
            }]""",

            # Structured content
            "legal_analysis": """JSON object: {
                "facts_summary": "string, key factual findings",
                "legal_issues": ["string, main legal questions"],
                "reasoning": "string, court's/authority's reasoning",
                "conclusion": "string, final legal conclusion"
            }""",

            # Document-type specific fields
            "judgment_specific": """JSON object (if judgment): {
                "court_name": "string, full court name",
                "court_type": "string (district/regional/appeal/supreme)",
                "department_name": "string, court department",
                "judges": [{"name": "string", "role": "string (presiding/member)"}],
                "legal_bases": ["string, legal basis for decision"],
                "judgment_type": "string (wyrok/postanowienie)"
            }""",

            "tax_interpretation_specific": """JSON object (if tax interpretation): {
                "interpretation_type": "string (individual/general)",
                "tax_authority": "string, issuing authority",
                "tax_matter": "string, specific tax question",
                "tax_type": "string (VAT/CIT/PIT/etc)"
            }""",
        },
        instructions="""
Extract factual information from the legal document.
- Use ISO 8601 format (YYYY-MM-DD) for all dates
- Return valid JSON for complex objects (outcome, parties, legal_references, etc.)
- Maintain original document language (Polish/English)
- Use empty string "" for missing simple fields
- Use empty array [] for missing list fields
- Use null for missing complex objects
- Extract ALL legal citations with complete information
- Generate comprehensive keywords covering all relevant legal concepts
        """,
        language="polish",
    )


def sample_documents(
    db: WeaviateLegalDocumentsDatabase, sample_size: int = 50
) -> List[Dict[str, Any]]:
    """Sample random documents with full_text from Weaviate.

    Args:
        db: Weaviate database connection
        sample_size: Number of documents to sample

    Returns:
        List of document properties
    """
    collection = db.legal_documents_collection

    # Fetch documents with full_text
    # Use larger fetch size to ensure we get enough valid documents
    # Avoid aggregate.over_all() which uses GRPC
    fetch_size = sample_size * 5  # Fetch 5x more to filter for valid full_text

    logger.info(f"Fetching up to {fetch_size} documents from Weaviate (REST API)...")

    try:
        response = collection.query.fetch_objects(
            limit=fetch_size,
            return_properties=["document_id", "document_type", "full_text", "language", "document_number"],
        )
    except Exception as e:
        logger.error(f"Failed to fetch documents: {e}")
        raise

    # Filter documents with non-empty full_text
    valid_docs = []
    for obj in response.objects:
        props = obj.properties
        full_text = props.get("full_text", "")
        if full_text and len(full_text.strip()) > 100:  # At least 100 chars
            valid_docs.append(props)

    logger.info(f"Found {len(valid_docs)} documents with valid full_text")

    if not valid_docs:
        logger.warning("No documents with valid full_text found!")
        return []

    # Random sample
    sample = random.sample(valid_docs, min(sample_size, len(valid_docs)))

    logger.info(f"Sampled {len(sample)} documents for extraction")

    return sample


def run_extraction(
    documents: List[Dict[str, Any]],
    chain: GeminiExtractionChain,
    schema: ExtractionSchema,
    langfuse_handler=None,
) -> List[Dict[str, Any]]:
    """Run extraction on sampled documents.

    Args:
        documents: List of document properties
        chain: Extraction chain
        schema: Extraction schema
        langfuse_handler: Optional Langfuse callback handler for observability

    Returns:
        List of extraction results with metadata
    """
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:

        task = progress.add_task("Extracting documents...", total=len(documents))

        for doc in documents:
            document_id = doc.get("document_id", "unknown")
            full_text = doc.get("full_text", "")
            doc_type_str = doc.get("document_type", "judgment")

            # Map document type
            if "interpret" in doc_type_str.lower():
                doc_type = DocumentType.TAX_INTERPRETATION
            else:
                doc_type = DocumentType.JUDGMENT

            try:
                # Run extraction
                extracted = chain.extract(
                    document_type=doc_type,
                    text=full_text,
                    schema=schema,
                    langfuse_handler=langfuse_handler,
                    max_text_length=150000,
                )

                # Add metadata
                result = {
                    "document_id": document_id,
                    "document_type": doc_type_str,
                    "extraction_status": "success",
                    "extracted_data": extracted,
                    "full_text_length": len(full_text),
                    "source_language": doc.get("language", "unknown"),
                }

                results.append(result)

                logger.info(
                    f"✓ Extracted {document_id} ({len(extracted)} fields)"
                )

            except Exception as e:
                logger.error(f"✗ Failed to extract {document_id}: {e}")
                results.append({
                    "document_id": document_id,
                    "document_type": doc_type_str,
                    "extraction_status": "failed",
                    "error": str(e),
                    "full_text_length": len(full_text),
                })

            progress.update(task, advance=1)

    return results


def save_results(
    documents: List[Dict[str, Any]],
    extraction_results: List[Dict[str, Any]],
    output_dir: Path,
):
    """Save full_text and extraction results to separate files.

    Args:
        documents: Original documents with full_text
        extraction_results: Extraction results
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full_text documents
    full_text_file = output_dir / "sample_documents_full_text.jsonl"
    with open(full_text_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(documents)} full_text documents to {full_text_file}")

    # Save extraction results
    extracted_file = output_dir / "sample_documents_extracted.jsonl"
    with open(extracted_file, "w", encoding="utf-8") as f:
        for result in extraction_results:
            f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    logger.info(f"Saved {len(extraction_results)} extraction results to {extracted_file}")

    # Save summary statistics
    summary_file = output_dir / "extraction_summary.json"

    successful = sum(1 for r in extraction_results if r.get("extraction_status") == "success")
    failed = len(extraction_results) - successful

    # Analyze field coverage
    field_coverage = {}
    for result in extraction_results:
        if result.get("extraction_status") == "success":
            extracted_data = result.get("extracted_data", {})
            for field, value in extracted_data.items():
                if field not in field_coverage:
                    field_coverage[field] = {"populated": 0, "empty": 0}

                # Check if field is populated
                if value:
                    if isinstance(value, str) and value.strip():
                        field_coverage[field]["populated"] += 1
                    elif isinstance(value, list) and value:
                        field_coverage[field]["populated"] += 1
                    elif isinstance(value, dict) and value:
                        field_coverage[field]["populated"] += 1
                    else:
                        field_coverage[field]["empty"] += 1
                else:
                    field_coverage[field]["empty"] += 1

    summary = {
        "total_documents": len(documents),
        "successful_extractions": successful,
        "failed_extractions": failed,
        "success_rate": f"{(successful / len(extraction_results) * 100):.1f}%",
        "field_coverage": field_coverage,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved extraction summary to {summary_file}")

    # Print summary to console
    console.print("\n[bold cyan]Extraction Summary[/bold cyan]")
    console.print(f"Total documents: {len(documents)}")
    console.print(f"[green]Successful: {successful}[/green]")
    console.print(f"[red]Failed: {failed}[/red]")
    console.print(f"Success rate: {summary['success_rate']}")

    console.print("\n[bold cyan]Field Coverage[/bold cyan]")
    for field, stats in sorted(field_coverage.items()):
        total = stats["populated"] + stats["empty"]
        coverage = (stats["populated"] / total * 100) if total > 0 else 0
        console.print(f"  {field}: {coverage:.1f}% ({stats['populated']}/{total})")


def main():
    """Main execution function."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Run Gemini extraction on random sample of documents"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of documents to sample",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        choices=["gemini-2.5-pro", "gemini-2.5-flash"],
        help="Gemini model to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/extraction_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=".cache/extraction_sample.db",
        help="Path to SQLite cache",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    console.print(
        f"\n[bold cyan]Gemini Extraction - Random Sample[/bold cyan]\n"
        f"Sample size: {args.sample_size}\n"
        f"Model: {args.model}\n"
        f"Output: {args.output_dir}\n"
        f"Random seed: {args.seed}\n"
    )

    # Initialize extraction chain
    logger.info("Initializing Gemini extraction chain...")
    chain = GeminiExtractionChain(
        model_name=args.model,
        cache_path=args.cache_path,
        temperature=0.0,
    )

    # Create schema
    schema = create_comprehensive_schema()
    logger.info(f"Created extraction schema with {len(schema.fields)} fields")

    # Initialize Langfuse (enable by default if keys are available)
    langfuse_handler = None
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            langfuse_handler = CallbackHandler()
            console.print(
                f"[green]✓[/green] Langfuse tracing enabled "
                f"(host: {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')})"
            )
            logger.info("Langfuse tracing initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse: {e}")
            console.print(f"[yellow]Warning: Langfuse initialization failed: {e}[/yellow]")
    else:
        console.print("[yellow]Langfuse tracing disabled (keys not set)[/yellow]")

    # Connect to Weaviate and sample documents
    logger.info("Connecting to Weaviate...")
    with WeaviateLegalDocumentsDatabase() as db:
        documents = sample_documents(db, sample_size=args.sample_size)

    if not documents:
        console.print("[red]No documents found for extraction![/red]")
        return

    # Run extraction
    logger.info(f"Running extraction on {len(documents)} documents...")
    extraction_results = run_extraction(documents, chain, schema, langfuse_handler)

    # Save results
    output_dir = Path(args.output_dir)
    save_results(documents, extraction_results, output_dir)

    console.print(f"\n[bold green]✓ Extraction complete![/bold green]")
    console.print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
