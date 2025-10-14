"""Run Gemini extraction using Weaviate REST API directly.

This script orchestrates document extraction from Weaviate using modular components:
- Schema: create_polish_legal_schema() from juddges.extraction.schema
- Client: WeaviateRestClient for all Weaviate operations
- Processor: BatchProcessor for parallel extraction
- Ingestion: WeaviateIngestionService for updating Weaviate
- Statistics: Comprehensive tracking and reporting
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLAlchemyMd5Cache
from langfuse.langchain import CallbackHandler
from loguru import logger
from rich.console import Console
from sqlalchemy import create_engine

from juddges.extraction import (
    BatchProcessor,
    ExtractionStorage,
    GeminiExtractionChain,
    WeaviateIngestionService,
    WeaviateRestClient,
    calculate_field_coverage,
    create_polish_legal_schema,
    display_extraction_results,
    display_ingestion_results,
    generate_extraction_summary,
    save_extraction_results,
    save_ingestion_report,
)
from juddges.settings import ROOT_PATH

# Load environment variables
load_dotenv(ROOT_PATH / ".env", override=True)

console = Console()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run Gemini extraction using Weaviate REST API")
    parser.add_argument(
        "--max-documents",
        type=int,
        default=5,
        help="Maximum number of documents to process",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        choices=[
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
        help="Gemini model to use (via Vertex AI)",
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
    parser.add_argument(
        "--weaviate-host",
        type=str,
        default=None,
        help="Weaviate host (defaults to env var WEAVIATE_HOST)",
    )
    parser.add_argument(
        "--weaviate-port",
        type=int,
        default=None,
        help="Weaviate port (defaults to env var WEAVIATE_PORT)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of documents to process in each batch (default: 5)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of parallel threads for batch processing (default: 5, max: 10)",
    )
    parser.add_argument(
        "--search-query",
        type=str,
        default=None,
        help="Optional search query for hybrid/semantic search (e.g., 'kredyt frankowy', 'VAT')",
    )
    parser.add_argument(
        "--document-type",
        type=str,
        default=None,
        choices=["judgment", "tax_interpretation"],
        help="Optional filter by document type (judgment or tax_interpretation)",
    )
    parser.add_argument(
        "--ingest-to-weaviate",
        action="store_true",
        help="Ingest extracted data back to Weaviate after extraction (updates existing documents)",
    )
    parser.add_argument(
        "--ingest-batch-size",
        type=int,
        default=50,
        help="Number of documents to ingest per batch when ingesting to Weaviate (default: 50)",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing non-empty values in Weaviate (default: only update empty fields)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable extended thinking mode for Gemini 2.5 models (default: disabled). "
        "Shows model's reasoning process before answering. Increases latency and token usage. "
        "Recommended for complex reasoning tasks, not for structured extraction.",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Get Weaviate connection details
    weaviate_host = args.weaviate_host or os.getenv("WEAVIATE_HOST", "localhost")
    weaviate_port = args.weaviate_port or int(os.getenv("WEAVIATE_PORT", "8084"))

    # Get GCP project
    vertex_project = os.getenv("VERTEX_PROJECT", "insbay-b32351")
    vertex_location = os.getenv("VERTEX_LOCATION", "us-central1")

    # Build filter display
    filter_display = []
    if args.search_query:
        filter_display.append(f"Search: '{args.search_query}'")
    if args.document_type:
        filter_display.append(f"Type: {args.document_type}")
    filter_str = "\n" + "\n".join(filter_display) if filter_display else ""

    console.print(
        f"\n[bold cyan]Gemini Extraction - Vertex AI Mode[/bold cyan]\n"
        f"Weaviate: {weaviate_host}:{weaviate_port}\n"
        f"GCP Project: {vertex_project}\n"
        f"GCP Location: {vertex_location}\n"
        f"Max documents: {args.max_documents}\n"
        f"Batch size: {args.batch_size}\n"
        f"Max workers: {args.max_workers} {'(parallel)' if args.max_workers > 1 else '(sequential)'}\n"
        f"Model: {args.model}\n"
        f"Thinking mode: {'[yellow]enabled[/yellow]' if args.enable_thinking else '[green]disabled[/green]'}\n"
        f"Output: {args.output_dir}\n"
        f"Random seed: {args.seed}{filter_str}\n"
    )

    # Initialize LangChain PostgreSQL Cache using SQLAlchemyMd5Cache
    postgres_cache_url = os.getenv("POSTGRES_CACHE_URL")
    if postgres_cache_url:
        try:
            logger.info(f"Initializing LangChain PostgreSQL cache (MD5) at {postgres_cache_url}...")
            engine = create_engine(postgres_cache_url)
            set_llm_cache(SQLAlchemyMd5Cache(engine=engine))
            console.print(
                f"[green]✓[/green] LangChain PostgreSQL cache (MD5) enabled ({postgres_cache_url})"
            )
            logger.info("LangChain PostgreSQL MD5 cache initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize LangChain cache: {e}")
            console.print(f"[yellow]Warning: LangChain cache initialization failed: {e}[/yellow]")
    else:
        console.print("[yellow]LangChain cache disabled (POSTGRES_CACHE_URL not set)[/yellow]")

    # Initialize extraction chain (Vertex AI uses application default credentials)
    logger.info("Initializing Vertex AI Gemini extraction chain...")
    chain = GeminiExtractionChain(
        model_name=args.model,
        project=vertex_project,
        location=vertex_location,
        cache_path=args.cache_path,
        temperature=0.0,
        enable_thinking=args.enable_thinking,
    )

    # Create schema using modular schema definition
    schema = create_polish_legal_schema()
    logger.info(f"Created extraction schema with {len(schema.fields)} fields")

    # Initialize extraction storage
    storage = None
    run_id = None
    try:
        storage = ExtractionStorage()
        logger.info("Initialized extraction storage (PostgreSQL)")
        console.print("[green]✓[/green] Extraction storage enabled (PostgreSQL)")
    except Exception as e:
        logger.warning(f"Failed to initialize extraction storage: {e}")
        console.print(f"[yellow]Warning: Extraction storage disabled - {e}[/yellow]")

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

    # Initialize Weaviate client using modular client
    api_key = os.getenv("WEAVIATE_API_KEY", "")
    client = WeaviateRestClient(host=weaviate_host, port=weaviate_port, api_key=api_key)

    # Fetch documents using REST API
    logger.info("Fetching documents from Weaviate...")
    documents = client.fetch_documents(
        max_documents=args.max_documents,
        search_query=args.search_query,
        document_type_filter=args.document_type,
        random_seed=args.seed,
    )

    if not documents:
        console.print("[red]No documents found for extraction![/red]")
        return

    # Create extraction run in database
    start_time = time.time()
    if storage:
        try:
            run_id = storage.create_extraction_run(
                model_name=args.model,
                sample_size=args.max_documents,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
                weaviate_host=weaviate_host,
                weaviate_port=weaviate_port,
                search_query=args.search_query,
                document_type_filter=args.document_type,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                temperature=0.0,
                prompt_template=schema.instructions,
                extraction_schema=schema.fields,
                random_seed=args.seed,
                notes=f"Extraction run with {args.max_documents} documents",
            )
            logger.info(f"Created extraction run: {run_id}")
            console.print(f"[cyan]Extraction run ID:[/cyan] {run_id}")
        except Exception as e:
            logger.error(f"Failed to create extraction run: {e}")
            console.print(f"[red]Failed to create extraction run: {e}[/red]")

    # Run extraction with parallel batch processing using BatchProcessor
    logger.info(
        f"Running extraction on {len(documents)} documents "
        f"(batch size: {args.batch_size}, workers: {args.max_workers})..."
    )
    processor = BatchProcessor(
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        show_progress=True,
    )
    extraction_results = processor.process_extraction_batches(
        documents=documents,
        chain=chain,
        schema=schema,
        langfuse_handler=langfuse_handler,
        storage=storage,
        run_id=run_id,
    )

    # Save results using modular save function
    output_dir = Path(args.output_dir)
    stats = save_extraction_results(documents, extraction_results, output_dir)

    # Display extraction results
    display_extraction_results(stats)

    # Complete extraction run in database
    if storage and run_id:
        try:
            duration = time.time() - start_time
            successful = sum(1 for r in extraction_results if r.get("extraction_status") == "success")
            failed = len(extraction_results) - successful

            storage.complete_extraction_run(
                run_id=run_id,
                total_documents=len(extraction_results),
                successful_extractions=successful,
                failed_extractions=failed,
                duration_seconds=duration,
            )

            # Save field coverage
            field_coverage_stats = calculate_field_coverage(extraction_results)
            if field_coverage_stats:
                # Convert to dictionary format expected by storage
                field_coverage_dict = {
                    field_name: {
                        "populated": coverage.populated,
                        "empty": coverage.empty,
                    }
                    for field_name, coverage in field_coverage_stats.items()
                }
                storage.save_field_coverage(run_id, field_coverage_dict)

            logger.info(f"Completed extraction run: {run_id}")
            console.print(f"[green]✓[/green] Extraction run completed and saved to database")
        except Exception as e:
            logger.error(f"Failed to complete extraction run: {e}")
            console.print(f"[red]Failed to complete extraction run: {e}[/red]")

    # Optional: Ingest extracted data back to Weaviate using WeaviateIngestionService
    if args.ingest_to_weaviate:
        console.print("\n[bold blue]Starting Weaviate ingestion...[/bold blue]")

        try:
            # Create ingestion service
            ingestion_service = WeaviateIngestionService(client=client)

            # Ingest results
            ingestion_stats = ingestion_service.ingest_results(
                extraction_results=extraction_results,
                batch_size=args.ingest_batch_size,
                skip_on_error=True,
                overwrite_existing=args.overwrite_existing,
                use_batch_api=True,
            )

            # Display results
            display_ingestion_results(ingestion_stats)

            # Log ingestion to database
            if storage and run_id:
                try:
                    storage.log_ingestion(
                        run_id=run_id,
                        batch_size=args.ingest_batch_size,
                        overwrite_existing=args.overwrite_existing,
                        total_documents=ingestion_stats.total_documents,
                        successful_updates=ingestion_stats.successful_updates,
                        failed_updates=ingestion_stats.failed_updates,
                        skipped_documents=ingestion_stats.skipped_documents,
                        duration_seconds=ingestion_stats.duration_seconds,
                        errors=ingestion_stats.errors,
                        status="completed",
                    )
                    logger.info(f"Logged ingestion for run: {run_id}")
                except Exception as e:
                    logger.error(f"Failed to log ingestion: {e}")

            # Save ingestion report
            save_ingestion_report(ingestion_stats, output_dir)
            console.print(f"\n[cyan]Ingestion report saved to:[/cyan] {output_dir / 'ingestion_report.json'}")

        except Exception as e:
            console.print(f"\n[red]✗ Ingestion failed: {e}[/red]")
            logger.exception("Ingestion error")
            raise

    console.print(f"\n[bold green]✓ Extraction complete![/bold green]")
    console.print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
