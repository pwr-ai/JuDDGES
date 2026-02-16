"""PostgreSQL storage for extraction inputs, outputs, and metadata.

This module provides database operations for storing extraction runs with full
traceability, supporting:
- Parallel execution safety (ACID transactions)
- Complete audit trail (prompts, parameters, timestamps)
- Easy analysis and review
- Ingestion support with full metadata
"""

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


class ExtractionStorage:
    """PostgreSQL storage for extraction runs and results."""

    def __init__(
        self,
        postgres_url: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """Initialize extraction storage.

        Args:
            postgres_url: Full PostgreSQL connection URL (takes precedence)
            host: PostgreSQL host (default: from env EXTRACTION_POSTGRES_HOST)
            port: PostgreSQL port (default: from env EXTRACTION_POSTGRES_PORT)
            user: PostgreSQL user (default: from env EXTRACTION_POSTGRES_USER)
            password: PostgreSQL password (default: from env EXTRACTION_POSTGRES_PASSWORD)
            database: Database name (default: from env EXTRACTION_POSTGRES_DB)
        """
        if postgres_url:
            self.connection_url = postgres_url
        else:
            host = host or os.getenv("EXTRACTION_POSTGRES_HOST", "localhost")
            port = port or int(os.getenv("EXTRACTION_POSTGRES_PORT", "5433"))
            user = user or os.getenv("EXTRACTION_POSTGRES_USER", "extraction_user")
            password = password or os.getenv("EXTRACTION_POSTGRES_PASSWORD", "extraction_pass")
            database = database or os.getenv("EXTRACTION_POSTGRES_DB", "legal_extraction")

            self.connection_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        self.engine: Engine = create_engine(
            self.connection_url,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=10,
            max_overflow=20,
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        logger.info(f"ExtractionStorage initialized with connection to {database}")

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_extraction_run(
        self,
        model_name: str,
        sample_size: int,
        batch_size: int,
        max_workers: int,
        weaviate_host: str,
        weaviate_port: int,
        search_query: Optional[str] = None,
        document_type_filter: Optional[str] = None,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        temperature: Optional[float] = None,
        prompt_template: Optional[str] = None,
        extraction_schema: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> UUID:
        """Create a new extraction run record.

        Args:
            model_name: Gemini model name
            sample_size: Number of documents sampled
            batch_size: Batch size for processing
            max_workers: Number of parallel workers
            weaviate_host: Weaviate host
            weaviate_port: Weaviate port
            search_query: Optional search query
            document_type_filter: Optional document type filter
            vertex_project: GCP project ID
            vertex_location: GCP location
            temperature: Model temperature
            prompt_template: Full prompt template
            extraction_schema: Complete extraction schema
            random_seed: Random seed for reproducibility
            notes: Optional notes

        Returns:
            UUID of the created extraction run
        """
        run_id = uuid4()

        with self.session_scope() as session:
            session.execute(
                text(
                    """
                INSERT INTO extraction_runs (
                    run_id, search_query, document_type_filter,
                    model_name, vertex_project, vertex_location, temperature,
                    prompt_template, extraction_schema,
                    sample_size, batch_size, max_workers,
                    weaviate_host, weaviate_port,
                    random_seed, notes
                ) VALUES (
                    :run_id, :search_query, :document_type_filter,
                    :model_name, :vertex_project, :vertex_location, :temperature,
                    :prompt_template, :extraction_schema,
                    :sample_size, :batch_size, :max_workers,
                    :weaviate_host, :weaviate_port,
                    :random_seed, :notes
                )
                """
                ),
                {
                    "run_id": run_id,
                    "search_query": search_query,
                    "document_type_filter": document_type_filter,
                    "model_name": model_name,
                    "vertex_project": vertex_project,
                    "vertex_location": vertex_location,
                    "temperature": temperature,
                    "prompt_template": prompt_template,
                    "extraction_schema": json.dumps(extraction_schema) if extraction_schema else None,
                    "sample_size": sample_size,
                    "batch_size": batch_size,
                    "max_workers": max_workers,
                    "weaviate_host": weaviate_host,
                    "weaviate_port": weaviate_port,
                    "random_seed": random_seed,
                    "notes": notes,
                },
            )

        logger.info(f"Created extraction run: {run_id}")
        return run_id

    def save_extraction_result(
        self,
        run_id: UUID,
        document_id: str,
        document_number: Optional[str],
        document_type: str,
        full_text: str,
        extraction_status: str,
        extracted_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        processing_time_seconds: Optional[float] = None,
        source_language: Optional[str] = None,
    ):
        """Save a single extraction result.

        Args:
            run_id: Extraction run UUID
            document_id: Weaviate document UUID
            document_number: Human-readable document number
            document_type: Document type
            full_text: Complete document text
            extraction_status: Status (success/failed/skipped)
            extracted_data: Extracted data as dict
            error_message: Error message if failed
            error_type: Error type if failed
            processing_time_seconds: Processing time
            source_language: Source language
        """
        with self.session_scope() as session:
            session.execute(
                text(
                    """
                INSERT INTO extraction_results (
                    run_id, document_id, document_number, document_type,
                    full_text, full_text_length, source_language,
                    extraction_status, extracted_data,
                    error_message, error_type, processing_time_seconds
                ) VALUES (
                    :run_id, :document_id, :document_number, :document_type,
                    :full_text, :full_text_length, :source_language,
                    :extraction_status, :extracted_data,
                    :error_message, :error_type, :processing_time_seconds
                )
                ON CONFLICT (run_id, document_id) DO UPDATE SET
                    extraction_status = EXCLUDED.extraction_status,
                    extracted_data = EXCLUDED.extracted_data,
                    error_message = EXCLUDED.error_message,
                    error_type = EXCLUDED.error_type,
                    processing_time_seconds = EXCLUDED.processing_time_seconds,
                    extracted_at = NOW()
                """
                ),
                {
                    "run_id": run_id,
                    "document_id": document_id,
                    "document_number": document_number,
                    "document_type": document_type,
                    "full_text": full_text,
                    "full_text_length": len(full_text) if full_text else 0,
                    "source_language": source_language,
                    "extraction_status": extraction_status,
                    "extracted_data": json.dumps(extracted_data) if extracted_data else None,
                    "error_message": error_message,
                    "error_type": error_type,
                    "processing_time_seconds": processing_time_seconds,
                },
            )

    def save_extraction_results_batch(
        self, run_id: UUID, results: List[Dict[str, Any]]
    ) -> Tuple[int, int]:
        """Save multiple extraction results in a single transaction.

        Args:
            run_id: Extraction run UUID
            results: List of result dictionaries

        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful = 0
        failed = 0

        with self.session_scope() as session:
            for result in results:
                try:
                    session.execute(
                        text(
                            """
                        INSERT INTO extraction_results (
                            run_id, document_id, document_number, document_type,
                            full_text, full_text_length, source_language,
                            extraction_status, extracted_data,
                            error_message, error_type, processing_time_seconds
                        ) VALUES (
                            :run_id, :document_id, :document_number, :document_type,
                            :full_text, :full_text_length, :source_language,
                            :extraction_status, :extracted_data,
                            :error_message, :error_type, :processing_time_seconds
                        )
                        ON CONFLICT (run_id, document_id) DO UPDATE SET
                            extraction_status = EXCLUDED.extraction_status,
                            extracted_data = EXCLUDED.extracted_data,
                            error_message = EXCLUDED.error_message,
                            error_type = EXCLUDED.error_type,
                            processing_time_seconds = EXCLUDED.processing_time_seconds,
                            extracted_at = NOW()
                        """
                        ),
                        {
                            "run_id": run_id,
                            "document_id": result["document_id"],
                            "document_number": result.get("document_number"),
                            "document_type": result["document_type"],
                            "full_text": result["full_text"],
                            "full_text_length": result.get("full_text_length", 0),
                            "source_language": result.get("source_language"),
                            "extraction_status": result["extraction_status"],
                            "extracted_data": json.dumps(result.get("extracted_data"))
                            if result.get("extracted_data")
                            else None,
                            "error_message": result.get("error"),
                            "error_type": result.get("error_type"),
                            "processing_time_seconds": result.get("processing_time_seconds"),
                        },
                    )

                    if result["extraction_status"] == "success":
                        successful += 1
                    else:
                        failed += 1

                except Exception as e:
                    logger.error(
                        f"Failed to save result for {result.get('document_id')}: {e}"
                    )
                    failed += 1

        logger.info(
            f"Saved {successful + failed} results for run {run_id} ({successful} successful, {failed} failed)"
        )
        return successful, failed

    def complete_extraction_run(
        self,
        run_id: UUID,
        total_documents: int,
        successful_extractions: int,
        failed_extractions: int,
        duration_seconds: float,
    ):
        """Mark extraction run as completed with statistics.

        Args:
            run_id: Extraction run UUID
            total_documents: Total documents processed
            successful_extractions: Count of successful extractions
            failed_extractions: Count of failed extractions
            duration_seconds: Total duration in seconds
        """
        with self.session_scope() as session:
            session.execute(
                text(
                    """
                UPDATE extraction_runs SET
                    total_documents = :total_documents,
                    successful_extractions = :successful_extractions,
                    failed_extractions = :failed_extractions,
                    completed_at = NOW(),
                    duration_seconds = :duration_seconds
                WHERE run_id = :run_id
                """
                ),
                {
                    "run_id": run_id,
                    "total_documents": total_documents,
                    "successful_extractions": successful_extractions,
                    "failed_extractions": failed_extractions,
                    "duration_seconds": duration_seconds,
                },
            )

        logger.info(f"Completed extraction run: {run_id}")

    def save_field_coverage(self, run_id: UUID, field_coverage: Dict[str, Dict[str, int]]):
        """Save field coverage statistics.

        Args:
            run_id: Extraction run UUID
            field_coverage: Dict mapping field names to {populated, empty} counts
        """
        with self.session_scope() as session:
            for field_name, stats in field_coverage.items():
                session.execute(
                    text(
                        """
                    INSERT INTO field_coverage (
                        run_id, field_name, populated_count, empty_count
                    ) VALUES (
                        :run_id, :field_name, :populated_count, :empty_count
                    )
                    ON CONFLICT (run_id, field_name) DO UPDATE SET
                        populated_count = EXCLUDED.populated_count,
                        empty_count = EXCLUDED.empty_count
                    """
                    ),
                    {
                        "run_id": run_id,
                        "field_name": field_name,
                        "populated_count": stats.get("populated", 0),
                        "empty_count": stats.get("empty", 0),
                    },
                )

        logger.info(f"Saved field coverage for {len(field_coverage)} fields in run {run_id}")

    def get_extraction_results_for_ingestion(
        self, run_id: UUID, status: str = "success"
    ) -> List[Dict[str, Any]]:
        """Get extraction results ready for Weaviate ingestion.

        Args:
            run_id: Extraction run UUID
            status: Filter by status (default: success)

        Returns:
            List of extraction result dicts with document_id, extracted_data, etc.
        """
        with self.session_scope() as session:
            result = session.execute(
                text(
                    """
                SELECT
                    document_id,
                    document_number,
                    document_type,
                    extracted_data,
                    full_text_length,
                    source_language
                FROM extraction_results
                WHERE run_id = :run_id AND extraction_status = :status
                ORDER BY extracted_at
                """
                ),
                {"run_id": run_id, "status": status},
            )

            results = []
            for row in result:
                results.append(
                    {
                        "document_id": row.document_id,
                        "document_number": row.document_number,
                        "document_type": row.document_type,
                        "extraction_status": status,
                        "extracted_data": json.loads(row.extracted_data)
                        if row.extracted_data
                        else {},
                        "full_text_length": row.full_text_length,
                        "source_language": row.source_language,
                    }
                )

            return results

    def get_run_summary(self, run_id: UUID) -> Optional[Dict[str, Any]]:
        """Get summary of extraction run.

        Args:
            run_id: Extraction run UUID

        Returns:
            Dict with run summary or None if not found
        """
        with self.session_scope() as session:
            result = session.execute(
                text(
                    """
                SELECT * FROM v_extraction_run_summary
                WHERE run_id = :run_id
                """
                ),
                {"run_id": run_id},
            ).first()

            if not result:
                return None

            return dict(result._mapping)

    def log_ingestion(
        self,
        run_id: UUID,
        batch_size: int,
        overwrite_existing: bool,
        total_documents: int,
        successful_updates: int,
        failed_updates: int,
        skipped_documents: int,
        duration_seconds: float,
        errors: Optional[List[Dict[str, Any]]] = None,
        status: str = "completed",
    ) -> int:
        """Log Weaviate ingestion operation.

        Args:
            run_id: Extraction run UUID
            batch_size: Batch size used for ingestion
            overwrite_existing: Whether existing values were overwritten
            total_documents: Total documents processed
            successful_updates: Count of successful updates
            failed_updates: Count of failed updates
            skipped_documents: Count of skipped documents
            duration_seconds: Total duration
            errors: List of error dicts
            status: Ingestion status (running/completed/failed)

        Returns:
            Ingestion log ID
        """
        with self.session_scope() as session:
            result = session.execute(
                text(
                    """
                INSERT INTO ingestion_logs (
                    run_id, batch_size, overwrite_existing,
                    total_documents, successful_updates, failed_updates,
                    skipped_documents, duration_seconds,
                    errors, status, ingestion_completed_at
                ) VALUES (
                    :run_id, :batch_size, :overwrite_existing,
                    :total_documents, :successful_updates, :failed_updates,
                    :skipped_documents, :duration_seconds,
                    :errors, :status, NOW()
                )
                RETURNING id
                """
                ),
                {
                    "run_id": run_id,
                    "batch_size": batch_size,
                    "overwrite_existing": overwrite_existing,
                    "total_documents": total_documents,
                    "successful_updates": successful_updates,
                    "failed_updates": failed_updates,
                    "skipped_documents": skipped_documents,
                    "duration_seconds": duration_seconds,
                    "errors": json.dumps(errors) if errors else None,
                    "status": status,
                },
            ).scalar()

            logger.info(f"Logged ingestion for run {run_id}: {result}")
            return result

    def get_processed_document_ids(
        self, status: Optional[str] = "success", run_id: Optional[UUID] = None
    ) -> set[str]:
        """Get all document IDs that have been successfully processed.

        Args:
            status: Filter by extraction status (default: "success", None for all statuses)
            run_id: Optional run_id to filter by specific extraction run

        Returns:
            Set of document IDs that have been processed
        """
        with self.session_scope() as session:
            if status is not None:
                if run_id is not None:
                    result = session.execute(
                        text(
                            """
                        SELECT DISTINCT document_id
                        FROM extraction_results
                        WHERE extraction_status = :status AND run_id = :run_id
                        """
                        ),
                        {"status": status, "run_id": run_id},
                    )
                else:
                    result = session.execute(
                        text(
                            """
                        SELECT DISTINCT document_id
                        FROM extraction_results
                        WHERE extraction_status = :status
                        """
                        ),
                        {"status": status},
                    )
            else:
                if run_id is not None:
                    result = session.execute(
                        text(
                            """
                        SELECT DISTINCT document_id
                        FROM extraction_results
                        WHERE run_id = :run_id
                        """
                        ),
                        {"run_id": run_id},
                    )
                else:
                    result = session.execute(
                        text(
                            """
                        SELECT DISTINCT document_id
                        FROM extraction_results
                        """
                        )
                    )

            document_ids = {row.document_id for row in result}
            logger.info(f"Found {len(document_ids)} processed document IDs")
            return document_ids

    def export_to_jsonl(
        self, run_id: UUID, output_path: str, include_full_text: bool = True
    ):
        """Export extraction results to JSONL file.

        Args:
            run_id: Extraction run UUID
            output_path: Output file path
            include_full_text: Whether to include full_text (default: True)
        """
        with self.session_scope() as session:
            query = """
                SELECT
                    document_id,
                    document_number,
                    document_type,
                    {}
                    extracted_data,
                    extraction_status,
                    error_message,
                    full_text_length,
                    source_language
                FROM extraction_results
                WHERE run_id = :run_id
                ORDER BY extracted_at
            """.format(
                "full_text," if include_full_text else ""
            )

            result = session.execute(text(query), {"run_id": run_id})

            count = 0
            with open(output_path, "w", encoding="utf-8") as f:
                for row in result:
                    record = dict(row._mapping)
                    # Parse JSONB fields
                    if record.get("extracted_data"):
                        record["extracted_data"] = json.loads(record["extracted_data"])

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

            logger.info(f"Exported {count} records to {output_path}")

    def get_extractions_for_hf_dataset(self) -> List[Dict[str, Any]]:
        """Export all successful extractions for HuggingFace dataset enrichment.

        Returns the latest extraction for each document_id, with both identifiers
        for flexible joining with different HF datasets.

        Returns:
            List of dicts with document_id, document_number, document_type,
            and all extracted fields (title, summary, thesis, keywords,
            factual_state, legal_state, outcome, legal_references,
            legal_concepts, parties, legal_analysis, judgment_specific,
            tax_interpretation_specific)
        """
        with self.session_scope() as session:
            result = session.execute(
                text(
                    """
                    SELECT DISTINCT ON (document_id)
                        document_id,
                        document_number,
                        document_type,
                        extracted_data->>'title' as extracted_title,
                        extracted_data->>'date_issued' as extracted_date_issued,
                        extracted_data->>'summary' as extracted_summary,
                        extracted_data->>'thesis' as extracted_thesis,
                        extracted_data->>'keywords' as extracted_keywords,
                        extracted_data->>'factual_state' as factual_state,
                        extracted_data->>'legal_state' as legal_state,
                        extracted_data->>'outcome' as extracted_outcome,
                        extracted_data->>'legal_references' as extracted_legal_references,
                        extracted_data->>'legal_concepts' as extracted_legal_concepts,
                        extracted_data->>'parties' as extracted_parties,
                        extracted_data->>'legal_analysis' as extracted_legal_analysis,
                        extracted_data->>'judgment_specific' as extracted_judgment_specific,
                        extracted_data->>'tax_interpretation_specific' as extracted_tax_interpretation_specific
                    FROM extraction_results
                    WHERE extraction_status = 'success'
                      AND extracted_data IS NOT NULL
                    ORDER BY document_id, extracted_at DESC
                    """
                )
            )

            extractions = [dict(row._mapping) for row in result]
            logger.info(f"Exported {len(extractions)} extractions for HF dataset enrichment")
            return extractions
