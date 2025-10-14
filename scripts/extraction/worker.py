#!/usr/bin/env python3
"""Distributed extraction worker for large-scale processing.

This worker:
1. Polls a Redis queue for document batches
2. Extracts data using GeminiExtractionChain
3. Saves results to PostgreSQL checkpoint database
4. Continues until queue is empty

Usage:
    python scripts/extraction/worker.py --worker-id 1 --redis-url redis://localhost:6379
"""

import argparse
import json
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

import redis
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLAlchemyMd5Cache
from langfuse.langchain import CallbackHandler
from loguru import logger
from sqlalchemy import create_engine

from juddges.extraction import (
    ExtractionStorage,
    GeminiExtractionChain,
    WeaviateRestClient,
    create_polish_legal_schema,
)
from juddges.settings import ROOT_PATH

# Load environment variables
load_dotenv(ROOT_PATH / ".env", override=True)

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global SHUTDOWN_REQUESTED
    logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
    SHUTDOWN_REQUESTED = True


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class ExtractionWorker:
    """Worker that processes extraction jobs from Redis queue."""

    def __init__(
        self,
        worker_id: int,
        redis_url: str,
        queue_name: str = "extraction_queue",
        batch_size: int = 3,
        model_name: str = "gemini-2.5-pro",
        use_langfuse: bool = False,
        langfuse_sample_rate: float = 0.01,
    ):
        """Initialize extraction worker.

        Args:
            worker_id: Unique worker identifier
            redis_url: Redis connection URL
            queue_name: Name of Redis queue to poll
            batch_size: Number of documents per extraction batch (keep small for long docs)
            model_name: Gemini model to use
            use_langfuse: Whether to enable Langfuse tracing
            langfuse_sample_rate: Fraction of requests to trace (reduces overhead)
        """
        self.worker_id = worker_id
        self.queue_name = queue_name
        self.batch_size = batch_size
        self.model_name = model_name

        # Connect to Redis
        logger.info(f"[Worker {worker_id}] Connecting to Redis: {redis_url}")
        self.redis_client = redis.from_url(redis_url, decode_responses=True)

        # Initialize LangChain cache
        self._init_cache()

        # Initialize Gemini chain
        logger.info(f"[Worker {worker_id}] Initializing Gemini extraction chain...")
        vertex_project = os.getenv("VERTEX_PROJECT", "insbay-b32351")
        vertex_location = os.getenv("VERTEX_LOCATION", "us-central1")

        self.chain = GeminiExtractionChain(
            model_name=model_name,
            project=vertex_project,
            location=vertex_location,
            temperature=0.0,
        )

        # Create schema
        self.schema = create_polish_legal_schema()
        logger.info(f"[Worker {worker_id}] Created schema with {len(self.schema.fields)} fields")

        # Initialize storage
        try:
            self.storage = ExtractionStorage()
            logger.info(f"[Worker {worker_id}] Connected to extraction storage")
        except Exception as e:
            logger.error(f"[Worker {worker_id}] Failed to connect to storage: {e}")
            self.storage = None

        # Initialize Weaviate client for fetching document content
        self.weaviate_client = WeaviateRestClient.from_env()

        # Initialize Langfuse with sampling
        self.langfuse_handler = None
        if use_langfuse and os.getenv("LANGFUSE_PUBLIC_KEY"):
            try:
                self.langfuse_handler = CallbackHandler()
                self.langfuse_sample_rate = langfuse_sample_rate
                logger.info(
                    f"[Worker {worker_id}] Langfuse enabled with {langfuse_sample_rate*100}% sampling"
                )
            except Exception as e:
                logger.warning(f"[Worker {worker_id}] Failed to init Langfuse: {e}")

        # Statistics
        self.stats = {
            "jobs_processed": 0,
            "documents_extracted": 0,
            "documents_failed": 0,
            "start_time": time.time(),
        }

    def _init_cache(self):
        """Initialize LangChain PostgreSQL cache."""
        postgres_cache_url = os.getenv("POSTGRES_CACHE_URL")
        if postgres_cache_url:
            try:
                logger.info(f"[Worker {self.worker_id}] Initializing LangChain cache...")
                engine = create_engine(postgres_cache_url)
                set_llm_cache(SQLAlchemyMd5Cache(engine=engine))
                logger.info(f"[Worker {self.worker_id}] LangChain cache enabled")
            except Exception as e:
                logger.warning(f"[Worker {self.worker_id}] Failed to init cache: {e}")

    def run(self):
        """Main worker loop - poll queue and process jobs."""
        logger.info(f"[Worker {self.worker_id}] Starting worker loop...")
        logger.info(f"[Worker {self.worker_id}] Queue: {self.queue_name}")
        logger.info(f"[Worker {self.worker_id}] Batch size: {self.batch_size}")

        consecutive_empty_polls = 0
        max_empty_polls = 10  # Exit after 10 empty polls (queue exhausted)

        while not SHUTDOWN_REQUESTED:
            try:
                # Poll queue with timeout
                job_data = self.redis_client.rpop(self.queue_name)

                if not job_data:
                    consecutive_empty_polls += 1
                    if consecutive_empty_polls >= max_empty_polls:
                        logger.info(
                            f"[Worker {self.worker_id}] Queue empty after {max_empty_polls} polls, exiting..."
                        )
                        break

                    logger.debug(
                        f"[Worker {self.worker_id}] Queue empty, waiting... ({consecutive_empty_polls}/{max_empty_polls})"
                    )
                    time.sleep(5)
                    continue

                # Reset empty poll counter
                consecutive_empty_polls = 0

                # Parse job
                job = json.loads(job_data)
                self.process_job(job)

            except Exception as e:
                logger.error(f"[Worker {self.worker_id}] Error in main loop: {e}")
                time.sleep(1)

        # Print final statistics
        self.print_statistics()

    def process_job(self, job: Dict[str, Any]):
        """Process a single extraction job.

        Args:
            job: Job data with format:
                {
                    "job_id": "unique_job_id",
                    "run_id": "extraction_run_id",
                    "document_ids": ["doc1", "doc2", ...],
                }
        """
        job_id = job.get("job_id")
        run_id = job.get("run_id")
        document_ids = job.get("document_ids", [])

        logger.info(
            f"[Worker {self.worker_id}] Processing job {job_id}: {len(document_ids)} documents"
        )

        try:
            # Fetch full documents from Weaviate
            documents = self._fetch_documents(document_ids)

            if not documents:
                logger.warning(f"[Worker {self.worker_id}] No documents found for job {job_id}")
                return

            # Extract in small batches
            results = self._extract_documents(documents, run_id)

            # Update statistics
            self.stats["jobs_processed"] += 1
            self.stats["documents_extracted"] += sum(
                1 for r in results if r.get("extraction_status") == "success"
            )
            self.stats["documents_failed"] += sum(
                1 for r in results if r.get("extraction_status") == "failed"
            )

            logger.info(
                f"[Worker {self.worker_id}] Completed job {job_id}: "
                f"{len(results)} docs processed"
            )

        except Exception as e:
            logger.error(f"[Worker {self.worker_id}] Failed to process job {job_id}: {e}")

    def _fetch_documents(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch full document content from Weaviate.

        Args:
            document_ids: List of document IDs to fetch

        Returns:
            List of document dictionaries with full_text
        """
        documents = []

        for doc_id in document_ids:
            try:
                doc = self.weaviate_client.get_document(doc_id)
                if doc and doc.get("properties"):
                    props = doc["properties"]
                    documents.append(
                        {
                            "document_id": props.get("document_id"),
                            "document_number": props.get("document_number"),
                            "document_type": props.get("document_type"),
                            "full_text": props.get("full_text", ""),
                            "language": props.get("language", "pl"),
                        }
                    )
            except Exception as e:
                logger.error(f"[Worker {self.worker_id}] Failed to fetch {doc_id}: {e}")

        return documents

    def _extract_documents(
        self, documents: List[Dict[str, Any]], run_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Extract data from documents in small batches.

        Args:
            documents: List of documents to extract
            run_id: Optional run ID for storage

        Returns:
            List of extraction results
        """
        results = []

        # Process in small batches (3 docs to avoid token limits)
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]

            # Determine if we should trace this batch
            use_langfuse = (
                self.langfuse_handler
                and self.langfuse_sample_rate
                and (time.time() % 1.0) < self.langfuse_sample_rate
            )

            try:
                # Prepare batch
                batch_texts = [doc.get("full_text", "") for doc in batch]
                batch_metadata = [
                    {
                        "document_id": doc.get("document_id"),
                        "document_type": doc.get("document_type", "judgment"),
                    }
                    for doc in batch
                ]

                # Determine document type
                doc_type_str = batch[0].get("document_type", "judgment")
                from juddges.extraction.gemini_chain import DocumentType

                if "interpret" in doc_type_str.lower():
                    doc_type = DocumentType.TAX_INTERPRETATION
                else:
                    doc_type = DocumentType.JUDGMENT

                # Extract batch
                extracted_batch = self.chain.batch_extract(
                    document_type=doc_type,
                    texts=batch_texts,
                    schema=self.schema,
                    langfuse_handler=self.langfuse_handler if use_langfuse else None,
                    max_text_length=150000,
                )

                # Process results
                for extracted, metadata, doc in zip(extracted_batch, batch_metadata, batch):
                    result = {
                        "document_id": metadata["document_id"],
                        "document_number": doc.get("document_number"),
                        "document_type": doc.get("document_type"),
                        "extraction_status": "success",
                        "extracted_data": extracted,
                        "full_text_length": len(doc.get("full_text", "")),
                        "source_language": doc.get("language", "pl"),
                    }
                    results.append(result)

                    # Save to storage if available
                    if self.storage and run_id:
                        try:
                            self.storage.save_extraction_result(
                                run_id=run_id,
                                document_id=metadata["document_id"],
                                document_number=doc.get("document_number"),
                                document_type=doc.get("document_type"),
                                full_text=doc.get("full_text", ""),
                                extraction_status="success",
                                extracted_data=extracted,
                                source_language=doc.get("language", "pl"),
                            )
                        except Exception as e:
                            logger.warning(
                                f"[Worker {self.worker_id}] Failed to save to DB: {e}"
                            )

                    logger.debug(
                        f"[Worker {self.worker_id}] ✓ Extracted {metadata['document_id']}"
                    )

            except Exception as e:
                error_msg = str(e)
                logger.error(f"[Worker {self.worker_id}] Batch extraction failed: {error_msg}")

                # Determine if error is retryable
                is_rate_limit = "429" in error_msg or "rate limit" in error_msg.lower()
                is_server_error = "500" in error_msg or "503" in error_msg or "502" in error_msg
                is_timeout = "timeout" in error_msg.lower()

                # Log error type for better debugging
                if is_rate_limit:
                    logger.warning(f"[Worker {self.worker_id}] Rate limit detected - will retry individual docs with backoff")
                elif is_server_error:
                    logger.warning(f"[Worker {self.worker_id}] API server error - will retry individual docs")
                elif is_timeout:
                    logger.warning(f"[Worker {self.worker_id}] Timeout error - will retry individual docs")

                # Fallback to individual processing with retry logic
                for doc in batch:
                    max_retries = 3
                    retry_delay = 2.0  # seconds

                    for attempt in range(max_retries):
                        try:
                            extracted = self.chain.extract(
                                document_type=doc_type,
                                text=doc.get("full_text", ""),
                                schema=self.schema,
                                langfuse_handler=self.langfuse_handler if use_langfuse else None,
                                max_text_length=150000,
                            )
                            result = {
                                "document_id": doc["document_id"],
                                "document_number": doc.get("document_number"),
                                "document_type": doc.get("document_type"),
                                "extraction_status": "success",
                                "extracted_data": extracted,
                            }
                            results.append(result)

                            # Save to storage if available
                            if self.storage and run_id:
                                try:
                                    self.storage.save_extraction_result(
                                        run_id=run_id,
                                        document_id=doc["document_id"],
                                        document_number=doc.get("document_number"),
                                        document_type=doc.get("document_type"),
                                        full_text=doc.get("full_text", ""),
                                        extraction_status="success",
                                        extracted_data=extracted,
                                        source_language=doc.get("language", "pl"),
                                    )
                                except Exception as storage_err:
                                    logger.warning(
                                        f"[Worker {self.worker_id}] Failed to save to DB: {storage_err}"
                                    )

                            logger.debug(f"[Worker {self.worker_id}] ✓ Extracted {doc['document_id']}")
                            break  # Success - exit retry loop

                        except Exception as e2:
                            error_str = str(e2)
                            is_last_attempt = (attempt == max_retries - 1)

                            # Log with appropriate severity
                            if is_last_attempt:
                                logger.error(
                                    f"[Worker {self.worker_id}] Failed {doc['document_id']} "
                                    f"after {max_retries} attempts: {error_str}"
                                )
                                # Mark as failed after all retries exhausted
                                result = {
                                    "document_id": doc["document_id"],
                                    "document_number": doc.get("document_number"),
                                    "extraction_status": "failed",
                                    "error": error_str,
                                    "attempts": max_retries,
                                }
                                results.append(result)
                            else:
                                # Retry with exponential backoff
                                backoff_time = retry_delay * (2 ** attempt)
                                logger.warning(
                                    f"[Worker {self.worker_id}] Attempt {attempt + 1}/{max_retries} "
                                    f"failed for {doc['document_id']}: {error_str}. "
                                    f"Retrying in {backoff_time:.1f}s..."
                                )
                                time.sleep(backoff_time)

        return results

    def print_statistics(self):
        """Print worker statistics."""
        duration = time.time() - self.stats["start_time"]
        docs_per_min = (
            (self.stats["documents_extracted"] / duration) * 60 if duration > 0 else 0
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"[Worker {self.worker_id}] Final Statistics")
        logger.info(f"{'='*60}")
        logger.info(f"  Jobs processed: {self.stats['jobs_processed']}")
        logger.info(f"  Documents extracted: {self.stats['documents_extracted']}")
        logger.info(f"  Documents failed: {self.stats['documents_failed']}")
        logger.info(f"  Duration: {duration:.1f} seconds")
        logger.info(f"  Throughput: {docs_per_min:.1f} docs/min")
        logger.info(f"{'='*60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Distributed extraction worker")
    parser.add_argument(
        "--worker-id",
        type=int,
        required=True,
        help="Unique worker ID",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default="redis://localhost:6379",
        help="Redis connection URL",
    )
    parser.add_argument(
        "--queue-name",
        type=str,
        default="extraction_queue",
        help="Redis queue name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Batch size for extraction (keep small for long documents)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        help="Gemini model to use",
    )
    parser.add_argument(
        "--use-langfuse",
        action="store_true",
        help="Enable Langfuse tracing (with sampling)",
    )
    parser.add_argument(
        "--langfuse-sample-rate",
        type=float,
        default=0.01,
        help="Fraction of requests to trace with Langfuse (default: 0.01 = 1%%)",
    )

    args = parser.parse_args()

    # Create and run worker
    worker = ExtractionWorker(
        worker_id=args.worker_id,
        redis_url=args.redis_url,
        queue_name=args.queue_name,
        batch_size=args.batch_size,
        model_name=args.model,
        use_langfuse=args.use_langfuse,
        langfuse_sample_rate=args.langfuse_sample_rate,
    )

    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info(f"[Worker {args.worker_id}] Interrupted by user")
    except Exception as e:
        logger.error(f"[Worker {args.worker_id}] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
