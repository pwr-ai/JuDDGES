"""Parallel batch processing for document extraction operations.

This module provides a generic batch processor that can handle parallel processing
of documents using ThreadPoolExecutor with progress tracking.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from juddges.extraction.gemini_chain import DocumentType, ExtractionSchema, GeminiExtractionChain
from juddges.extraction.extraction_storage import ExtractionStorage

console = Console()


class BatchProcessor:
    """Generic batch processor for parallel document processing with progress tracking."""

    def __init__(
        self,
        batch_size: int = 5,
        max_workers: int = 5,
        show_progress: bool = True,
    ):
        """Initialize batch processor.

        Args:
            batch_size: Number of documents to process in each batch
            max_workers: Number of parallel threads (1 = sequential, >1 = parallel)
            show_progress: Whether to show progress bar
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.show_progress = show_progress

    def process_extraction_batches(
        self,
        documents: List[Dict[str, Any]],
        chain: GeminiExtractionChain,
        schema: ExtractionSchema,
        langfuse_handler: Optional[Any] = None,
        storage: Optional[ExtractionStorage] = None,
        run_id: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Process documents in batches using extraction chain.

        Args:
            documents: List of document properties
            chain: Extraction chain for LLM processing
            schema: Extraction schema
            langfuse_handler: Optional Langfuse callback handler
            storage: Optional ExtractionStorage for database persistence
            run_id: Optional run_id for database storage

        Returns:
            List of extraction results with metadata
        """
        # Create batches
        batches = []
        for batch_start in range(0, len(documents), self.batch_size):
            batch_docs = documents[batch_start : batch_start + self.batch_size]
            batches.append((batch_start // self.batch_size, batch_docs))

        logger.info(
            f"Created {len(batches)} batches of size {self.batch_size} with {self.max_workers} parallel workers"
        )

        results = []
        results_lock = Lock()

        # Setup progress bar
        progress_ctx = self._create_progress_bar()

        with progress_ctx as progress:
            task = progress.add_task("Extracting documents...", total=len(documents))

            if self.max_workers == 1:
                # Sequential processing (no threading)
                for batch_idx, batch_docs in batches:
                    batch_results = self._process_extraction_batch(
                        batch_docs=batch_docs,
                        chain=chain,
                        schema=schema,
                        langfuse_handler=langfuse_handler,
                        batch_idx=batch_idx,
                        storage=storage,
                        run_id=run_id,
                    )
                    with results_lock:
                        results.extend(batch_results)
                        progress.update(task, advance=len(batch_results))
            else:
                # Parallel processing with ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all batches
                    future_to_batch = {
                        executor.submit(
                            self._process_extraction_batch,
                            batch_docs=batch_docs,
                            chain=chain,
                            schema=schema,
                            langfuse_handler=langfuse_handler,
                            batch_idx=batch_idx,
                            storage=storage,
                            run_id=run_id,
                        ): batch_idx
                        for batch_idx, batch_docs in batches
                    }

                    # Process completed batches
                    for future in as_completed(future_to_batch):
                        batch_idx = future_to_batch[future]
                        try:
                            batch_results = future.result()
                            with results_lock:
                                results.extend(batch_results)
                                progress.update(task, advance=len(batch_results))
                        except Exception as e:
                            logger.error(f"[Batch {batch_idx}] Thread execution failed: {e}")

        return results

    def _process_extraction_batch(
        self,
        batch_docs: List[Dict[str, Any]],
        chain: GeminiExtractionChain,
        schema: ExtractionSchema,
        langfuse_handler: Optional[Any],
        batch_idx: int,
        storage: Optional[ExtractionStorage] = None,
        run_id: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Process a single batch of documents for extraction.

        Args:
            batch_docs: Documents in this batch
            chain: Extraction chain
            schema: Extraction schema
            langfuse_handler: Langfuse handler
            batch_idx: Batch index for logging
            storage: Optional storage for persistence
            run_id: Optional run ID for storage

        Returns:
            List of extraction results
        """
        batch_results = []

        # Prepare batch data
        batch_texts = []
        batch_metadata = []

        for doc in batch_docs:
            document_id = doc.get("document_id", "unknown")
            full_text = doc.get("full_text", "")
            doc_type_str = doc.get("document_type", "judgment")

            # Map document type
            if "interpret" in doc_type_str.lower():
                doc_type = DocumentType.TAX_INTERPRETATION
            else:
                doc_type = DocumentType.JUDGMENT

            batch_texts.append(full_text)
            batch_metadata.append({
                "document_id": document_id,
                "document_type": doc_type,
                "doc_type_str": doc_type_str,
                "full_text_length": len(full_text),
                "language": doc.get("language", "unknown"),
            })

        # Get document type for batch
        batch_doc_type = batch_metadata[0]["document_type"]

        try:
            # Run batch extraction
            logger.info(f"[Batch {batch_idx}] Processing {len(batch_texts)} documents...")
            extracted_batch = chain.batch_extract(
                document_type=batch_doc_type,
                texts=batch_texts,
                schema=schema,
                langfuse_handler=langfuse_handler,
                max_text_length=150000,
            )

            # Process batch results
            for extracted, metadata, doc in zip(extracted_batch, batch_metadata, batch_docs):
                result = {
                    "document_id": metadata["document_id"],
                    "document_number": doc.get("document_number"),
                    "document_type": metadata["doc_type_str"],
                    "extraction_status": "success",
                    "extracted_data": extracted,
                    "full_text": doc.get("full_text", ""),
                    "full_text_length": metadata["full_text_length"],
                    "source_language": metadata["language"],
                }
                batch_results.append(result)

                # Save to database if storage provided
                if storage and run_id:
                    self._save_to_storage(storage, run_id, result, metadata, doc)

                logger.info(
                    f"[Batch {batch_idx}] ✓ Extracted {metadata['document_id']} ({len(extracted)} fields)"
                )

        except Exception as e:
            # If batch fails, fall back to individual processing
            logger.warning(
                f"[Batch {batch_idx}] Batch extraction failed: {e}, falling back to individual processing"
            )

            for text, metadata, doc in zip(batch_texts, batch_metadata, batch_docs):
                try:
                    extracted = chain.extract(
                        document_type=metadata["document_type"],
                        text=text,
                        schema=schema,
                        langfuse_handler=langfuse_handler,
                        max_text_length=150000,
                    )

                    result = {
                        "document_id": metadata["document_id"],
                        "document_number": doc.get("document_number"),
                        "document_type": metadata["doc_type_str"],
                        "extraction_status": "success",
                        "extracted_data": extracted,
                        "full_text": doc.get("full_text", ""),
                        "full_text_length": metadata["full_text_length"],
                        "source_language": metadata["language"],
                    }
                    batch_results.append(result)

                    # Save to database if storage provided
                    if storage and run_id:
                        self._save_to_storage(storage, run_id, result, metadata, doc)

                    logger.info(
                        f"[Batch {batch_idx}] ✓ Extracted {metadata['document_id']} ({len(extracted)} fields)"
                    )

                except Exception as e2:
                    logger.error(
                        f"[Batch {batch_idx}] ✗ Failed to extract {metadata['document_id']}: {e2}"
                    )
                    result = {
                        "document_id": metadata["document_id"],
                        "document_number": doc.get("document_number"),
                        "document_type": metadata["doc_type_str"],
                        "extraction_status": "failed",
                        "error": str(e2),
                        "full_text": doc.get("full_text", ""),
                        "full_text_length": metadata["full_text_length"],
                    }
                    batch_results.append(result)

                    # Save failed result to database if storage provided
                    if storage and run_id:
                        self._save_failed_to_storage(storage, run_id, result, metadata, doc, e2)

        return batch_results

    def _save_to_storage(
        self,
        storage: ExtractionStorage,
        run_id: Any,
        result: Dict[str, Any],
        metadata: Dict[str, Any],
        doc: Dict[str, Any],
    ):
        """Save successful extraction to storage."""
        try:
            storage.save_extraction_result(
                run_id=run_id,
                document_id=metadata["document_id"],
                document_number=doc.get("document_number"),
                document_type=metadata["doc_type_str"],
                full_text=doc.get("full_text", ""),
                extraction_status="success",
                extracted_data=result["extracted_data"],
                source_language=metadata["language"],
            )
        except Exception as db_error:
            logger.warning(f"Failed to save to database: {db_error}")

    def _save_failed_to_storage(
        self,
        storage: ExtractionStorage,
        run_id: Any,
        result: Dict[str, Any],
        metadata: Dict[str, Any],
        doc: Dict[str, Any],
        error: Exception,
    ):
        """Save failed extraction to storage."""
        try:
            storage.save_extraction_result(
                run_id=run_id,
                document_id=metadata["document_id"],
                document_number=doc.get("document_number"),
                document_type=metadata["doc_type_str"],
                full_text=doc.get("full_text", ""),
                extraction_status="failed",
                error_message=str(error),
                error_type=type(error).__name__,
            )
        except Exception as db_error:
            logger.warning(f"Failed to save error to database: {db_error}")

    def _create_progress_bar(self):
        """Create progress bar context manager."""
        if not self.show_progress:
            # Return a dummy context manager that does nothing
            from contextlib import nullcontext
            return nullcontext()

        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        )
