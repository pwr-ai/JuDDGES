"""Weaviate ingestion service for extracted legal document data.

This module handles ingesting extracted data back into Weaviate, including:
- Building update payloads
- Checking existing values
- Batch updates via Weaviate REST API
- Error tracking and statistics
"""

import time
from typing import Any, Dict, List, Optional

import weaviate.util
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from juddges.extraction.field_mapping import FieldMapper
from juddges.extraction.statistics import IngestionStatistics
from juddges.extraction.weaviate_client import WeaviateRestClient

console = Console()


class WeaviateIngestionService:
    """Service for ingesting extraction results into Weaviate."""

    def __init__(
        self,
        client: WeaviateRestClient,
        field_mapper: Optional[FieldMapper] = None,
    ):
        """Initialize ingestion service.

        Args:
            client: WeaviateRestClient for Weaviate operations
            field_mapper: Optional FieldMapper for transformations (uses default if not provided)
        """
        self.client = client
        self.field_mapper = field_mapper or FieldMapper()

    def ingest_results(
        self,
        extraction_results: List[Dict[str, Any]],
        batch_size: int = 50,
        skip_on_error: bool = True,
        delay_between_batches: float = 0.5,
        overwrite_existing: bool = False,
        use_batch_api: bool = True,
    ) -> IngestionStatistics:
        """Ingest extracted data back into Weaviate.

        This function:
        1. Filters for successfully extracted documents
        2. Builds update payloads mapping extracted fields to Weaviate properties
        3. Checks existing document values before updating (unless overwrite_existing=True)
        4. Uses batch API or individual PATCH requests to update documents
        5. Tracks success/failure statistics

        Args:
            extraction_results: List of extraction results from batch processor
            batch_size: Number of documents to update per batch
            skip_on_error: Continue on individual document errors
            delay_between_batches: Seconds to wait between batches
            overwrite_existing: If False, only update empty/null fields
            use_batch_api: If True, use Weaviate's batch API (recommended)

        Returns:
            IngestionStatistics with complete ingestion summary
        """
        start_time = time.time()

        # Filter for successful extractions only
        successful_results = [
            r for r in extraction_results if r.get("extraction_status") == "success"
        ]

        total_documents = len(extraction_results)
        skipped_documents = total_documents - len(successful_results)

        console.print("\n[cyan]Ingestion Plan:[/cyan]")
        console.print(f"  • Total extraction results: {total_documents}")
        console.print(f"  • Successful extractions: {len(successful_results)}")
        console.print(f"  • Skipped (failed extractions): {skipped_documents}")
        console.print(f"  • Batch size: {batch_size}")
        console.print(
            f"  • Batches to process: {(len(successful_results) + batch_size - 1) // batch_size}"
        )

        if not successful_results:
            logger.warning("No successful extractions to ingest")
            return IngestionStatistics(
                total_documents=total_documents,
                successful_updates=0,
                failed_updates=0,
                skipped_documents=skipped_documents,
                duration_seconds=0,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                errors=[],
            )

        successful_updates = 0
        failed_updates = 0
        errors = []

        # Process in batches with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
        ) as progress:
            task = progress.add_task("Ingesting to Weaviate...", total=len(successful_results))

            # Process in batches
            for batch_idx, batch_start in enumerate(range(0, len(successful_results), batch_size)):
                batch = successful_results[batch_start : batch_start + batch_size]

                logger.info(
                    f"Processing batch {batch_idx + 1}/{(len(successful_results) + batch_size - 1) // batch_size}"
                )

                if use_batch_api:
                    # Use efficient batch API
                    batch_successful, batch_failed, batch_errors = self._ingest_batch_via_batch_api(
                        batch=batch,
                        overwrite_existing=overwrite_existing,
                    )

                    successful_updates += batch_successful
                    failed_updates += batch_failed
                    errors.extend(batch_errors)

                    progress.update(task, advance=len(batch))

                    logger.info(
                        f"Batch {batch_idx + 1}: {batch_successful} successful, {batch_failed} failed"
                    )
                else:
                    # Legacy: individual PATCH requests
                    for result in batch:
                        success = self._ingest_single_document(
                            result=result,
                            overwrite_existing=overwrite_existing,
                        )

                        if success:
                            successful_updates += 1
                        else:
                            failed_updates += 1
                            errors.append({
                                "document_id": result.get("document_id", "unknown"),
                                "error": "Update failed",
                            })

                        progress.update(task, advance=1)

                # Small delay between batches to avoid overwhelming Weaviate
                if batch_idx < (len(successful_results) + batch_size - 1) // batch_size - 1:
                    time.sleep(delay_between_batches)

        # Calculate final statistics
        duration = time.time() - start_time

        return IngestionStatistics(
            total_documents=total_documents,
            successful_updates=successful_updates,
            failed_updates=failed_updates,
            skipped_documents=skipped_documents,
            duration_seconds=duration,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            errors=errors,
        )

    def _ingest_batch_via_batch_api(
        self,
        batch: List[Dict[str, Any]],
        overwrite_existing: bool = False,
    ) -> tuple[int, int, List[Dict[str, str]]]:
        """Ingest a batch using Weaviate's batch API.

        Args:
            batch: List of extraction results
            overwrite_existing: Whether to overwrite existing values

        Returns:
            Tuple of (successful_count, failed_count, errors_list)
        """
        successful = 0
        failed = 0
        errors = []

        # Build batch objects
        batch_objects = []

        for result in batch:
            document_id = result.get("document_id", "")
            extracted_data = result.get("extracted_data", {})

            if not document_id or not extracted_data:
                logger.warning("Skipping result with missing document_id or extracted_data")
                failed += 1
                continue

            # Convert document_id to Weaviate UUID
            weaviate_uuid = weaviate.util.generate_uuid5(document_id)

            # Build update payload
            try:
                update_payload = self.field_mapper.build_update_payload(extracted_data)

                if not update_payload:
                    logger.debug(f"No non-empty fields to update for {document_id}")
                    successful += 1
                    continue

                # Fetch existing data if not overwriting
                if not overwrite_existing:
                    existing_doc = self.client.get_document(document_id)
                    if existing_doc:
                        existing_properties = existing_doc.get("properties", {})
                        update_payload = self._filter_empty_fields(update_payload, existing_properties)

                        if not update_payload:
                            # All fields already populated
                            successful += 1
                            continue

                batch_objects.append({
                    "id": weaviate_uuid,
                    "class": "LegalDocuments",
                    "properties": update_payload,
                })

            except Exception as e:
                logger.error(f"Error building payload for {document_id}: {e}")
                failed += 1
                errors.append({
                    "document_id": document_id,
                    "error": f"Payload build error: {e}",
                })

        if not batch_objects:
            # All documents skipped (no updates needed)
            return successful, failed, errors

        # Send batch request
        batch_successful, batch_failed, batch_errors = self.client.batch_update(
            batch_objects=batch_objects,
            action="MERGE",  # MERGE updates only specified fields
        )

        successful += batch_successful
        failed += batch_failed
        errors.extend(batch_errors)

        return successful, failed, errors

    def _ingest_single_document(
        self,
        result: Dict[str, Any],
        overwrite_existing: bool = False,
    ) -> bool:
        """Ingest a single document using individual PATCH request.

        Args:
            result: Extraction result
            overwrite_existing: Whether to overwrite existing values

        Returns:
            True if successful, False otherwise
        """
        document_id = result.get("document_id", "")
        extracted_data = result.get("extracted_data", {})

        if not document_id or not extracted_data:
            logger.warning("Skipping result with missing document_id or extracted_data")
            return False

        # Build update payload
        try:
            update_payload = self.field_mapper.build_update_payload(extracted_data)

            if not update_payload:
                logger.debug(f"No non-empty fields to update for {document_id}")
                return True  # Consider this success (nothing to update)

            # Fetch existing document if not overwriting
            if not overwrite_existing:
                existing_doc = self.client.get_document(document_id)
                if existing_doc:
                    existing_properties = existing_doc.get("properties", {})
                    update_payload = self._filter_empty_fields(update_payload, existing_properties)

                    if not update_payload:
                        logger.debug(
                            f"No empty fields to update for {document_id} (all fields already populated)"
                        )
                        return True

            # Update document
            return self.client.update_document(document_id, update_payload)

        except Exception as e:
            logger.error(f"Error processing {document_id}: {e}")
            return False

    def _filter_empty_fields(
        self,
        update_payload: Dict[str, Any],
        existing_properties: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Filter out fields that already have non-empty values.

        Args:
            update_payload: Payload to filter
            existing_properties: Existing document properties

        Returns:
            Filtered payload containing only empty fields
        """
        filtered_payload = {}

        for field, value in update_payload.items():
            existing_value = existing_properties.get(field)

            # Check if existing value is empty/null
            is_empty = (
                existing_value is None
                or existing_value == ""
                or (isinstance(existing_value, list) and not existing_value)
            )

            # Only include field if existing value is empty
            if is_empty:
                filtered_payload[field] = value
            else:
                logger.debug(f"Skipping field '{field}' - already has value")

        return filtered_payload
