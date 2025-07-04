#!/usr/bin/env python3
"""
Enhanced streaming ingester interface designed for standalone packaging.
Shows how the current ingester should be refactored for external use.
"""

from typing import Dict, Any, List, Optional, Iterator, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from abc import ABC, abstractmethod

import numpy as np
import weaviate
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn


# Import our configuration and transformation systems
from ingestion_config_example import DatasetConfig, FieldMapping, ConfigurationManager
from transformation_system_example import TransformationEngine, setup_polish_legal_transformations


@dataclass
class IngestionResult:
    """Result of document ingestion process."""

    success: bool
    total_documents: int
    processed_documents: int
    skipped_documents: int
    failed_documents: int
    total_chunks: int
    processing_time: float
    errors: List[str]

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_documents == 0:
            return 0.0
        return self.processed_documents / self.total_documents


class DocumentProcessor(ABC):
    """Abstract base class for document processing."""

    @abstractmethod
    def process_document(
        self, document: Dict[str, Any], config: DatasetConfig
    ) -> Optional[Dict[str, Any]]:
        """Process a single document according to configuration."""
        pass

    @abstractmethod
    def validate_document(
        self, document: Dict[str, Any], config: DatasetConfig
    ) -> tuple[bool, List[str]]:
        """Validate document against configuration requirements."""
        pass


class ConfigurableDocumentProcessor(DocumentProcessor):
    """Document processor that uses configuration and transformations."""

    def __init__(self, transformation_engine: TransformationEngine):
        self.transformation_engine = transformation_engine
        self.console = Console()

    def process_document(
        self, document: Dict[str, Any], config: DatasetConfig
    ) -> Optional[Dict[str, Any]]:
        """Process document using field mappings and transformations."""
        try:
            processed_doc = {}

            # Apply field mappings
            for mapping in config.field_mappings:
                value = document.get(mapping.source_field)

                # Use default value if field is missing
                if value is None and mapping.default_value is not None:
                    value = mapping.default_value

                # Skip if required field is missing
                if value is None and mapping.required:
                    self.console.print(f"[red]Missing required field: {mapping.source_field}[/red]")
                    return None

                # Apply transformation if specified
                if value is not None and mapping.transform:
                    try:
                        value = self.transformation_engine.transform(value, mapping.transform)
                    except Exception as e:
                        self.console.print(
                            f"[yellow]Transformation failed for {mapping.source_field}: {e}[/yellow]"
                        )
                        # Continue with original value

                # Set target field (supporting nested fields like metadata.judges)
                self._set_nested_field(processed_doc, mapping.target_field, value)

            # Apply any additional transformations from config
            processed_doc = self._apply_additional_transformations(processed_doc, config)

            return processed_doc

        except Exception as e:
            self.console.print(f"[red]Error processing document: {e}[/red]")
            return None

    def validate_document(
        self, document: Dict[str, Any], config: DatasetConfig
    ) -> tuple[bool, List[str]]:
        """Validate document against configuration."""
        errors = []

        # Check required fields
        for field in config.required_fields:
            if field not in document or document[field] is None:
                errors.append(f"Missing required field: {field}")

        # Validate field mappings
        for mapping in config.field_mappings:
            if mapping.required and mapping.source_field not in document:
                errors.append(f"Missing required mapped field: {mapping.source_field}")

            # Validate transformation if specified
            value = document.get(mapping.source_field)
            if value is not None and mapping.transform:
                if not self.transformation_engine.validate(value, mapping.transform):
                    errors.append(f"Invalid value for transformation {mapping.transform}: {value}")

        # Validate text fields have content
        for field in config.text_fields:
            if field in document:
                value = document[field]
                if value is not None and len(str(value).strip()) == 0:
                    errors.append(f"Text field '{field}' is empty")

        return len(errors) == 0, errors

    def _set_nested_field(self, doc: Dict[str, Any], field_path: str, value: Any):
        """Set nested field using dot notation (e.g., metadata.judges)."""
        if value is None:
            return

        parts = field_path.split(".")
        current = doc

        # Navigate to the parent of the target field
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the final field
        current[parts[-1]] = value

    def _apply_additional_transformations(
        self, doc: Dict[str, Any], config: DatasetConfig
    ) -> Dict[str, Any]:
        """Apply any additional transformations specified in config."""
        # Example: Add computed fields
        if "metadata" not in doc:
            doc["metadata"] = {}

        doc["metadata"]["processed_at"] = time.time()

        # Apply dataset-specific transformations
        for transform_name, transform_config in config.transformations.items():
            try:
                doc = self._apply_transform(doc, transform_name, transform_config)
            except Exception as e:
                self.console.print(
                    f"[yellow]Additional transformation '{transform_name}' failed: {e}[/yellow]"
                )

        return doc

    def _apply_transform(
        self, doc: Dict[str, Any], transform_name: str, transform_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply a specific transformation based on config."""
        # This is where you'd implement custom business logic
        # For now, just return the document unchanged
        return doc


class ConfigurableStreamingIngester:
    """Enhanced streaming ingester with configuration support."""

    def __init__(
        self,
        config: DatasetConfig,
        weaviate_url: str = "http://localhost:8080",
        embedding_model: str = "sdadas/mmlw-roberta-large",
        transformation_engine: Optional[TransformationEngine] = None,
        custom_processor: Optional[DocumentProcessor] = None,
    ):
        self.config = config
        self.console = Console()

        # Setup Weaviate connection
        self.weaviate_client = self._connect_to_weaviate(weaviate_url)

        # Setup embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Setup transformation engine
        self.transformation_engine = transformation_engine or TransformationEngine()

        # Setup document processor
        self.document_processor = custom_processor or ConfigurableDocumentProcessor(
            self.transformation_engine
        )

        # Setup document tracker (could be SQLite, Redis, etc.)
        self.tracker = self._create_tracker()

        self.console.print(f"[green]Initialized ingester for dataset: {config.name}[/green]")

    def _connect_to_weaviate(self, weaviate_url: str) -> weaviate.Client:
        """Connect to Weaviate with proper authentication."""
        import os
        
        url_parts = weaviate_url.split("://")[-1].split(":")
        host = url_parts[0]
        port = int(url_parts[1]) if len(url_parts) > 1 else 8080

        # Get API key from environment variables
        api_key = os.getenv('WEAVIATE_API_KEY') or os.getenv('WV_API_KEY')
        
        if api_key:
            import weaviate.auth as wv_auth

            return weaviate.connect_to_local(
                host=host, port=port, auth_credentials=wv_auth.AuthApiKey(api_key)
            )
        else:
            return weaviate.connect_to_local(host=host, port=port)

    def _create_tracker(self):
        """Create document tracker for resume capability."""
        # In the standalone package, this could be configurable
        # (SQLite, Redis, PostgreSQL, etc.)
        from stream_ingester import ProcessedDocTracker

        return ProcessedDocTracker(f"{self.config.name}_processed.db")

    def ingest_dataset(
        self, streaming: bool = True, max_documents: Optional[int] = None, resume: bool = True
    ) -> IngestionResult:
        """Ingest dataset with configuration-driven processing."""

        start_time = time.time()
        result = IngestionResult(
            success=False,
            total_documents=0,
            processed_documents=0,
            skipped_documents=0,
            failed_documents=0,
            total_chunks=0,
            processing_time=0.0,
            errors=[],
        )

        try:
            # Load dataset
            dataset = load_dataset(self.config.huggingface_path, split="train", streaming=streaming)

            if max_documents:
                dataset = dataset.take(max_documents)

            # Process documents
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task("Processing documents...", total=max_documents)

                for doc in dataset:
                    result.total_documents += 1

                    # Extract document ID for tracking
                    doc_id = self._extract_document_id(doc)
                    if not doc_id:
                        result.failed_documents += 1
                        result.errors.append(
                            f"No document ID found in document {result.total_documents}"
                        )
                        continue

                    # Skip if already processed and resume is enabled
                    if resume and self.tracker.is_processed(doc_id):
                        result.skipped_documents += 1
                        progress.advance(task)
                        continue

                    # Validate document
                    is_valid, validation_errors = self.document_processor.validate_document(
                        doc, self.config
                    )
                    if not is_valid:
                        result.failed_documents += 1
                        result.errors.extend(validation_errors)
                        continue

                    # Process document
                    processed_doc = self.document_processor.process_document(doc, self.config)
                    if not processed_doc:
                        result.failed_documents += 1
                        result.errors.append(f"Failed to process document {doc_id}")
                        continue

                    # Generate embeddings and ingest
                    try:
                        chunks_count = self._ingest_document(processed_doc)
                        result.total_chunks += chunks_count
                        result.processed_documents += 1

                        # Mark as processed
                        self.tracker.mark_processed(doc_id, chunks_count, True)

                    except Exception as e:
                        result.failed_documents += 1
                        result.errors.append(f"Failed to ingest document {doc_id}: {e}")
                        self.tracker.mark_processed(doc_id, 0, False)

                    progress.advance(task)

            result.processing_time = time.time() - start_time
            result.success = result.failed_documents == 0

            self._display_results(result)
            return result

        except Exception as e:
            result.errors.append(f"Fatal error during ingestion: {e}")
            result.processing_time = time.time() - start_time
            return result

    def _extract_document_id(self, document: Dict[str, Any]) -> Optional[str]:
        """Extract document ID from document using field mappings."""
        # Look for document_id mapping in config
        for mapping in self.config.field_mappings:
            if mapping.target_field == "document_id":
                return document.get(mapping.source_field)

        # Fallback to common ID fields
        for field in ["document_id", "judgment_id", "id", "case_id"]:
            if field in document:
                return document[field]

        return None

    def _ingest_document(self, document: Dict[str, Any]) -> int:
        """Ingest a single processed document and return chunks count."""
        # This is a simplified version - in the real implementation,
        # you'd chunk the text, generate embeddings, and upload to Weaviate

        # For now, just return a mock count
        text = document.get("full_text", "")
        estimated_chunks = max(1, len(text) // self.config.chunk_size)
        return estimated_chunks

    def _display_results(self, result: IngestionResult):
        """Display ingestion results using Rich."""
        from rich.table import Table
        from rich.panel import Panel

        table = Table(title="📊 Ingestion Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")
        table.add_column("Rate", style="yellow", justify="right")

        rate = (
            result.processed_documents / result.processing_time if result.processing_time > 0 else 0
        )

        table.add_row("Total Documents", str(result.total_documents), "-")
        table.add_row("✅ Processed", str(result.processed_documents), f"{rate:.1f}/sec")
        table.add_row("⏭️ Skipped", str(result.skipped_documents), "-")
        table.add_row("❌ Failed", str(result.failed_documents), "-")
        table.add_row("📄 Total Chunks", str(result.total_chunks), "-")
        table.add_row("⏱️ Time", f"{result.processing_time:.1f}s", "-")
        table.add_row("📈 Success Rate", f"{result.success_rate:.1%}", "-")

        self.console.print(table)

        if result.errors:
            error_panel = Panel(
                "\n".join(result.errors[:10]),  # Show first 10 errors
                title="❌ Errors",
                border_style="red",
            )
            self.console.print(error_panel)


# Example usage for standalone package
def create_ingester_for_dataset(dataset_name: str, **kwargs) -> ConfigurableStreamingIngester:
    """Factory function to create configured ingester."""

    # Get configuration
    config_manager = ConfigurationManager()
    config = config_manager.get_config(dataset_name)

    if not config:
        raise ValueError(f"No configuration found for dataset: {dataset_name}")

    # Setup transformation engine
    transformation_engine = TransformationEngine()

    # Setup domain-specific transformations
    if "pl-court" in dataset_name or dataset_name == "tax-interpretations":
        data_dir = Path("data/mappings")  # Would be included in package
        setup_polish_legal_transformations(transformation_engine, data_dir)

    # Create ingester
    return ConfigurableStreamingIngester(
        config=config, transformation_engine=transformation_engine, **kwargs
    )


if __name__ == "__main__":
    # Example usage
    try:
        # Create ingester for Polish courts (set API key first)
        import os
        os.environ['WEAVIATE_API_KEY'] = 'your-api-key'
        
        ingester = create_ingester_for_dataset(
            "pl-court-raw", weaviate_url="http://localhost:8084"
        )

        # Ingest dataset
        result = ingester.ingest_dataset(streaming=True, max_documents=100, resume=True)

        print(f"Ingestion completed: {result.success}")
        print(f"Processed: {result.processed_documents}/{result.total_documents}")

    except Exception as e:
        print(f"Error: {e}")
