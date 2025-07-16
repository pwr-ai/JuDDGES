"""
Simplified streaming ingester for legal documents.
Processes documents one by one, generates embeddings, and ingests immediately to Weaviate.
"""

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from sentence_transformers import SentenceTransformer

import weaviate
from juddges.preprocessing.text_chunker import TextChunker
from juddges.settings import VectorName
from juddges.utils.date_utils import convert_date_to_rfc3339
from weaviate.classes.query import Metrics

load_dotenv(".env", override=True)

POLISH_STOP_WORDS = [
    # Polish stop words
    "a",
    "aby",
    "ale",
    "albo",
    "am",
    "ani",
    "oraz",
    "oraz",
    "będzie",
    "będą",
    "być",
    "ci",
    "co",
    "czy",
    "da",
    "dla",
    "do",
    "i",
    "ich",
    "ja",
    "jak",
    "jako",
    "je",
    "jego",
    "jej",
    "jestem",
    "jest",
    "już",
    "lub",
    "ma",
    "może",
    "na",
    "nad",
    "nie",
    "nią",
    "nim",
    "niż",
    "o",
    "od",
    "po",
    "pod",
    "są",
    "się",
    "ta",
    "tak",
    "także",
    "te",
    "tej",
    "tem",
    "ten",
    "to",
    "ty",
    "w",
    "we",
    "więc",
    "za",
    "że",
    "z",
]


@dataclass
class TextChunk:
    """Simple text chunk representation."""

    document_id: str
    chunk_id: str
    text: str
    position: int


@dataclass
class ProcessingStats:
    """Processing statistics."""

    total_documents: int = 0
    processed_documents: int = 0
    skipped_documents: int = 0
    failed_documents: int = 0
    total_chunks: int = 0
    processing_time: float = 0.0


class ProcessedDocTracker:
    """Track processed documents using SQLite for resume capability."""

    def __init__(self, db_path: str = "processed_documents.db", dataset_name: Optional[str] = None):
        if dataset_name:
            # Create dataset-specific tracker file
            base_path = Path(db_path).parent
            safe_name = dataset_name.replace("/", "_").replace("\\", "_")
            self.db_path = str(base_path / f"processed_documents_{safe_name}.db")
        else:
            self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_documents (
                    document_id TEXT PRIMARY KEY,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunks_count INTEGER,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT
                )
            """)
            # Add error_message column if it doesn't exist (for existing databases)
            try:
                conn.execute("""
                    ALTER TABLE processed_documents 
                    ADD COLUMN error_message TEXT
                """)
            except sqlite3.OperationalError:
                # Column already exists, ignore
                pass
            conn.commit()

    def is_processed(self, document_id: str) -> bool:
        """Check if document has been processed successfully."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT success FROM processed_documents WHERE document_id = ? AND success = TRUE",
                (document_id,),
            )
            return cursor.fetchone() is not None

    def mark_processed(self, document_id: str, chunks_count: int, success: bool = True):
        """Mark document as processed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_documents 
                (document_id, chunks_count, success) 
                VALUES (?, ?, ?)
            """,
                (document_id, chunks_count, success),
            )
            conn.commit()

    def mark_error(self, document_id: str, error_message: str):
        """Mark document with error details without marking as processed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_documents 
                (document_id, chunks_count, success, error_message) 
                VALUES (?, 0, FALSE, ?)
            """,
                (document_id, error_message),
            )
            conn.commit()

    def get_error_documents(self) -> List[Dict[str, Any]]:
        """Get list of documents that failed with error details."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT document_id, processed_at, error_message 
                FROM processed_documents 
                WHERE success = FALSE AND error_message IS NOT NULL
                ORDER BY processed_at DESC
            """
            )
            return [
                {
                    "document_id": row[0],
                    "processed_at": row[1],
                    "error_message": row[2],
                }
                for row in cursor.fetchall()
            ]

    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN success = TRUE THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN success = FALSE THEN 1 ELSE 0 END) as failed
                FROM processed_documents
            """)
            row = cursor.fetchone()
            return {"total": row[0] or 0, "successful": row[1] or 0, "failed": row[2] or 0}

    def reset_documents_by_pattern(self, pattern: str):
        """Reset documents matching a specific pattern."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM processed_documents WHERE document_id LIKE ?", (f"%{pattern}%",)
            )
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count


class StreamingIngester:
    """Simplified streaming ingester for legal documents."""

    # Use PascalCase collection names for consistency
    LEGAL_DOCUMENTS_COLLECTION = "LegalDocuments"
    DOCUMENT_CHUNKS_COLLECTION = "DocumentChunks"

    def __init__(
        self,
        weaviate_url: str = "http://localhost:8084",
        embedding_model: str = "sdadas/mmlw-roberta-large",
        chunk_size: int = 512,
        min_chunk_size: int = 256,
        overlap: int = 128,
        batch_size: int = 32,
        tracker_db: str = "processed_documents.db",
        dataset_config=None,
        embedding_models: Optional[Dict[str, str]] = None,
    ):
        self.console = Console()

        # Parse URL to get host and port
        url_parts = weaviate_url.split("://")[-1].split(":")
        host = url_parts[0]
        port = int(url_parts[1]) if len(url_parts) > 1 else 8084

        # Get API key from environment variables
        api_key = os.getenv("WEAVIATE_API_KEY") or os.getenv("WV_API_KEY")

        # Connect with or without API key
        if api_key:
            import weaviate.auth as wv_auth

            self.weaviate_client = weaviate.connect_to_local(
                host=host, port=port, auth_credentials=wv_auth.AuthApiKey(api_key)
            )
            logger.info("Connected to Weaviate with API key authentication")
        else:
            self.weaviate_client = weaviate.connect_to_local(host=host, port=port)
            logger.info("Connected to Weaviate without authentication")

        # Set up embedding models - use only base model
        if not embedding_models:
            # Use default mapping with only base model
            self.embedding_models = {
                "base": "sdadas/mmlw-roberta-large",
            }
            logger.info("No embedding_models provided, using default base model configuration")
        else:
            # Filter to only use base model
            self.embedding_models = {
                "base": embedding_models.get("base", "sdadas/mmlw-roberta-large")
            }

        # Log the single base model being used
        logger.info(f"Using single base embedding model: {self.embedding_models['base']}")

        logger.info(f"Using embedding models: {self.embedding_models}")

        # Initialize SentenceTransformer models for each named vector
        self.transformers = {}
        for vector_name, model_name in self.embedding_models.items():
            transformer = SentenceTransformer(model_name)
            self.transformers[vector_name] = transformer
            logger.info(f"Initialized {vector_name} vector with model: {model_name}")

        # Keep the original embedding_model for backward compatibility
        self.embedding_model = self.transformers.get("base")

        # Use the primary tokenizer model name for chunking
        primary_tokenizer_name = self.embedding_models.get("base")
        if primary_tokenizer_name:
            logger.info(
                f"Using tokenizer-aware chunking with {primary_tokenizer_name} tokenizer"
            )
        else:
            logger.warning("No tokenizer available, falling back to character-based chunking")

        self.chunker = TextChunker(
            id_col="document_id",
            text_col="text",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            min_split_chars=min_chunk_size,
            tokenizer=primary_tokenizer_name,
        )
        self.batch_size = batch_size

        # Create dataset-specific tracker if dataset config is provided
        dataset_name = (
            dataset_config.name if dataset_config and hasattr(dataset_config, "name") else None
        )
        self.tracker = ProcessedDocTracker(tracker_db, dataset_name)
        self.stats = ProcessingStats()
        self.dataset_config = dataset_config

        # Verify Weaviate connection
        self._verify_weaviate_connection()

        # Validate dataset configuration if provided
        if self.dataset_config:
            self._validate_dataset_configuration()

        logger.info(f"Initialized StreamingIngester with {embedding_model}")

    def _verify_weaviate_connection(self):
        """Verify Weaviate connection and schema."""
        try:
            # Check if collections exist
            collections = self.weaviate_client.collections.list_all()

            # Handle different return types from list_all()
            if isinstance(collections, dict):
                collection_names = list(collections.keys())
            elif hasattr(collections, "__iter__"):
                collection_names = [collection.name for collection in collections]
            else:
                collection_names = []

            logger.info(f"Existing collections: {collection_names}")

            # Check for collections with different casing or naming patterns
            legal_docs_exists = any(
                name.lower()
                in [self.LEGAL_DOCUMENTS_COLLECTION.lower(), "legaldocuments", "legal_documents"]
                for name in collection_names
            )
            chunks_exists = any(
                name.lower()
                in [self.DOCUMENT_CHUNKS_COLLECTION.lower(), "documentchunks", "document_chunks"]
                for name in collection_names
            )

            if not legal_docs_exists:
                self._create_legal_document_class()
            else:
                logger.info(
                    f"Legal documents collection already exists (found variation in: {collection_names})"
                )

            if not chunks_exists:
                self._create_document_chunk_class()
            else:
                logger.info(
                    f"Document chunks collection already exists (found variation in: {collection_names})"
                )

            logger.info("Weaviate connection verified and schema ready")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise

    def _validate_dataset_configuration(self):
        """Validate dataset configuration including column mappings."""
        self.console.print("[bold cyan]🔍 Validating dataset configuration...[/bold cyan]")

        try:
            # Get Weaviate schema properties for validation
            legal_doc_properties = self._get_collection_properties(self.LEGAL_DOCUMENTS_COLLECTION)

            # Validate column mappings
            self._validate_column_mappings(legal_doc_properties)

            # Validate required fields
            self._validate_required_fields()

            # Validate embedding models configuration
            self._validate_embedding_models_config()

            # Show configuration summary
            self._display_configuration_summary(legal_doc_properties)

            self.console.print("[green]✅ Dataset configuration validation passed[/green]")
            logger.info("Dataset configuration validation completed successfully")

        except Exception as e:
            self.console.print(
                f"[bold red]❌ Dataset configuration validation failed: {e}[/bold red]"
            )
            logger.error(f"Dataset configuration validation failed: {e}")
            raise ValueError(f"Invalid dataset configuration: {e}")

    def _get_collection_properties(self, collection_name: str) -> set:
        """Get property names from a Weaviate collection."""
        try:
            collections = self.weaviate_client.collections.list_all()

            # Handle different return types from list_all()
            if isinstance(collections, dict):
                collection_names = list(collections.keys())
            elif hasattr(collections, "__iter__"):
                collection_names = [collection.name for collection in collections]
            else:
                collection_names = []

            if collection_name not in collection_names:
                # Collection doesn't exist yet - return expected properties
                return self._get_expected_properties(collection_name)

            # Get actual properties from existing collection
            collection = self.weaviate_client.collections.get(collection_name)
            config = collection.config.get()
            return {prop.name for prop in config.properties}

        except Exception as e:
            logger.warning(f"Could not retrieve properties for {collection_name}: {e}")
            return self._get_expected_properties(collection_name)

    def _get_expected_properties(self, collection_name: str) -> set:
        """Get expected properties for a collection based on our schema."""
        if collection_name == self.LEGAL_DOCUMENTS_COLLECTION:
            return {
                "document_id",
                "document_type",
                "title",
                "date_issued",
                "document_number",
                "language",
                "country",
                "full_text",
                "summary",
                "thesis",
                "keywords",
                "issuing_body",
                "ingestion_date",
                "last_updated",
                "processing_status",
                "source_url",
                "legal_references",
                "parties",
                "outcome",
                "metadata",
                "publication_date",
                "raw_content",
                "presiding_judge",
                "judges",
                "legal_bases",
                "court_name",
                "department_name",
                "extracted_legal_bases",
                "references",
                "x",
                "y",
            }
        elif collection_name == self.DOCUMENT_CHUNKS_COLLECTION:
            return {
                "document_id",
                "document_type",
                "language",
                "chunk_id",
                "chunk_text",
                "segment_type",
                "position",
                "confidence_score",
                "cited_references",
                "tags",
                "parent_segment_id",
                "x",
                "y",
            }
        return set()

    def _validate_column_mappings(self, legal_doc_properties: set):
        """Validate that column mappings reference valid Weaviate properties."""
        if not self.dataset_config or not self.dataset_config.column_mapping:
            logger.info("No column mappings defined - using field names as-is")
            return

        validation_results = []
        for source_field, target_field in self.dataset_config.column_mapping.items():
            if target_field not in legal_doc_properties:
                validation_results.append(
                    {
                        "status": "warning",
                        "source": source_field,
                        "target": target_field,
                        "message": f"Target field '{target_field}' not found in Weaviate schema",
                    }
                )
            else:
                validation_results.append(
                    {
                        "status": "valid",
                        "source": source_field,
                        "target": target_field,
                        "message": "Valid mapping",
                    }
                )

        # Display validation results
        self.console.print("\n[bold]Column Mapping Validation:[/bold]")
        for result in validation_results:
            if result["status"] == "valid":
                self.console.print(f"  ✅ {result['source']} → {result['target']}")
            else:
                self.console.print(
                    f"  ⚠️  {result['source']} → {result['target']} ({result['message']})"
                )

        # Count warnings
        warnings = [r for r in validation_results if r["status"] == "warning"]
        if warnings:
            self.console.print(f"\n[yellow]Found {len(warnings)} mapping warning(s)[/yellow]")
            logger.warning(f"Found {len(warnings)} column mapping warnings")

    def _validate_required_fields(self):
        """Validate that required fields are properly configured."""
        if not self.dataset_config or not self.dataset_config.required_fields:
            logger.warning("No required fields defined in dataset configuration")
            return

        self.console.print(f"\n[bold]Required Fields:[/bold] {self.dataset_config.required_fields}")

        # Check if required fields have mappings or exist as-is
        missing_mappings = []
        for field in self.dataset_config.required_fields:
            # Check if field has a mapping or is expected to exist in source data
            if field not in self.dataset_config.column_mapping:
                missing_mappings.append(field)

        if missing_mappings:
            self.console.print(
                f"[yellow]Fields without explicit mappings:[/yellow] {missing_mappings}"
            )
            self.console.print("[dim]These fields will be used as-is from source data[/dim]")

    def _validate_embedding_models_config(self):
        """Validate embedding models configuration."""
        if (
            not self.dataset_config
            or not hasattr(self.dataset_config, "embedding_models")
            or not self.dataset_config.embedding_models
        ):
            logger.warning("No embedding models configured in dataset config - using defaults")
            return

        models = self.dataset_config.embedding_models
        self.console.print("\n[bold]Embedding Models Configuration:[/bold]")

        for vector_name, model_name in models.items():
            self.console.print(f"  {vector_name}: {model_name}")

        # Check for duplicates
        unique_models = set(models.values())
        if len(unique_models) < len(models):
            self.console.print("[yellow]⚠️  Some models are duplicated across vectors[/yellow]")

    def _display_configuration_summary(self, legal_doc_properties: set):
        """Display a summary of the dataset configuration."""
        if not self.dataset_config:
            return

        self.console.print("\n[bold cyan]📋 Configuration Summary:[/bold cyan]")
        self.console.print(f"Dataset: {self.dataset_config.name}")
        self.console.print(f"Document type: {self.dataset_config.document_type}")
        self.console.print(f"Chunk size: {self.dataset_config.max_chunk_size}")
        self.console.print(f"Chunk overlap: {self.dataset_config.chunk_overlap}")

        # Show field counts
        self.console.print("\nField Configuration:")
        self.console.print(f"  Column mappings: {len(self.dataset_config.column_mapping)}")
        self.console.print(f"  Required fields: {len(self.dataset_config.required_fields)}")
        self.console.print(f"  Default values: {len(self.dataset_config.default_values)}")
        self.console.print(f"  Available Weaviate properties: {len(legal_doc_properties)}")

        # Show coverage
        mapped_fields = set(self.dataset_config.column_mapping.values())
        unmapped_properties = (
            legal_doc_properties - mapped_fields - set(self.dataset_config.required_fields)
        )
        if unmapped_properties:
            self.console.print(f"  Unmapped Weaviate properties: {len(unmapped_properties)}")

    def _create_legal_document_class(self):
        """Create LegalDocument class in Weaviate."""
        import weaviate.classes.config as wvc

        try:
            # Use same schema as documents_weaviate_db.py but simplified for streaming
            self.weaviate_client.collections.create(
                name=self.LEGAL_DOCUMENTS_COLLECTION,
                properties=[
                    wvc.Property(
                        name="document_id",
                        data_type=wvc.DataType.TEXT,
                        description="Unique identifier for the document",
                        skip_vectorization=True,
                        index_filterable=True,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="document_type",
                        data_type=wvc.DataType.TEXT,
                        description="Type of the document, like judgment, tax_interpretation, etc.",
                        skip_vectorization=False,
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="title",
                        data_type=wvc.DataType.TEXT,
                        description="Title of the document",
                        skip_vectorization=False,
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="date_issued",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        description="Date when the document was issued",
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="document_number",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        description="Unique document number or identifier",
                        index_filterable=True,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="language",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        description="Language of the document",
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="country",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        description="Country of the document",
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="full_text",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=False,
                        description="Full text content of the document",
                        index_filterable=False,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="summary",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=False,
                        description="Summary of the document",
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="thesis",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=False,
                        description="Thesis statement of the document",
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="keywords",
                        data_type=wvc.DataType.TEXT_ARRAY,
                        skip_vectorization=False,
                        description="Keywords associated with the document",
                        index_filterable=True,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="issuing_body",
                        description="Issuing body or authority of the document",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        index_filterable=True,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="ingestion_date",
                        data_type=wvc.DataType.DATE,
                        skip_vectorization=True,
                        description="Date when the document was ingested into the system",
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="last_updated",
                        data_type=wvc.DataType.DATE,
                        skip_vectorization=True,
                        description="Date when the document was last updated",
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="processing_status",
                        description="Current processing status of the document",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="source_url",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        description="Source URL of the document",
                        index_filterable=True,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="legal_references",
                        description="Legal references or citations in the document",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="parties",
                        description="Parties involved in the document",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="outcome",
                        description="Outcome of the document",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="source",
                        description="Source of the document",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="metadata",
                        description="Metadata associated with the document",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="publication_date",
                        description="Publication date of the document",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="raw_content",
                        description="Raw content of the document",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        index_filterable=False,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="presiding_judge",
                        description="Presiding judge of the document",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        index_filterable=True,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="judges",
                        data_type=wvc.DataType.TEXT_ARRAY,
                        skip_vectorization=True,
                        description="List of judges involved in the document",
                        index_filterable=True,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="legal_bases",
                        data_type=wvc.DataType.TEXT_ARRAY,
                        skip_vectorization=True,
                        description="List of legal bases referenced in the document",
                        index_filterable=True,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="court_name",
                        description="Name of the court",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        index_filterable=True,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="department_name",
                        description="Name of the department",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        index_filterable=True,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="extracted_legal_bases",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        description="Extracted legal bases from the document",
                        index_filterable=True,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="references",
                        data_type=wvc.DataType.TEXT_ARRAY,
                        skip_vectorization=True,
                        description="References cited in the document",
                        index_filterable=True,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="x",
                        data_type=wvc.DataType.NUMBER,
                        skip_vectorization=True,
                        index_filterable=False,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="y",
                        data_type=wvc.DataType.NUMBER,
                        skip_vectorization=True,
                        index_filterable=False,
                        index_searchable=False,
                    ),
                ],
                vectorizer_config=[
                    wvc.Configure.NamedVectors.text2vec_transformers(
                        name=VectorName.BASE,
                        vectorize_collection_name=False,
                        source_properties=["full_text"],
                        vector_index_config=wvc.Configure.VectorIndex.hnsw(),
                        inference_url="http://t2v-transformers-base:8080",
                    ),
                    # wvc.Configure.NamedVectors.text2vec_transformers(
                    #     name=VectorName.DEV,
                    #     vectorize_collection_name=False,
                    #     source_properties=["full_text"],
                    #     vector_index_config=wvc.Configure.VectorIndex.hnsw(),
                    #     inference_url="http://t2v-transformers-dev:8080",
                    # ),
                    # wvc.Configure.NamedVectors.text2vec_transformers(
                    #     name=VectorName.FAST,
                    #     vectorize_collection_name=False,
                    #     source_properties=["full_text"],
                    #     vector_index_config=wvc.Configure.VectorIndex.hnsw(),
                    #     inference_url="http://t2v-transformers-fast:8080",
                    # ),
                ],
                inverted_index_config=wvc.Configure.inverted_index(
                    stopwords_preset=wvc.StopwordsPreset.EN,
                    stopwords_additions=POLISH_STOP_WORDS,
                ),
            )
            logger.info(f"Created {self.LEGAL_DOCUMENTS_COLLECTION} collection")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(
                    f"Collection {self.LEGAL_DOCUMENTS_COLLECTION} already exists, skipping creation"
                )
            else:
                logger.error(f"Failed to create {self.LEGAL_DOCUMENTS_COLLECTION} collection: {e}")
                raise

    def _create_document_chunk_class(self):
        """Create DocumentChunk class in Weaviate."""
        import weaviate.classes.config as wvc

        try:
            self.weaviate_client.collections.create(
                name=self.DOCUMENT_CHUNKS_COLLECTION,
                properties=[
                    wvc.Property(
                        name="document_id",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        description="Unique identifier for the document",
                        index_filterable=True,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="document_type",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        description="Type of the document, like judgment, tax_interpretation, etc.",
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="language",
                        description="Language of the document",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="chunk_id",
                        data_type=wvc.DataType.NUMBER,
                        skip_vectorization=True,
                        description="Unique identifier for the chunk",
                        index_filterable=False,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="chunk_text",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=False,
                        description="Text content of the chunk",
                        index_filterable=False,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="segment_type",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=False,
                        description="Type of the segment (e.g., paragraph, sentence, etc.)",
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="position",
                        data_type=wvc.DataType.NUMBER,
                        skip_vectorization=True,
                        description="Position of the chunk within the document",
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="source",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        description="Source of the chunk",
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="confidence_score",
                        data_type=wvc.DataType.NUMBER,
                        skip_vectorization=True,
                        description="Confidence score for the chunk's content",
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="cited_references",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        description="References cited within the chunk",
                        index_filterable=True,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="tags",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        description="Tags associated with the chunk",
                        index_filterable=True,
                        index_searchable=True,
                    ),
                    wvc.Property(
                        name="parent_segment_id",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                        description="ID of the parent segment if this chunk is part of a larger segment",
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="x",
                        data_type=wvc.DataType.NUMBER,
                        skip_vectorization=True,
                        description="X coordinate of the chunk",
                        index_filterable=False,
                        index_searchable=False,
                    ),
                    wvc.Property(
                        name="y",
                        data_type=wvc.DataType.NUMBER,
                        skip_vectorization=True,
                        description="Y coordinate of the chunk",
                        index_filterable=False,
                        index_searchable=False,
                    ),
                ],
                vectorizer_config=[
                    wvc.Configure.NamedVectors.text2vec_transformers(
                        name=VectorName.BASE,
                        vectorize_collection_name=False,
                        source_properties=["chunk_text"],
                        vector_index_config=wvc.Configure.VectorIndex.hnsw(),
                        inference_url="http://t2v-transformers-base:8080",
                    ),
                    # wvc.Configure.NamedVectors.text2vec_transformers(
                    #     name=VectorName.DEV,
                    #     vectorize_collection_name=False,
                    #     source_properties=["chunk_text"],
                    #     vector_index_config=wvc.Configure.VectorIndex.hnsw(),
                    #     inference_url="http://t2v-transformers-dev:8080",
                    # ),
                    # wvc.Configure.NamedVectors.text2vec_transformers(
                    #     name=VectorName.FAST,
                    #     vectorize_collection_name=False,
                    #     source_properties=["chunk_text"],
                    #     vector_index_config=wvc.Configure.VectorIndex.hnsw(),
                    #     inference_url="http://t2v-transformers-fast:8080",
                    # ),
                ],
                inverted_index_config=wvc.Configure.inverted_index(
                    stopwords_preset=wvc.StopwordsPreset.EN,
                    stopwords_additions=POLISH_STOP_WORDS,
                ),
            )
            logger.info(f"Created {self.DOCUMENT_CHUNKS_COLLECTION} collection")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(
                    f"Collection {self.DOCUMENT_CHUNKS_COLLECTION} already exists, skipping creation"
                )
            else:
                logger.error(f"Failed to create {self.DOCUMENT_CHUNKS_COLLECTION} collection: {e}")
                raise

    def _generate_embeddings(self, texts: List[str]) -> Dict[str, List[List[float]]]:
        """Generate embeddings for batch of texts using all models."""
        embeddings_dict = {}

        for vector_name, transformer in self.transformers.items():
            try:
                embeddings = transformer.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                embeddings_dict[vector_name] = embeddings.tolist()
                logger.debug(f"Generated {vector_name} embeddings for {len(texts)} texts")
            except Exception as e:
                logger.error(f"Failed to generate {vector_name} embeddings: {e}")
                # Return zero embeddings as fallback - get dimensions from model
                try:
                    dim = transformer.get_sentence_embedding_dimension()
                except Exception:
                    dim = 768  # Default fallback
                embeddings_dict[vector_name] = [[0.0] * dim for _ in texts]

        return embeddings_dict

    def _aggregate_embeddings(
        self, embeddings_dict: Dict[str, List[List[float]]]
    ) -> Dict[str, List[float]]:
        """Aggregate chunk embeddings into document embeddings for each vector."""
        aggregated_dict = {}

        for vector_name, embeddings in embeddings_dict.items():
            if not embeddings:
                # Get default size from the transformer
                try:
                    dim = self.transformers[vector_name].get_sentence_embedding_dimension()
                except Exception:
                    dim = 768  # Default fallback
                aggregated_dict[vector_name] = [0.0] * dim
            else:
                aggregated_dict[vector_name] = np.mean(embeddings, axis=0).tolist()

        return aggregated_dict

    def _apply_column_mapping(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply column mapping from dataset config to document data."""
        if not self.dataset_config or not self.dataset_config.column_mapping:
            return doc_data

        mapped_data = {}

        # Apply column mappings
        for source_field, target_field in self.dataset_config.column_mapping.items():
            if source_field in doc_data:
                mapped_data[target_field] = doc_data[source_field]

        # Copy unmapped fields that exist in target schema
        for field in doc_data:
            if field not in mapped_data:
                mapped_data[field] = doc_data[field]

        # Apply default values
        if self.dataset_config.default_values:
            for field, default_value in self.dataset_config.default_values.items():
                if field not in mapped_data or not mapped_data[field]:
                    mapped_data[field] = default_value

        return mapped_data

    def _serialize_field_value(self, value: Any) -> str:
        """Serialize complex field values to JSON strings."""
        if isinstance(value, (list, dict)):
            return json.dumps(value, default=str, ensure_ascii=False)
        elif isinstance(value, datetime):
            return convert_date_to_rfc3339(value)
        else:
            return str(value) if value is not None else ""

    def _ingest_document(
        self, doc_data: Dict[str, Any], embeddings: Dict[str, List[float]]
    ) -> bool:
        """Ingest single document to Weaviate."""
        try:
            # Apply column mapping
            mapped_data = self._apply_column_mapping(doc_data)

            # Generate deterministic UUID
            doc_id = mapped_data["document_id"]
            uuid = self._generate_uuid(doc_id)

            # Prepare document object with proper field mapping
            # Ensure legal_bases is a list of strings
            legal_bases = mapped_data.get("legal_bases", [])
            if not isinstance(legal_bases, list):
                legal_bases = [str(legal_bases)] if legal_bases else []
            else:
                legal_bases = [str(item) for item in legal_bases]

            doc_obj = {
                "document_id": doc_id,
                "document_type": self.dataset_config.document_type
                if self.dataset_config
                else mapped_data.get("document_type", ""),
                "title": mapped_data.get("title", ""),
                "date_issued": self._serialize_field_value(mapped_data.get("date_issued")),
                "document_number": mapped_data.get("document_number", ""),
                "language": mapped_data.get("language", ""),
                "country": mapped_data.get("country", ""),
                "full_text": mapped_data.get("full_text", ""),
                "summary": mapped_data.get("summary", ""),
                "thesis": mapped_data.get("thesis", ""),
                "keywords": mapped_data.get("keywords", []),
                "issuing_body": self._serialize_field_value(mapped_data.get("issuing_body", "")),
                "ingestion_date": convert_date_to_rfc3339(datetime.now()),
                "last_updated": convert_date_to_rfc3339(datetime.now()),
                "processing_status": "completed",
                "source_url": mapped_data.get("source_url", ""),
                "source": mapped_data.get("source", ""),
                "legal_references": self._serialize_field_value(
                    mapped_data.get("legal_references", [])
                ),
                "parties": self._serialize_field_value(mapped_data.get("parties", [])),
                "outcome": self._serialize_field_value(mapped_data.get("outcome", "")),
                "publication_date": self._serialize_field_value(
                    mapped_data.get("publication_date")
                ),
                "raw_content": mapped_data.get("raw_content", ""),
                "presiding_judge": mapped_data.get("presiding_judge", ""),
                "judges": mapped_data.get("judges", []),
                "legal_bases": legal_bases,
                "court_name": mapped_data.get("court_name", ""),
                "department_name": mapped_data.get("department_name", ""),
                "extracted_legal_bases": self._serialize_field_value(
                    mapped_data.get("extracted_legal_bases", [])
                ),
                "references": mapped_data.get("references", []),
                "metadata": json.dumps(
                    {
                        k: self._serialize_field_value(v)
                        for k, v in mapped_data.items()
                        if k not in ["document_id", "title", "full_text", "summary", "thesis"]
                    },
                    default=str,
                    ensure_ascii=False,
                ),
            }

            # Clean None and empty values
            doc_obj = {k: v for k, v in doc_obj.items() if v is not None and v != ""}

            # Get collection and insert with named vectors
            collection = self.weaviate_client.collections.get(self.LEGAL_DOCUMENTS_COLLECTION)
            collection.data.insert(properties=doc_obj, uuid=uuid, vector=embeddings)

            return True

        except Exception as e:
            # Log the error but don't mark as processed - let it retry next time
            logger.error(f"Failed to ingest document {doc_data.get('document_id', 'unknown')}: {e}")
            # Save error details to database for analysis
            self.tracker.mark_error(doc_data.get("document_id", "unknown"), str(e))
            return False

    def _ingest_chunks(
        self,
        chunks: List[TextChunk],
        embeddings_dict: Dict[str, List[List[float]]],
        doc_data: Dict[str, Any],
    ) -> bool:
        """Ingest document chunks to Weaviate."""
        try:
            collection = self.weaviate_client.collections.get(self.DOCUMENT_CHUNKS_COLLECTION)
            mapped_data = self._apply_column_mapping(doc_data)

            # Prepare objects for batch insert
            objects = []
            for i, chunk in enumerate(chunks):
                uuid = self._generate_uuid(chunk.chunk_id)

                chunk_obj = {
                    "document_id": chunk.document_id,
                    "document_type": self.dataset_config.document_type
                    if self.dataset_config
                    else mapped_data.get("document_type", ""),
                    "language": mapped_data.get("language", ""),
                    "chunk_id": chunk.position,  # Use position as numeric chunk_id
                    "chunk_text": chunk.text,
                    "segment_type": "content",  # Default segment type
                    "position": chunk.position,
                    "confidence_score": 1.0,  # Default confidence
                    "cited_references": "",  # Empty for now
                    "tags": "",  # Empty for now
                    "parent_segment_id": chunk.document_id,
                }

                # Create vector dictionary for this chunk across all models
                chunk_vectors = {}
                for vector_name, chunk_embeddings in embeddings_dict.items():
                    chunk_vectors[vector_name] = chunk_embeddings[i]

                objects.append({"properties": chunk_obj, "uuid": uuid, "vector": chunk_vectors})

            # Batch insert
            with collection.batch.dynamic() as batch:
                for obj in objects:
                    batch.add_object(
                        properties=obj["properties"], uuid=obj["uuid"], vector=obj["vector"]
                    )

            return True

        except Exception as e:
            logger.error(f"Failed to ingest chunks: {e}")
            return False

    def _generate_uuid(self, identifier: str) -> str:
        """Generate deterministic UUID from identifier."""
        uuid = weaviate.util.generate_uuid5(identifier)
        if uuid is None:
            raise ValueError(f"Failed to generate UUID for identifier: {identifier}")
        return uuid

    def _process_document(self, doc: Dict[str, Any]) -> bool:
        """Process a single document: chunk, embed, and ingest."""
        # Try multiple possible document ID fields
        doc_id = doc.get("document_id") or doc.get("judgment_id") or doc.get("id")
        if not doc_id:
            logger.warning("Document missing document_id/judgment_id/id field, skipping")
            return False

        # Check if already processed
        if self.tracker.is_processed(doc_id):
            self.stats.skipped_documents += 1
            return True

        try:
            # Extract text
            #  TODO: html contnet temporary for tax interpretations
            text = doc.get("full_text", "") or doc.get("html_content", "")
            if not text:
                logger.warning(f"Document {doc_id} has no text content")
                self.tracker.mark_processed(doc_id, 0, False)
                return False

            # Create chunks using TextChunker
            chunk_data = self.chunker({"document_id": [doc_id], "text": [text]})

            # Convert to TextChunk objects for compatibility
            chunks = []
            for i, (chunk_id, chunk_text) in enumerate(
                zip(chunk_data["chunk_id"], chunk_data["chunk_text"])
            ):
                chunks.append(
                    TextChunk(
                        document_id=doc_id,
                        chunk_id=f"{doc_id}_chunk_{chunk_id}",
                        text=chunk_text,
                        position=chunk_id,
                    )
                )

            if not chunks:
                logger.warning(f"Document {doc_id} produced no chunks")
                self.tracker.mark_processed(doc_id, 0, False)
                return False

            # Generate embeddings
            chunk_texts = [chunk.text for chunk in chunks]
            chunk_embeddings = self._generate_embeddings(chunk_texts)

            # Create document embedding
            doc_embedding = self._aggregate_embeddings(chunk_embeddings)

            # Prepare document data
            doc_data = dict(doc)
            doc_data["document_id"] = doc_id  # Ensure document_id is always present
            doc_data["chunks_count"] = len(chunks)

            # Ingest document
            doc_success = self._ingest_document(doc_data, doc_embedding)

            # Ingest chunks
            chunks_success = self._ingest_chunks(chunks, chunk_embeddings, doc_data)

            success = doc_success and chunks_success

            # Update tracker
            self.tracker.mark_processed(doc_id, len(chunks), success)

            # Update stats
            if success:
                self.stats.processed_documents += 1
                self.stats.total_chunks += len(chunks)
            else:
                self.stats.failed_documents += 1

            return success

        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}")
            self.tracker.mark_processed(doc_id, 0, False)
            self.stats.failed_documents += 1
            return False

    def process_dataset(self, dataset_path: str, streaming: bool = True) -> ProcessingStats:
        """Process entire dataset with streaming."""
        start_time = datetime.now()

        self.console.print(f"[bold green]Starting dataset processing: {dataset_path}[/bold green]")

        # Load dataset
        try:
            if streaming:
                dataset = load_dataset(dataset_path, split="train", streaming=True)
                # For streaming datasets, we can't get total count easily
                total_docs = None
            else:
                dataset = load_dataset(dataset_path, split="train")
                total_docs = len(dataset) if hasattr(dataset, "__len__") else None

            self.console.print(f"Dataset loaded. Streaming: {streaming}")

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_path}: {e}")
            return self.stats

        # Process documents
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn() if total_docs else TextColumn(""),
            TaskProgressColumn() if total_docs else TextColumn("{task.completed} processed"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Processing documents...", total=total_docs)

            for doc in dataset:
                self.stats.total_documents += 1
                self._process_document(dict(doc))
                progress.update(task, advance=1)

                # Progress logging every 100 documents
                if self.stats.total_documents % 100 == 0:
                    logger.info(f"Processed {self.stats.total_documents} documents")

        # Final statistics
        end_time = datetime.now()
        self.stats.processing_time = (end_time - start_time).total_seconds()

        self._print_final_stats()
        return self.stats

    def _print_final_stats(self):
        """Print final processing statistics."""
        tracker_stats = self.tracker.get_stats()

        self.console.print("\n[bold blue]Processing Complete![/bold blue]")
        self.console.print(f"Total documents: {self.stats.total_documents}")
        self.console.print(f"Processed: {self.stats.processed_documents}")
        self.console.print(f"Skipped (already processed): {self.stats.skipped_documents}")
        self.console.print(f"Failed: {self.stats.failed_documents}")
        self.console.print(f"Total chunks: {self.stats.total_chunks}")
        self.console.print(f"Processing time: {self.stats.processing_time:.2f} seconds")

        if tracker_stats["total"] > 0:
            self.console.print("\n[bold yellow]Tracker Statistics:[/bold yellow]")
            self.console.print(f"Total in tracker: {tracker_stats['total']}")
            self.console.print(f"Successful: {tracker_stats['successful']}")
            self.console.print(f"Failed: {tracker_stats['failed']}")

        # Show error documents if any
        if tracker_stats["failed"] > 0:
            self._print_error_documents()

    def _print_error_documents(self):
        """Print details about failed documents."""
        try:
            with sqlite3.connect(self.tracker.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT document_id, error_message 
                    FROM processed_documents 
                    WHERE success = FALSE AND error_message IS NOT NULL
                    ORDER BY document_id
                """
                )
                failed_docs = cursor.fetchall()

            if failed_docs:
                self.console.print("\n[bold red]Failed Documents:[/bold red]")
                for doc_id, error_msg in failed_docs:
                    # Truncate long error messages for display
                    display_error = error_msg[:100] + "..." if len(error_msg) > 100 else error_msg
                    self.console.print(f"  - {doc_id}: [red]{display_error}[/red]")

                if len(failed_docs) > 10:
                    self.console.print(
                        f"  [dim]... and {len(failed_docs) - 10} more failed documents[/dim]"
                    )

        except Exception as e:
            logger.error(f"Error retrieving failed documents: {e}")
            self.console.print(f"[red]Error retrieving failed documents: {e}[/red]")

    def delete_all_collections(self):
        """Delete all legal document collections from Weaviate."""
        try:
            collections_to_delete = [
                self.LEGAL_DOCUMENTS_COLLECTION,
                self.DOCUMENT_CHUNKS_COLLECTION,
            ]

            deleted_count = 0
            for collection_name in collections_to_delete:
                try:
                    self.weaviate_client.collections.delete(collection_name)
                    self.console.print(f"[green]✓ Deleted collection: {collection_name}[/green]")
                    deleted_count += 1
                except Exception as e:
                    if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                        self.console.print(
                            f"[dim]Collection {collection_name} does not exist, skipping[/dim]"
                        )
                    else:
                        self.console.print(f"[red]✗ Failed to delete {collection_name}: {e}[/red]")
                        logger.error(f"Error deleting collection {collection_name}: {e}")

            if deleted_count > 0:
                self.console.print(
                    f"[bold green]Successfully deleted {deleted_count} collection(s)[/bold green]"
                )
            else:
                self.console.print("[yellow]No collections were deleted[/yellow]")

        except Exception as e:
            self.console.print(f"[bold red]Error during collection deletion: {e}[/bold red]")
            logger.error(f"Failed to delete collections: {e}")
            raise

    def list_collections(self):
        """List all collections in Weaviate."""
        try:
            collections = self.weaviate_client.collections.list_all()

            try:
                if isinstance(collections, dict):
                    collection_names = list(collections.keys())
                else:
                    # Try to iterate and get names, handle different collection types
                    collection_names = []
                    for collection in collections:
                        if hasattr(collection, "name"):
                            collection_names.append(collection.name)
                        else:
                            collection_names.append(str(collection))
            except (TypeError, AttributeError):
                collection_names = []

            if collection_names:
                self.console.print("[bold cyan]Existing collections:[/bold cyan]")
                for name in sorted(collection_names):
                    # Get collection size if possible
                    try:
                        collection = self.weaviate_client.collections.get(name)
                        response = collection.aggregate.over_all(total_count=True)
                        count = response.total_count
                        self.console.print(f"  - {name} ({count:,} objects)")
                    except Exception:
                        self.console.print(f"  - {name}")
            else:
                self.console.print("[dim]No collections found[/dim]")

        except Exception as e:
            self.console.print(f"[bold red]Error listing collections: {e}[/bold red]")
            logger.error(f"Failed to list collections: {e}")
            raise

    def reset_tracker(self):
        """Reset the processed documents tracker."""
        Path(self.tracker.db_path).unlink(missing_ok=True)
        dataset_name = (
            self.dataset_config.name
            if self.dataset_config and hasattr(self.dataset_config, "name")
            else None
        )
        self.tracker = ProcessedDocTracker("processed_documents.db", dataset_name)
        self.console.print(
            f"[bold yellow]Tracker database reset: {self.tracker.db_path}[/bold yellow]"
        )

    def reset_tracker_for_dataset(self, pattern: str):
        """Reset tracker for documents matching a specific pattern."""
        deleted_count = self.tracker.reset_documents_by_pattern(pattern)
        self.console.print(
            f"[bold yellow]Reset {deleted_count} documents matching pattern '{pattern}'[/bold yellow]"
        )

    def close(self):
        """Close Weaviate connection."""
        try:
            self.weaviate_client.close()
            logger.info("Weaviate connection closed successfully")
        except Exception as e:
            logger.warning(f"Error closing Weaviate client: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def get_document_type_stats(self) -> Dict[str, int]:
        """Get statistics about processed document types using Weaviate Metrics."""

        try:
            collection = self.weaviate_client.collections.get(self.LEGAL_DOCUMENTS_COLLECTION)
            response = collection.aggregate.over_all(
                return_metrics=Metrics("document_type").text(count=True)
            )
            stats = {}
            logger.info(f"Document type stats response: {response}")
            doc_type_agg = response.properties.get("document_type")
            if doc_type_agg and hasattr(doc_type_agg, "top_occurrences"):
                for occ in doc_type_agg.top_occurrences:
                    stats[occ.value] = occ.count
            return stats
        except Exception as e:
            logger.error(f"Failed to get document type statistics: {e}")
            return {}

    def get_country_stats(self) -> Dict[str, int]:
        """Get statistics about processed documents by country using Weaviate Metrics."""
        try:
            collection = self.weaviate_client.collections.get(self.LEGAL_DOCUMENTS_COLLECTION)
            response = collection.aggregate.over_all(
                return_metrics=Metrics("country").text(count=True)
            )
            stats = {}
            logger.info(f"Country stats response: {response}")
            country_agg = response.properties.get("country")
            if country_agg and hasattr(country_agg, "top_occurrences"):
                for occ in country_agg.top_occurrences:
                    stats[occ.value] = occ.count
            return stats
        except Exception as e:
            logger.error(f"Failed to get country statistics: {e}")
            return {}

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            logger.info(f"Exiting context manager due to exception: {exc_type.__name__}")
        self.close()
        return False
