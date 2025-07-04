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
from typing import Any, Dict, List

import numpy as np
from datasets import load_dataset
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from sentence_transformers import SentenceTransformer

import weaviate
from juddges.settings import VectorName
from juddges.utils.date_utils import convert_date_to_rfc3339


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


class SimpleChunker:
    """Simple text chunker with configurable parameters."""

    def __init__(self, chunk_size: int = 512, overlap: int = 128, min_chunk_size: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(self, text: str, document_id: str) -> List[TextChunk]:
        """Split text into overlapping chunks."""
        if not text or len(text) < self.min_chunk_size:
            return []

        chunks = []
        start = 0
        position = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                chunk_id = f"{document_id}_chunk_{position}"
                chunks.append(
                    TextChunk(
                        document_id=document_id,
                        chunk_id=chunk_id,
                        text=chunk_text,
                        position=position,
                    )
                )
                position += 1

            if end >= len(text):
                break

            start = end - self.overlap

        return chunks


class ProcessedDocTracker:
    """Track processed documents using SQLite for resume capability."""

    def __init__(self, db_path: str = "processed_documents.db"):
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
                    success BOOLEAN DEFAULT TRUE
                )
            """)
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


class StreamingIngester:
    """Simplified streaming ingester for legal documents."""

    # Use same collection names as documents_weaviate_db.py
    LEGAL_DOCUMENTS_COLLECTION = "legal_documents"
    DOCUMENT_CHUNKS_COLLECTION = "document_chunks"

    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        embedding_model: str = "sdadas/mmlw-roberta-large",
        chunk_size: int = 512,
        overlap: int = 128,
        batch_size: int = 32,
        tracker_db: str = "processed_documents.db",
        dataset_config=None,
    ):
        self.console = Console()

        # Parse URL to get host and port
        url_parts = weaviate_url.split("://")[-1].split(":")
        host = url_parts[0]
        port = int(url_parts[1]) if len(url_parts) > 1 else 8080

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

        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunker = SimpleChunker(chunk_size, overlap)
        self.batch_size = batch_size
        self.tracker = ProcessedDocTracker(tracker_db)
        self.stats = ProcessingStats()
        self.dataset_config = dataset_config

        # Verify Weaviate connection
        self._verify_weaviate_connection()

        logger.info(f"Initialized StreamingIngester with {embedding_model}")

    def _verify_weaviate_connection(self):
        """Verify Weaviate connection and schema."""
        try:
            # Check if collections exist
            collections = self.weaviate_client.collections.list_all()

            # Handle different return types from list_all()
            if isinstance(collections, dict):
                collection_names = list(collections.keys())
            else:
                collection_names = [collection.name for collection in collections]

            if self.LEGAL_DOCUMENTS_COLLECTION not in collection_names:
                self._create_legal_document_class()
            if self.DOCUMENT_CHUNKS_COLLECTION not in collection_names:
                self._create_document_chunk_class()

            logger.info("Weaviate connection verified and schema ready")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise

    def _create_legal_document_class(self):
        """Create LegalDocument class in Weaviate."""
        import weaviate.classes.config as wvc

        # Use same schema as documents_weaviate_db.py but simplified for streaming
        self.weaviate_client.collections.create(
            name=self.LEGAL_DOCUMENTS_COLLECTION,
            properties=[
                wvc.Property(
                    name="document_id", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(
                    name="document_type", data_type=wvc.DataType.TEXT, skip_vectorization=False
                ),
                wvc.Property(name="title", data_type=wvc.DataType.TEXT, skip_vectorization=False),
                wvc.Property(
                    name="date_issued", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(
                    name="document_number", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(name="language", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(name="country", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(
                    name="full_text", data_type=wvc.DataType.TEXT, skip_vectorization=False
                ),
                wvc.Property(name="summary", data_type=wvc.DataType.TEXT, skip_vectorization=False),
                wvc.Property(name="thesis", data_type=wvc.DataType.TEXT, skip_vectorization=False),
                wvc.Property(
                    name="keywords", data_type=wvc.DataType.TEXT_ARRAY, skip_vectorization=False
                ),
                wvc.Property(
                    name="issuing_body", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(
                    name="ingestion_date", data_type=wvc.DataType.DATE, skip_vectorization=True
                ),
                wvc.Property(
                    name="last_updated", data_type=wvc.DataType.DATE, skip_vectorization=True
                ),
                wvc.Property(
                    name="processing_status", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(
                    name="source_url", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(
                    name="legal_references", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(name="parties", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(name="outcome", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(name="metadata", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                # Additional properties from dataset mappings
                wvc.Property(
                    name="publication_date", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(
                    name="raw_content", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(
                    name="presiding_judge", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(name="judges", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(
                    name="legal_bases", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(
                    name="court_name", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(
                    name="department_name", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(
                    name="extracted_legal_bases",
                    data_type=wvc.DataType.TEXT,
                    skip_vectorization=True,
                ),
                wvc.Property(
                    name="references", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(name="x", data_type=wvc.DataType.NUMBER, skip_vectorization=True),
                wvc.Property(name="y", data_type=wvc.DataType.NUMBER, skip_vectorization=True),
            ],
            vectorizer_config=[
                wvc.Configure.NamedVectors.text2vec_transformers(
                    name=VectorName.BASE,
                    vectorize_collection_name=False,
                    source_properties=["full_text"],
                    vector_index_config=wvc.Configure.VectorIndex.hnsw(),
                ),
                wvc.Configure.NamedVectors.text2vec_transformers(
                    name=VectorName.DEV,
                    vectorize_collection_name=False,
                    source_properties=["full_text"],
                    vector_index_config=wvc.Configure.VectorIndex.hnsw(),
                ),
                wvc.Configure.NamedVectors.text2vec_transformers(
                    name=VectorName.FAST,
                    vectorize_collection_name=False,
                    source_properties=["full_text"],
                    vector_index_config=wvc.Configure.VectorIndex.hnsw(),
                ),
            ],
        )
        logger.info(f"Created {self.LEGAL_DOCUMENTS_COLLECTION} collection")

    def _create_document_chunk_class(self):
        """Create DocumentChunk class in Weaviate."""
        import weaviate.classes.config as wvc

        # Use same schema as documents_weaviate_db.py
        self.weaviate_client.collections.create(
            name=self.DOCUMENT_CHUNKS_COLLECTION,
            properties=[
                wvc.Property(
                    name="document_id", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(
                    name="document_type", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(name="language", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(
                    name="chunk_id", data_type=wvc.DataType.NUMBER, skip_vectorization=True
                ),
                wvc.Property(
                    name="chunk_text", data_type=wvc.DataType.TEXT, skip_vectorization=False
                ),
                wvc.Property(
                    name="segment_type", data_type=wvc.DataType.TEXT, skip_vectorization=False
                ),
                wvc.Property(
                    name="position", data_type=wvc.DataType.NUMBER, skip_vectorization=True
                ),
                wvc.Property(
                    name="confidence_score", data_type=wvc.DataType.NUMBER, skip_vectorization=True
                ),
                wvc.Property(
                    name="cited_references", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(name="tags", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(
                    name="parent_segment_id", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(name="x", data_type=wvc.DataType.NUMBER, skip_vectorization=True),
                wvc.Property(name="y", data_type=wvc.DataType.NUMBER, skip_vectorization=True),
            ],
            vectorizer_config=[
                wvc.Configure.NamedVectors.text2vec_transformers(
                    name=VectorName.BASE,
                    vectorize_collection_name=False,
                    source_properties=["full_text"],
                    vector_index_config=wvc.Configure.VectorIndex.hnsw(),
                ),
                wvc.Configure.NamedVectors.text2vec_transformers(
                    name=VectorName.DEV,
                    vectorize_collection_name=False,
                    source_properties=["full_text"],
                    vector_index_config=wvc.Configure.VectorIndex.hnsw(),
                ),
                wvc.Configure.NamedVectors.text2vec_transformers(
                    name=VectorName.FAST,
                    vectorize_collection_name=False,
                    source_properties=["full_text"],
                    vector_index_config=wvc.Configure.VectorIndex.hnsw(),
                ),
            ],
        )
        logger.info(f"Created {self.DOCUMENT_CHUNKS_COLLECTION} collection")

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        try:
            embeddings = self.embedding_model.encode(
                texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * 768 for _ in texts]

    def _aggregate_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Aggregate chunk embeddings into document embedding."""
        if not embeddings:
            return [0.0] * 768  # Default embedding size

        return np.mean(embeddings, axis=0).tolist()

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

    def _ingest_document(self, doc_data: Dict[str, Any], embedding: List[float]) -> bool:
        """Ingest single document to Weaviate."""
        try:
            # Apply column mapping
            mapped_data = self._apply_column_mapping(doc_data)

            # Generate deterministic UUID
            doc_id = mapped_data["document_id"]
            uuid = self._generate_uuid(doc_id)

            # Prepare document object with proper field mapping
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
                "keywords": self._serialize_field_value(mapped_data.get("keywords", [])),
                "issuing_body": self._serialize_field_value(mapped_data.get("issuing_body", "")),
                "ingestion_date": convert_date_to_rfc3339(datetime.now()),
                "last_updated": convert_date_to_rfc3339(datetime.now()),
                "processing_status": "completed",
                "source_url": mapped_data.get("source_url", ""),
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
                "judges": self._serialize_field_value(mapped_data.get("judges", [])),
                "legal_bases": self._serialize_field_value(mapped_data.get("legal_bases", [])),
                "court_name": mapped_data.get("court_name", ""),
                "department_name": mapped_data.get("department_name", ""),
                "extracted_legal_bases": self._serialize_field_value(
                    mapped_data.get("extracted_legal_bases", [])
                ),
                "references": self._serialize_field_value(mapped_data.get("references", [])),
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

            # Get collection and insert
            collection = self.weaviate_client.collections.get(self.LEGAL_DOCUMENTS_COLLECTION)
            collection.data.insert(properties=doc_obj, uuid=uuid, vector=embedding)

            return True

        except Exception as e:
            logger.error(f"Failed to ingest document {doc_data.get('document_id', 'unknown')}: {e}")
            return False

    def _ingest_chunks(
        self, chunks: List[TextChunk], embeddings: List[List[float]], doc_data: Dict[str, Any]
    ) -> bool:
        """Ingest document chunks to Weaviate."""
        try:
            collection = self.weaviate_client.collections.get(self.DOCUMENT_CHUNKS_COLLECTION)
            mapped_data = self._apply_column_mapping(doc_data)

            # Prepare objects for batch insert
            objects = []
            for chunk, embedding in zip(chunks, embeddings):
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

                objects.append({"properties": chunk_obj, "uuid": uuid, "vector": embedding})

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
        return weaviate.util.generate_uuid5(identifier)

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
            text = doc.get("full_text", "")
            if not text:
                logger.warning(f"Document {doc_id} has no text content")
                self.tracker.mark_processed(doc_id, 0, False)
                return False

            # Create chunks
            chunks = self.chunker.chunk_text(text, doc_id)
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
                total_docs = len(dataset)

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
                self._process_document(doc)
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

    def reset_tracker(self):
        """Reset the processed documents tracker."""
        Path(self.tracker.db_path).unlink(missing_ok=True)
        self.tracker = ProcessedDocTracker(self.tracker.db_path)
        self.console.print("[bold yellow]Tracker database reset[/bold yellow]")

    def close(self):
        """Close Weaviate connection."""
        try:
            self.weaviate_client.close()
        except Exception as e:
            logger.warning(f"Error closing Weaviate client: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
