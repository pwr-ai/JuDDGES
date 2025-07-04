"""
Universal dataset processor for handling any HuggingFace dataset.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset
from loguru import logger

from juddges.data.config import IngestConfig
from juddges.data.converters import RobustDataConverter
from juddges.data.dataset_registry import DatasetConfig, DatasetRegistry, get_registry
from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase
from juddges.data.ingesters import ChunkIngester, DocumentIngester
from juddges.data.schema_adapter import WeaviateSchemaAdapter
from juddges.data.smart_mapper import SmartColumnMapper


@dataclass
class ProcessResult:
    """Result of dataset processing."""

    success: bool
    dataset_name: str
    total_rows: int
    processed_rows: int
    errors: List[str]
    warnings: List[str]
    ingested_documents: int = 0
    ingested_chunks: int = 0
    processing_time_seconds: float = 0.0


@dataclass
class DatasetPreview:
    """Preview information about a dataset."""

    dataset_name: str
    total_rows: int
    columns: List[str]
    sample_rows: List[Dict[str, Any]]
    suggested_mapping: Dict[str, str]
    suggested_config: DatasetConfig
    schema_compatibility: Dict[str, Any]
    estimated_processing_time: float


class UniversalDatasetProcessor:
    """Process any HuggingFace dataset for Weaviate ingestion."""

    def __init__(
        self,
        registry: Optional[DatasetRegistry] = None,
        schema_adapter: Optional[WeaviateSchemaAdapter] = None,
        converter: Optional[RobustDataConverter] = None,
        mapper: Optional[SmartColumnMapper] = None,
    ):
        """Initialize the universal processor."""
        self.registry = registry or get_registry()
        self.schema_adapter = schema_adapter or WeaviateSchemaAdapter()
        self.converter = converter or RobustDataConverter()
        self.mapper = mapper or SmartColumnMapper()

        logger.info("Initialized UniversalDatasetProcessor")

    def preview_dataset(
        self, dataset_name: str, sample_size: int = 10, force_reload: bool = False
    ) -> DatasetPreview:
        """Preview a dataset and suggest configuration."""
        logger.info(f"Creating preview for dataset: {dataset_name}")

        try:
            # Load a small sample for analysis
            dataset = self._load_dataset_sample(dataset_name, sample_size)

            # Get basic dataset info
            total_rows = self._get_dataset_size(dataset_name)
            columns = dataset.column_names
            sample_rows = dataset.to_list()

            # Check for existing configuration
            existing_config = self.registry.get_config(dataset_name)

            if existing_config and not force_reload:
                logger.info(f"Using existing configuration for {dataset_name}")
                suggested_config = existing_config
                suggested_mapping = existing_config.column_mapping
            else:
                # Generate automatic configuration
                logger.info(f"Generating automatic configuration for {dataset_name}")
                suggested_mapping = self._suggest_mapping(columns, sample_rows)
                suggested_config = self._create_auto_config(
                    dataset_name, columns, sample_rows, suggested_mapping
                )

            # Check schema compatibility
            schema_compatibility = self._check_schema_compatibility(
                suggested_config, sample_rows[0] if sample_rows else {}
            )

            # Estimate processing time
            estimated_time = self._estimate_processing_time(total_rows, len(columns))

            return DatasetPreview(
                dataset_name=dataset_name,
                total_rows=total_rows,
                columns=columns,
                sample_rows=sample_rows,
                suggested_mapping=suggested_mapping,
                suggested_config=suggested_config,
                schema_compatibility=schema_compatibility,
                estimated_processing_time=estimated_time,
            )

        except Exception as e:
            logger.error(f"Failed to create preview for {dataset_name}: {e}")
            raise

    def process_dataset(
        self,
        dataset_name: str,
        config: Optional[DatasetConfig] = None,
        ingest_config: Optional[IngestConfig] = None,
        create_embeddings: bool = True,
        preview_only: bool = False,
    ) -> ProcessResult:
        """Main entry point for processing any HF dataset."""
        import time

        start_time = time.time()

        logger.info(f"Processing dataset: {dataset_name}")

        try:
            # Get or create configuration
            if config is None:
                config = self.registry.get_config(dataset_name)
                if config is None:
                    logger.info(
                        f"No existing config found, creating automatic config for {dataset_name}"
                    )
                    preview = self.preview_dataset(dataset_name)
                    config = preview.suggested_config
                    # Register the auto-generated config
                    self.registry.register_dataset(config)

            if preview_only:
                preview = self.preview_dataset(dataset_name)
                return ProcessResult(
                    success=True,
                    dataset_name=dataset_name,
                    total_rows=preview.total_rows,
                    processed_rows=0,
                    errors=[],
                    warnings=["Preview mode - no data processed"],
                    processing_time_seconds=time.time() - start_time,
                )

            # Load the full dataset (with max_documents limit if specified)
            max_docs = (
                ingest_config.max_documents
                if ingest_config and ingest_config.max_documents
                else None
            )
            if max_docs:
                logger.info(f"Loading dataset: {dataset_name} (limited to {max_docs} documents)")
            else:
                logger.info(f"Loading full dataset: {dataset_name}")
            dataset = self._load_full_dataset(dataset_name, max_documents=max_docs)

            # Process the dataset
            result = self._process_full_dataset(dataset, config, ingest_config, create_embeddings)
            result.processing_time_seconds = time.time() - start_time

            return result

        except Exception as e:
            error_msg = f"Failed to process dataset {dataset_name}: {e}"
            logger.error(error_msg)
            return ProcessResult(
                success=False,
                dataset_name=dataset_name,
                total_rows=0,
                processed_rows=0,
                errors=[error_msg],
                warnings=[],
                processing_time_seconds=time.time() - start_time,
            )

    def _load_dataset_sample(self, dataset_name: str, sample_size: int) -> Dataset:
        """Load a small sample of the dataset for analysis."""
        try:
            # Try to load just a small sample first
            dataset = load_dataset(dataset_name, split="train", streaming=True)
            sample_data = []

            for i, item in enumerate(dataset):
                if i >= sample_size:
                    break
                sample_data.append(item)

            # Convert to Dataset
            if sample_data:
                return Dataset.from_list(sample_data)
            else:
                # Fallback to loading small portion
                dataset = load_dataset(dataset_name, split=f"train[:{sample_size}]")
                return dataset

        except Exception as e:
            logger.warning(f"Could not load streaming sample, trying direct load: {e}")
            # Fallback to direct loading
            return load_dataset(dataset_name, split=f"train[:{sample_size}]")

    def _load_full_dataset(self, dataset_name: str, max_documents: Optional[int] = None) -> Dataset:
        """Load the complete dataset, optionally limited to max_documents."""
        if max_documents:
            return load_dataset(dataset_name, split=f"train[:{max_documents}]")
        else:
            return load_dataset(dataset_name, split="train")

    def _get_dataset_size(self, dataset_name: str) -> int:
        """Get the total size of a dataset efficiently."""
        try:
            # Try to get info without downloading
            dataset_info = load_dataset(dataset_name, split="train")
            return dataset_info.num_rows
        except Exception:
            # Fallback - this will be less accurate
            return 1000  # Default estimate

    def _suggest_mapping(self, columns: List[str], sample_rows: List[Dict]) -> Dict[str, str]:
        """Suggest column mapping for the dataset."""
        # Use the smart mapper to suggest mappings
        suggestions = self.mapper.suggest_mapping(
            columns, required_fields=["document_id", "full_text"]
        )

        # Convert suggestions to simple mapping
        mapping = {}
        for column, suggestion in suggestions.items():
            mapping[column] = suggestion.target_field

        logger.info(f"Suggested mapping: {mapping}")
        return mapping

    def _create_auto_config(
        self,
        dataset_name: str,
        columns: List[str],
        sample_rows: List[Dict],
        mapping: Dict[str, str],
    ) -> DatasetConfig:
        """Create automatic configuration for unknown dataset."""

        # Analyze sample data to categorize fields
        text_fields = []
        date_fields = []
        array_fields = []
        json_fields = []

        sample_row = sample_rows[0] if sample_rows else {}

        for column in columns:
            if column in sample_row:
                field_suggestions = self.mapper.suggest_field_types(column, [sample_row[column]])

                if field_suggestions.get("is_text_field") or field_suggestions.get(
                    "should_vectorize"
                ):
                    target_field = mapping.get(column, column)
                    text_fields.append(target_field)

                if field_suggestions.get("is_date_field"):
                    target_field = mapping.get(column, column)
                    date_fields.append(target_field)

                if field_suggestions.get("is_array_field"):
                    target_field = mapping.get(column, column)
                    array_fields.append(target_field)

                if field_suggestions.get("is_json_field"):
                    target_field = mapping.get(column, column)
                    json_fields.append(target_field)

        # Determine document type and defaults
        document_type = "judgment"  # Default
        default_values = {}

        # Try to infer language and country
        if "language" not in mapping.values():
            # Try to infer from dataset name or sample content
            if "pl" in dataset_name.lower() or "polish" in dataset_name.lower():
                default_values["language"] = "pl"
                default_values["country"] = "Poland"
            elif "en" in dataset_name.lower() or "english" in dataset_name.lower():
                default_values["language"] = "en"
                default_values["country"] = "England"

        return DatasetConfig(
            name=dataset_name,
            column_mapping=mapping,
            required_fields=["document_id", "full_text"],
            text_fields=text_fields,
            date_fields=date_fields,
            array_fields=array_fields,
            json_fields=json_fields,
            default_values=default_values,
            document_type=document_type,
        )

    def _check_schema_compatibility(
        self, config: DatasetConfig, sample_data: Dict
    ) -> Dict[str, Any]:
        """Check compatibility with existing Weaviate schema."""
        try:
            # For now, we'll assume compatibility (in a real implementation,
            # you'd check against existing Weaviate schema)
            return {
                "compatible": True,
                "new_properties": [],
                "conflicts": [],
                "recommendations": [],
            }
        except Exception as e:
            return {
                "compatible": False,
                "error": str(e),
                "recommendations": ["Review dataset configuration"],
            }

    def _estimate_processing_time(self, total_rows: int, num_columns: int) -> float:
        """Estimate processing time based on dataset size."""
        # Simple estimation formula (adjust based on actual performance)
        base_time_per_row = 0.01  # seconds
        column_factor = num_columns * 0.001
        estimated_seconds = total_rows * (base_time_per_row + column_factor)

        # Add overhead for embeddings and ingestion
        estimated_seconds *= 1.5

        return estimated_seconds

    def _process_full_dataset(
        self,
        dataset: Dataset,
        config: DatasetConfig,
        ingest_config: Optional[IngestConfig],
        create_embeddings: bool,
    ) -> ProcessResult:
        """Process the complete dataset."""
        logger.info(f"Processing {dataset.num_rows} rows from {config.name}")

        # Apply column mapping and conversion
        converted_dataset = self._convert_dataset(dataset, config)

        if not create_embeddings:
            # Just convert and validate, don't create embeddings or ingest
            return ProcessResult(
                success=True,
                dataset_name=config.name,
                total_rows=dataset.num_rows,
                processed_rows=converted_dataset.num_rows,
                errors=[],
                warnings=["Embeddings creation skipped"],
            )

        # Create embeddings (this would integrate with existing embedding pipeline)
        embeddings_dataset = self._create_embeddings(converted_dataset, config)

        # Ingest to Weaviate
        ingestion_result = self._ingest_to_weaviate(embeddings_dataset, config, ingest_config)

        return ingestion_result

    def _convert_dataset(self, dataset: Dataset, config: DatasetConfig) -> Dataset:
        """Convert dataset using the robust converter."""
        logger.info("Converting dataset columns and data types")

        def convert_batch(batch):
            result = self.converter.convert_dataset_batch(batch, config)
            if not result.success:
                logger.warning(f"Conversion errors: {result.errors}")
            return result.converted_data

        # Apply conversion in batches
        converted_dataset = dataset.map(
            convert_batch,
            batched=True,
            batch_size=config.batch_size,
            desc="Converting dataset",
            num_proc=config.num_proc,
        )

        return converted_dataset

    def _create_embeddings(self, dataset: Dataset, config: DatasetConfig) -> Dataset:
        """Create embeddings for the dataset or load pre-computed ones."""
        logger.info("Creating embeddings for dataset")

        # Check if pre-computed chunk embeddings are available
        if config.chunks_path:
            chunks_path = Path(config.chunks_path)
            if chunks_path.exists():
                logger.info(f"Loading pre-computed chunk embeddings from {chunks_path}")
                try:
                    return self._load_chunk_embeddings(dataset, config, chunks_path)
                except Exception as e:
                    logger.error(f"Failed to load pre-computed chunk embeddings: {e}")
                    logger.info("Falling back to document-level embeddings")

        # Check if pre-computed embeddings path is specified in config
        if config.embedding_path:
            embeddings_path = Path(config.embedding_path)
            if embeddings_path.exists():
                logger.info(f"Loading pre-computed embeddings from {embeddings_path}")
                try:
                    # Load the embeddings dataset from parquet files
                    # The embeddings are saved as weighted_avg_embeddings_{i}.parquet files
                    embeddings_dataset = load_dataset(
                        "parquet", data_dir=str(embeddings_path), split="train"
                    )

                    # Determine the document ID column name in embeddings
                    embedding_columns = embeddings_dataset.column_names
                    doc_id_col = (
                        "document_id" if "document_id" in embedding_columns else "judgment_id"
                    )

                    # Rename the document ID column in embeddings to match the main dataset if needed
                    if doc_id_col != "document_id":
                        embeddings_dataset = embeddings_dataset.rename_column(
                            doc_id_col, "document_id"
                        )

                    logger.info(
                        f"Dataset has {len(dataset)} documents, embeddings have {len(embeddings_dataset)} entries"
                    )

                    # Filter embeddings if there are more embeddings than documents in dataset
                    if len(embeddings_dataset) > len(dataset):
                        doc_ids_needed = set(dataset["document_id"])
                        logger.info(
                            f"Filtering embeddings to only include {len(doc_ids_needed)} needed documents"
                        )

                        embeddings_dataset = embeddings_dataset.filter(
                            lambda x: x["document_id"] in doc_ids_needed,
                            desc="Filtering embeddings by document_id",
                            num_proc=config.num_proc,
                        )
                        logger.info(
                            f"Filtered embeddings dataset to {len(embeddings_dataset)} entries"
                        )

                    # Create document_id to embedding mapping
                    doc_id_to_embedding = {
                        row["document_id"]: row["embedding"] for row in embeddings_dataset
                    }

                    def add_embeddings_vectorized(examples):
                        embeddings = []
                        for doc_id in examples["document_id"]:
                            if doc_id in doc_id_to_embedding:
                                embeddings.append(doc_id_to_embedding[doc_id])
                            else:
                                logger.warning(f"No embedding found for document_id: {doc_id}")
                                embeddings.append([0.1] * config.vector_size)
                        examples["embedding"] = embeddings
                        return examples

                    result_dataset = dataset.map(
                        add_embeddings_vectorized,
                        batched=True,
                        batch_size=config.batch_size,
                        num_proc=config.num_proc,
                    )
                    logger.info(
                        f"Successfully loaded pre-computed embeddings for {len(result_dataset)} documents"
                    )
                    return result_dataset

                except Exception as e:
                    logger.error(f"Failed to load pre-computed embeddings: {e}")
                    logger.info("Falling back to mock embeddings")
            else:
                logger.warning(f"Embedding path specified but doesn't exist: {embeddings_path}")
                logger.info("Falling back to mock embeddings")

        # Fallback to mock embeddings if no pre-computed ones exist
        logger.info("No pre-computed embeddings found, using mock embeddings")

        def add_mock_embeddings(batch):
            batch_size = len(batch["document_id"])
            # Mock embeddings with configurable vector size
            batch["embedding"] = [[0.1] * config.vector_size for _ in range(batch_size)]
            return batch

        return dataset.map(
            add_mock_embeddings,
            batched=True,
            batch_size=config.batch_size,
            num_proc=config.num_proc,
        )

    def _load_chunk_embeddings(
        self, dataset: Dataset, config: DatasetConfig, chunks_path: Path
    ) -> Dataset:
        """Load pre-computed chunk embeddings without aggregation."""
        logger.info(f"Loading chunk embeddings from {chunks_path}")

        # Load the chunk embeddings dataset from parquet files
        chunks_dataset = load_dataset("parquet", data_dir=str(chunks_path), split="train")

        # Determine the document ID column name in chunks
        chunk_columns = chunks_dataset.column_names
        doc_id_col = (
            "document_id"
            if "document_id" in chunk_columns
            else "judgment_id"
            if "judgment_id" in chunk_columns
            else None
        )

        if doc_id_col is None:
            raise ValueError("Chunk embeddings must contain 'document_id' or 'judgment_id' column")

        # Rename the document ID column in chunks to match the main dataset if needed
        if doc_id_col != "document_id":
            chunks_dataset = chunks_dataset.rename_column(doc_id_col, "document_id")

        # Validate required columns for chunk embeddings
        required_chunk_columns = ["document_id", "chunk_id", "chunk_text", "embedding"]
        missing_columns = [
            col for col in required_chunk_columns if col not in chunks_dataset.column_names
        ]
        if missing_columns:
            raise ValueError(f"Chunk embeddings missing required columns: {missing_columns}")

        logger.info(
            f"Dataset has {len(dataset)} documents, chunk embeddings have {len(chunks_dataset)} chunks"
        )

        # Filter chunks to only include those from documents in the current dataset
        doc_ids_needed = set(dataset["document_id"])
        logger.info(
            f"Filtering chunks to only include documents in dataset ({len(doc_ids_needed)} documents)"
        )

        filtered_chunks = chunks_dataset.filter(
            lambda x: x["document_id"] in doc_ids_needed,
            desc="Filtering chunks by document_id",
            num_proc=config.num_proc,
        )

        logger.info(f"Filtered chunks dataset to {len(filtered_chunks)} chunks")

        # Pre-group chunks by document_id for efficient lookup and create document embeddings
        logger.info("Pre-grouping chunks by document_id for efficient lookup")
        chunks_by_doc_id = {}
        for chunk in filtered_chunks:
            doc_id = chunk["document_id"]
            if doc_id not in chunks_by_doc_id:
                chunks_by_doc_id[doc_id] = {
                    "chunk_texts": [],
                    "chunk_ids": [],
                    "chunk_embeddings": []
                }
            chunks_by_doc_id[doc_id]["chunk_texts"].append(chunk["chunk_text"])
            chunks_by_doc_id[doc_id]["chunk_ids"].append(chunk["chunk_id"])
            chunks_by_doc_id[doc_id]["chunk_embeddings"].append(chunk["embedding"])

        # Create document-level embeddings by aggregating chunk embeddings
        logger.info("Creating document-level embeddings from chunk embeddings")
        import numpy as np
        
        doc_embeddings = {}
        for doc_id, doc_chunks in chunks_by_doc_id.items():
            # Calculate mean embedding for this document
            chunk_embs = doc_chunks["chunk_embeddings"]
            if chunk_embs:
                avg_embedding = np.mean(chunk_embs, axis=0).tolist()
                doc_embeddings[doc_id] = avg_embedding
            else:
                # Fallback to zero embedding if no chunks
                doc_embeddings[doc_id] = [0.1] * config.vector_size

        def add_chunk_data_and_embeddings(examples):
            chunk_texts = []
            chunk_ids = []
            chunk_embeddings = []
            document_embeddings = []

            for doc_id in examples["document_id"]:
                # Get chunks for this document using efficient lookup
                if doc_id in chunks_by_doc_id:
                    chunk_texts.append(chunks_by_doc_id[doc_id]["chunk_texts"])
                    chunk_ids.append(chunks_by_doc_id[doc_id]["chunk_ids"])
                    chunk_embeddings.append(chunks_by_doc_id[doc_id]["chunk_embeddings"])
                else:
                    # No chunks for this document
                    chunk_texts.append([])
                    chunk_ids.append([])
                    chunk_embeddings.append([])
                
                # Add document-level embedding
                if doc_id in doc_embeddings:
                    document_embeddings.append(doc_embeddings[doc_id])
                else:
                    logger.warning(f"No embedding found for document_id: {doc_id}")
                    document_embeddings.append([0.1] * config.vector_size)

            examples["chunk_text"] = chunk_texts
            examples["chunk_id"] = chunk_ids
            examples["chunk_embedding"] = chunk_embeddings
            examples["embedding"] = document_embeddings
            return examples

        # Add chunk data and embeddings to the main dataset
        result_dataset = dataset.map(
            add_chunk_data_and_embeddings,
            batched=True,
            batch_size=config.batch_size,
            num_proc=config.num_proc,
        )

        # Store chunk data as a separate dataset for ingestion
        # Keep document_id for consistency across all collections
        chunk_ingestion_data = filtered_chunks

        # Add chunk dataset as an attribute of the main dataset for later use
        if len(chunk_ingestion_data) > 0:
            result_dataset._chunk_dataset = chunk_ingestion_data
            logger.info(
                f"Successfully loaded chunk embeddings for {len(result_dataset)} documents with {len(chunk_ingestion_data)} chunks"
            )
        else:
            logger.warning("No chunk embeddings found for any documents in the dataset")

        return result_dataset

    def _create_document_embeddings_from_chunks(
        self, chunks_dataset: Dataset
    ) -> Dataset:
        """Create document-level embeddings by aggregating chunk embeddings."""
        logger.info("Creating document-level embeddings from chunks")

        # Group chunks by document_id and aggregate embeddings
        doc_embeddings = {}
        for chunk in chunks_dataset:
            doc_id = chunk["document_id"]
            if doc_id not in doc_embeddings:
                doc_embeddings[doc_id] = []
            doc_embeddings[doc_id].append(chunk["embedding"])

        # Calculate weighted average for each document
        aggregated_embeddings = []
        for doc_id, chunk_embs in doc_embeddings.items():
            # Simple mean aggregation (could be weighted by chunk length/importance)
            import numpy as np

            avg_embedding = np.mean(chunk_embs, axis=0).tolist()
            aggregated_embeddings.append({"document_id": doc_id, "embedding": avg_embedding})

        return Dataset.from_list(aggregated_embeddings)

    def _ingest_to_weaviate(
        self,
        embeddings_dataset: Dataset,
        config: DatasetConfig,
        ingest_config: Optional[IngestConfig],
    ) -> ProcessResult:
        """Ingest processed dataset to Weaviate."""
        logger.info("Ingesting dataset to Weaviate")

        if ingest_config is None:
            ingest_config = IngestConfig()

        errors = []
        warnings = []
        ingested_docs = 0
        ingested_chunks = 0

        try:
            with WeaviateLegalDocumentsDatabase() as db:
                # Ensure schema exists and is up to date
                logger.info("Checking and updating Weaviate schema")
                try:
                    # Ensure collections exist in Weaviate
                    db.create_collections()
                    logger.info("Schema validation completed")
                except Exception as schema_error:
                    logger.error(f"Schema creation/validation failed: {schema_error}")
                    raise

                # Ingest documents
                logger.info(
                    f"Starting document ingestion for {embeddings_dataset.num_rows} documents"
                )
                doc_ingester = DocumentIngester(db=db, config=ingest_config)
                ingested_docs = doc_ingester.ingest(embeddings_dataset)
                logger.info(f"Successfully ingested {ingested_docs} documents")

                # Create and ingest chunks (if chunk dataset is available)
                if (
                    hasattr(embeddings_dataset, "_chunk_dataset")
                    and embeddings_dataset._chunk_dataset is not None
                ):
                    logger.info("Starting chunk ingestion from pre-computed chunks")
                    chunk_ingester = ChunkIngester(db=db, config=ingest_config)
                    ingested_chunks = chunk_ingester.ingest(embeddings_dataset._chunk_dataset)
                    logger.info(f"Successfully ingested {ingested_chunks} chunks")
                elif "chunk_text" in embeddings_dataset.column_names:
                    logger.info("Starting chunk ingestion from dataset chunks")
                    chunk_ingester = ChunkIngester(db=db, config=ingest_config)
                    ingested_chunks = chunk_ingester.ingest(embeddings_dataset)
                    logger.info(f"Successfully ingested {ingested_chunks} chunks")
                else:
                    ingested_chunks = 0
                    logger.info("No chunk data found, skipping chunk ingestion")

        except Exception as e:
            error_msg = f"Ingestion failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

        return ProcessResult(
            success=len(errors) == 0,
            dataset_name=config.name,
            total_rows=embeddings_dataset.num_rows,
            processed_rows=embeddings_dataset.num_rows,
            errors=errors,
            warnings=warnings,
            ingested_documents=ingested_docs,
            ingested_chunks=ingested_chunks,
        )

    def register_new_dataset(
        self, dataset_name: str, config_overrides: Optional[Dict[str, Any]] = None
    ) -> DatasetConfig:
        """Register a new dataset with automatic configuration."""
        logger.info(f"Registering new dataset: {dataset_name}")

        # Create preview to generate auto-config
        preview = self.preview_dataset(dataset_name)
        config = preview.suggested_config

        # Apply any manual overrides
        if config_overrides:
            config_dict = config.to_dict()
            config_dict.update(config_overrides)
            config = DatasetConfig.from_dict(config_dict)

        # Register the configuration
        self.registry.register_dataset(config)

        logger.info(f"Successfully registered dataset {dataset_name}")
        return config
