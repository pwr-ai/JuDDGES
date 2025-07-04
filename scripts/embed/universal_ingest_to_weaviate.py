"""
Universal ingestion script using the new enhanced system.
This replaces the old ingest_to_weaviate.py with more flexible capabilities.
"""

import sys

import hydra
from loguru import logger
from omegaconf import DictConfig

from juddges.data.config import IngestConfig
from juddges.data.dataset_registry import get_registry
from juddges.data.universal_processor import UniversalDatasetProcessor
from juddges.data.validators import DatasetValidator
from juddges.settings import CONFIG_PATH
from juddges.utils.config import resolve_config


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="embedding.yaml")
def main(cfg: DictConfig) -> None:
    """Universal ingestion with enhanced capabilities."""
    logger.info("Starting universal dataset ingestion")

    # Resolve configuration
    config = resolve_config(cfg, resolve=True)

    # Extract parameters
    dataset_name = config.get("dataset_name")
    if not dataset_name:
        logger.error("dataset_name is required in configuration")
        sys.exit(1)

    # Initialize components
    registry = get_registry()
    processor = UniversalDatasetProcessor(registry=registry)
    validator = DatasetValidator()

    # Command line overrides
    max_documents = config.get("max_documents")
    if "max_documents" in cfg:
        max_documents = int(cfg.max_documents)
        logger.info(f"Overriding max_documents: {max_documents}")

    upsert = config.get("upsert", True)
    if "upsert" in cfg:
        upsert = str(cfg.upsert).lower() == "true"
        logger.info(f"Overriding upsert: {upsert}")

    batch_size = config.get("ingest_batch_size", 32)
    if "ingest_batch_size" in cfg:
        batch_size = int(cfg.ingest_batch_size)
        logger.info(f"Overriding batch_size: {batch_size}")

    # Check if dataset is registered
    dataset_config = registry.get_config(dataset_name)

    # Command line override for chunks_path
    if "chunks_path" in cfg and cfg.chunks_path:
        chunks_path = str(cfg.chunks_path)
        logger.info(f"Overriding chunks_path: {chunks_path}")
        if dataset_config:
            dataset_config.chunks_path = chunks_path

    if not dataset_config:
        logger.info(f"Dataset {dataset_name} not registered. Creating automatic configuration...")
        try:
            dataset_config = processor.register_new_dataset(dataset_name)
            logger.info(f"Successfully auto-registered {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to auto-register dataset: {e}")
            logger.info("You can manually register the dataset using:")
            logger.info(f"python scripts/dataset_manager.py add '{dataset_name}' --auto")
            sys.exit(1)

    # Validation step (can be skipped with --skip-validation)
    skip_validation = cfg.get("skip_validation", False)
    if not skip_validation:
        logger.info("Running dataset validation...")
        try:
            validation_result = validator.validate_dataset(dataset_name, dataset_config)

            if validation_result.has_critical_issues():
                logger.error("Critical validation issues found:")
                for issue in validation_result.get_issues_by_level("critical"):
                    logger.error(f"  {issue.message}")
                logger.error("Cannot proceed with ingestion")
                sys.exit(1)

            error_count = len(validation_result.get_issues_by_level("error"))
            warning_count = len(validation_result.get_issues_by_level("warning"))

            if error_count > 0:
                logger.warning(
                    f"Found {error_count} validation errors and {warning_count} warnings"
                )
                for issue in validation_result.get_issues_by_level("error")[:5]:  # Show first 5
                    logger.warning(f"  {issue.message}")

                if not cfg.get("force", False):
                    logger.error("Use --force to proceed despite validation errors")
                    sys.exit(1)

            logger.info(f"✓ Validation passed ({warning_count} warnings)")

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            if not cfg.get("force", False):
                sys.exit(1)
    else:
        logger.info("Skipping validation (--skip-validation)")

    # Preview mode
    if cfg.get("preview_only", False):
        logger.info("Running in preview mode...")
        try:
            preview = processor.preview_dataset(dataset_name)
            logger.info(f"Dataset: {preview.dataset_name}")
            logger.info(f"Total rows: {preview.total_rows:,}")
            logger.info(f"Columns: {len(preview.columns)}")
            logger.info(
                f"Estimated processing time: {preview.estimated_processing_time / 60:.1f} minutes"
            )
            logger.info("Preview completed successfully")
            return
        except Exception as e:
            logger.error(f"Preview failed: {e}")
            sys.exit(1)

    # Prepare ingestion configuration
    ingest_config = IngestConfig(max_documents=max_documents, batch_size=batch_size, upsert=upsert)

    logger.info(f"Starting ingestion with config: {ingest_config}")

    # Process the dataset
    try:
        result = processor.process_dataset(
            dataset_name=dataset_name,
            config=dataset_config,
            ingest_config=ingest_config,
            create_embeddings=True,
            preview_only=False,
        )

        # Display results
        if result.success:
            logger.info("🎉 Ingestion completed successfully!")
            logger.info(f"Dataset: {result.dataset_name}")
            logger.info(f"Processed: {result.processed_rows:,} / {result.total_rows:,} rows")
            logger.info(f"Documents ingested: {result.ingested_documents:,}")
            logger.info(f"Chunks ingested: {result.ingested_chunks:,}")
            logger.info(f"Processing time: {result.processing_time_seconds:.2f} seconds")

            if result.warnings:
                logger.warning("Warnings:")
                for warning in result.warnings:
                    logger.warning(f"  {warning}")
        else:
            logger.error("❌ Ingestion failed!")
            for error in result.errors:
                logger.error(f"  {error}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Ingestion failed with exception: {e}")
        logger.error("For help with dataset configuration, use:")
        logger.error(f"python scripts/dataset_manager.py preview '{dataset_name}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
