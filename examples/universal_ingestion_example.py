#!/usr/bin/env python3
"""
Comprehensive example of using the Universal Dataset Ingestion System.

This example demonstrates:
1. Auto-registering a new dataset
2. Previewing dataset structure
3. Validating dataset quality
4. Ingesting to Weaviate with error handling

Usage:
    python examples/universal_ingestion_example.py
"""

from loguru import logger
from rich.console import Console

from juddges.data.config import IngestConfig
from juddges.data.dataset_registry import get_registry
from juddges.data.universal_processor import UniversalDatasetProcessor
from juddges.data.validators import DatasetValidator, ValidationLevel

console = Console()


def main():
    """Demonstrate the universal ingestion system."""

    # Example dataset - replace with your own HuggingFace dataset
    dataset_name = "juddges/pl-court-raw"  # Or try: "your-org/your-legal-dataset"

    logger.info("🚀 Universal Dataset Ingestion Example")
    logger.info(f"Target dataset: {dataset_name}")

    # Initialize components
    registry = get_registry()
    processor = UniversalDatasetProcessor(registry=registry)
    validator = DatasetValidator()

    try:
        # Step 1: Preview the dataset
        logger.info("\n📋 Step 1: Previewing dataset structure...")
        preview = processor.preview_dataset(dataset_name, sample_size=5)

        console.print(f"Dataset: {preview.dataset_name}")
        console.print(f"Total rows: {preview.total_rows:,}")
        console.print(f"Columns: {preview.columns}")
        console.print(f"Estimated processing time: {preview.estimated_processing_time / 60:.1f} minutes")

        console.print("\nSample data:")
        for i, row in enumerate(preview.sample_rows[:2]):
            console.print(f"  Row {i + 1}: {list(row.keys())}")

        console.print(f"\nSuggested mapping:")
        for source, target in preview.suggested_mapping.items():
            console.print(f"  {source} -> {target}")

        # Step 2: Register the dataset (auto-configuration)
        logger.info("\n⚙️ Step 2: Auto-registering dataset...")

        existing_config = registry.get_config(dataset_name)
        if existing_config:
            logger.info(f"Dataset already registered: {dataset_name}")
            config = existing_config
        else:
            config = processor.register_new_dataset(dataset_name)
            logger.info(f"Successfully registered: {dataset_name}")

        console.print(f"Configuration:")
        console.print(f"  Document type: {config.document_type}")
        console.print(f"  Required fields: {config.required_fields}")
        console.print(f"  Text fields: {config.text_fields}")
        console.print(f"  Default values: {config.default_values}")

        # Step 3: Validate the dataset
        logger.info("\n🔍 Step 3: Validating dataset quality...")
        validation_result = validator.validate_dataset(dataset_name, config)

        summary = validation_result.get_summary()
        console.print(
            f"Validation result: {'✅ PASSED' if validation_result.validation_passed else '❌ FAILED'}"
        )
        console.print(
            f"Issues: {summary['critical']} critical, {summary['error']} errors, {summary['warning']} warnings"
        )

        if validation_result.resource_estimates:
            estimates = validation_result.resource_estimates
            console.print(f"\nResource estimates:")
            console.print(
                f"  Processing time: {estimates.get('estimated_processing_time_minutes', 0):.1f} minutes"
            )
            console.print(f"  Memory needed: {estimates.get('estimated_memory_mb', 0):.1f} MB")
            console.print(f"  Recommended batch size: {estimates.get('recommended_batch_size', 32)}")

        # Show any critical issues
        critical_issues = validation_result.get_issues_by_level(ValidationLevel.CRITICAL)
        if critical_issues:
            logger.error("Critical issues found:")
            for issue in critical_issues:
                logger.error(f"  {issue.message}")
            logger.error("Cannot proceed with ingestion")
            return

        # Step 4: Configure ingestion
        logger.info("\n⚡ Step 4: Configuring ingestion...")

        ingest_config = IngestConfig(
            max_documents=100,  # Limit for example
            batch_size=16,  # Smaller batch for example
            upsert=True,
        )

        console.print(f"Ingestion config:")
        console.print(f"  Max documents: {ingest_config.max_documents}")
        console.print(f"  Batch size: {ingest_config.batch_size}")
        console.print(f"  Upsert enabled: {ingest_config.upsert}")

        # Step 5: Dry run (preview only)
        logger.info("\n🏃 Step 5: Running dry run...")
        dry_run_result = processor.process_dataset(
            dataset_name=dataset_name, config=config, ingest_config=ingest_config, preview_only=True
        )

        if dry_run_result.success:
            logger.info("✅ Dry run successful!")
            console.print(f"Would process: {dry_run_result.total_rows} rows")
        else:
            logger.error("❌ Dry run failed!")
            for error in dry_run_result.errors:
                logger.error(f"  {error}")
            return

        # Step 6: Actual ingestion (commented out for safety)
        logger.info("\n🚀 Step 6: Ready for ingestion!")
        logger.info(
            "To run actual ingestion, uncomment the code below and ensure Weaviate is running"
        )

        # Uncomment below to run actual ingestion:
        """
        logger.info("Starting actual ingestion...")
        result = processor.process_dataset(
            dataset_name=dataset_name,
            config=config,
            ingest_config=ingest_config,
            create_embeddings=True,
            preview_only=False
        )
        
        if result.success:
            logger.info("🎉 Ingestion completed successfully!")
            console.print(f"Documents ingested: {result.ingested_documents}")
            console.print(f"Chunks ingested: {result.ingested_chunks}")
            console.print(f"Processing time: {result.processing_time_seconds:.2f} seconds")
        else:
            logger.error("❌ Ingestion failed!")
            for error in result.errors:
                logger.error(f"  {error}")
        """

        # Step 7: Show CLI commands for manual operation
        logger.info("\n🛠️ CLI Commands for manual operation:")
        console.print(f"Preview dataset:")
        console.print(f"  python scripts/dataset_manager.py preview '{dataset_name}'")
        console.print(f"\nValidate dataset:")
        console.print(f"  python scripts/dataset_manager.py validate '{dataset_name}'")
        console.print(f"\nIngest dataset:")
        console.print(
            f"  python scripts/dataset_manager.py ingest '{dataset_name}' --max-docs 100 --dry-run"
        )
        console.print(f"  python scripts/dataset_manager.py ingest '{dataset_name}' --max-docs 1000")

        logger.info("\n✨ Example completed successfully!")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        logger.error("Make sure you have a valid HuggingFace dataset name")
        logger.error("You can try with 'juddges/pl-court-raw' or any other legal dataset")


if __name__ == "__main__":
    main()
