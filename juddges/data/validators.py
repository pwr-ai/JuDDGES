"""
Comprehensive validation framework for datasets before ingestion.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset
from loguru import logger

from juddges.data.dataset_registry import DatasetConfig


class ValidationLevel(Enum):
    """Validation severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A single validation issue."""

    level: ValidationLevel
    category: str
    message: str
    field: Optional[str] = None
    row_index: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    dataset_name: str
    total_rows: int
    validation_passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    resource_estimates: Dict[str, Any] = field(default_factory=dict)

    def add_issue(
        self,
        level: ValidationLevel,
        category: str,
        message: str,
        field: Optional[str] = None,
        row_index: Optional[int] = None,
        suggestion: Optional[str] = None,
    ):
        """Add a validation issue."""
        self.issues.append(
            ValidationIssue(
                level=level,
                category=category,
                message=message,
                field=field,
                row_index=row_index,
                suggestion=suggestion,
            )
        )

        # Update validation status
        if level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]:
            self.validation_passed = False

    def get_issues_by_level(self, level: ValidationLevel) -> List[ValidationIssue]:
        """Get all issues of a specific level."""
        return [issue for issue in self.issues if issue.level == level]

    def get_issues_by_category(self, category: str) -> List[ValidationIssue]:
        """Get all issues of a specific category."""
        return [issue for issue in self.issues if issue.category == category]

    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return any(issue.level == ValidationLevel.CRITICAL for issue in self.issues)

    def get_summary(self) -> Dict[str, int]:
        """Get summary of issues by level."""
        summary = {level.value: 0 for level in ValidationLevel}
        for issue in self.issues:
            summary[issue.level.value] += 1
        return summary


class DatasetValidator:
    """Comprehensive dataset validation before ingestion."""

    def __init__(self, sample_size: int = 1000):
        """Initialize validator with sample size for performance."""
        self.sample_size = sample_size

    def validate_dataset(self, dataset_name: str, config: DatasetConfig) -> ValidationResult:
        """Perform comprehensive dataset validation."""
        logger.info(f"Starting validation for dataset: {dataset_name}")
        start_time = time.time()

        result = ValidationResult(dataset_name=dataset_name, total_rows=0, validation_passed=True)

        try:
            # 1. Check dataset accessibility
            self._validate_accessibility(dataset_name, result)

            if result.has_critical_issues():
                return result

            # 2. Load dataset sample for analysis
            dataset_sample = self._load_dataset_sample(dataset_name, result)

            if dataset_sample is None:
                return result

            result.total_rows = self._get_total_rows(dataset_name)

            # 3. Validate configuration
            self._validate_configuration(config, dataset_sample, result)

            # 4. Validate data quality
            self._validate_data_quality(dataset_sample, config, result)

            # 5. Check required fields
            self._validate_required_fields(dataset_sample, config, result)

            # 6. Validate data types and formats
            self._validate_data_types(dataset_sample, config, result)

            # 7. Check for common issues
            self._validate_common_issues(dataset_sample, config, result)

            # 8. Estimate resource requirements
            self._estimate_resources(dataset_name, config, result)

            # 9. Performance analysis
            validation_time = time.time() - start_time
            result.performance_metrics = {
                "validation_time_seconds": validation_time,
                "rows_per_second": len(dataset_sample) / validation_time
                if validation_time > 0
                else 0,
                "sample_size": len(dataset_sample),
            }

            logger.info(f"Validation completed in {validation_time:.2f} seconds")

        except Exception as e:
            result.add_issue(
                ValidationLevel.CRITICAL,
                "validation_error",
                f"Validation failed with error: {str(e)}",
                suggestion="Check dataset accessibility and configuration",
            )

        return result

    def _validate_accessibility(self, dataset_name: str, result: ValidationResult):
        """Validate that the dataset is accessible."""
        try:
            # Try to load just the dataset info
            dataset_info = load_dataset(dataset_name, split="train[:1]")
            result.add_issue(
                ValidationLevel.INFO, "accessibility", f"Dataset {dataset_name} is accessible"
            )
        except Exception as e:
            result.add_issue(
                ValidationLevel.CRITICAL,
                "accessibility",
                f"Cannot access dataset {dataset_name}: {str(e)}",
                suggestion="Check dataset name and network connectivity",
            )

    def _load_dataset_sample(
        self, dataset_name: str, result: ValidationResult
    ) -> Optional[Dataset]:
        """Load a sample of the dataset for validation."""
        try:
            # Load a manageable sample
            dataset = load_dataset(dataset_name, split=f"train[:{self.sample_size}]")

            if len(dataset) == 0:
                result.add_issue(
                    ValidationLevel.CRITICAL,
                    "data_availability",
                    "Dataset appears to be empty",
                    suggestion="Check if dataset has been properly uploaded",
                )
                return None

            return dataset

        except Exception as e:
            result.add_issue(
                ValidationLevel.CRITICAL,
                "data_loading",
                f"Failed to load dataset sample: {str(e)}",
                suggestion="Check dataset format and accessibility",
            )
            return None

    def _get_total_rows(self, dataset_name: str) -> int:
        """Get total number of rows in dataset."""
        try:
            dataset = load_dataset(dataset_name, split="train")
            return len(dataset)
        except Exception:
            return 0

    def _validate_configuration(
        self, config: DatasetConfig, dataset: Dataset, result: ValidationResult
    ):
        """Validate the dataset configuration."""
        dataset_columns = set(dataset.column_names)

        # Check if mapped columns exist
        missing_columns = []
        for source_col in config.column_mapping.keys():
            if source_col not in dataset_columns:
                missing_columns.append(source_col)

        if missing_columns:
            result.add_issue(
                ValidationLevel.ERROR,
                "configuration",
                f"Mapped columns not found in dataset: {missing_columns}",
                suggestion="Update column mapping or check dataset structure",
            )

        # Check for unmapped important columns
        important_patterns = ["id", "text", "content", "date", "title"]
        unmapped_important = []

        for col in dataset_columns:
            if any(pattern in col.lower() for pattern in important_patterns):
                if col not in config.column_mapping:
                    unmapped_important.append(col)

        if unmapped_important:
            result.add_issue(
                ValidationLevel.WARNING,
                "configuration",
                f"Potentially important columns not mapped: {unmapped_important}",
                suggestion="Consider adding these columns to the mapping",
            )

    def _validate_required_fields(
        self, dataset: Dataset, config: DatasetConfig, result: ValidationResult
    ):
        """Validate that required fields can be satisfied."""
        mapped_targets = set(config.column_mapping.values())

        missing_required = []
        for required_field in config.required_fields:
            if required_field not in mapped_targets:
                missing_required.append(required_field)

        if missing_required:
            result.add_issue(
                ValidationLevel.ERROR,
                "required_fields",
                f"Required fields not mapped: {missing_required}",
                suggestion="Add mappings for required fields or update requirements",
            )

    def _validate_data_quality(
        self, dataset: Dataset, config: DatasetConfig, result: ValidationResult
    ):
        """Validate overall data quality."""

        # Check for completely empty rows
        empty_rows = 0
        null_heavy_rows = 0

        for i, row in enumerate(dataset):
            non_null_values = sum(1 for v in row.values() if v is not None and v != "")

            if non_null_values == 0:
                empty_rows += 1
            elif non_null_values < len(row) * 0.3:  # Less than 30% filled
                null_heavy_rows += 1

        if empty_rows > 0:
            result.add_issue(
                ValidationLevel.WARNING,
                "data_quality",
                f"Found {empty_rows} completely empty rows",
                suggestion="Consider filtering out empty rows",
            )

        if null_heavy_rows > len(dataset) * 0.1:  # More than 10% sparse rows
            result.add_issue(
                ValidationLevel.WARNING,
                "data_quality",
                f"Found {null_heavy_rows} sparse rows (>70% null values)",
                suggestion="Review data quality and consider additional cleaning",
            )

    def _validate_data_types(
        self, dataset: Dataset, config: DatasetConfig, result: ValidationResult
    ):
        """Validate data types and formats."""

        # Check text fields
        for text_field in config.text_fields:
            source_fields = [k for k, v in config.column_mapping.items() if v == text_field]
            for source_field in source_fields:
                if source_field in dataset.column_names:
                    self._validate_text_field(dataset, source_field, result)

        # Check date fields
        for date_field in config.date_fields:
            source_fields = [k for k, v in config.column_mapping.items() if v == date_field]
            for source_field in source_fields:
                if source_field in dataset.column_names:
                    self._validate_date_field(dataset, source_field, result)

        # Check array fields
        for array_field in config.array_fields:
            source_fields = [k for k, v in config.column_mapping.items() if v == array_field]
            for source_field in source_fields:
                if source_field in dataset.column_names:
                    self._validate_array_field(dataset, source_field, result)

    def _validate_text_field(self, dataset: Dataset, field_name: str, result: ValidationResult):
        """Validate a text field."""
        values = dataset[field_name]

        # Check for very short or very long texts
        short_texts = sum(1 for v in values if isinstance(v, str) and len(v) < 10)
        long_texts = sum(1 for v in values if isinstance(v, str) and len(v) > 100000)

        if short_texts > len(values) * 0.3:
            result.add_issue(
                ValidationLevel.WARNING,
                "text_validation",
                f"Field {field_name}: {short_texts} very short texts (<10 chars)",
                field=field_name,
                suggestion="Consider if this field contains substantial text content",
            )

        if long_texts > 0:
            result.add_issue(
                ValidationLevel.INFO,
                "text_validation",
                f"Field {field_name}: {long_texts} very long texts (>100k chars)",
                field=field_name,
                suggestion="Consider chunking strategy for large texts",
            )

    def _validate_date_field(self, dataset: Dataset, field_name: str, result: ValidationResult):
        """Validate a date field."""
        values = dataset[field_name]
        invalid_dates = 0

        for i, value in enumerate(values):
            if value is not None and not self._is_valid_date_format(value):
                invalid_dates += 1
                if invalid_dates <= 5:  # Only report first 5 examples
                    result.add_issue(
                        ValidationLevel.WARNING,
                        "date_validation",
                        f"Invalid date format in {field_name}: '{value}'",
                        field=field_name,
                        row_index=i,
                        suggestion="Ensure dates are in recognizable format",
                    )

        if invalid_dates > len(values) * 0.1:
            result.add_issue(
                ValidationLevel.ERROR,
                "date_validation",
                f"Field {field_name}: {invalid_dates} invalid date formats",
                field=field_name,
                suggestion="Review date format consistency",
            )

    def _validate_array_field(self, dataset: Dataset, field_name: str, result: ValidationResult):
        """Validate an array field."""
        values = dataset[field_name]
        non_list_values = 0

        for value in values:
            if value is not None and not isinstance(value, list):
                # Check if it's a string that might be parseable as array
                if isinstance(value, str):
                    if not any(delimiter in value for delimiter in [",", ";", "|"]):
                        non_list_values += 1
                else:
                    non_list_values += 1

        if non_list_values > 0:
            result.add_issue(
                ValidationLevel.INFO,
                "array_validation",
                f"Field {field_name}: {non_list_values} non-array values",
                field=field_name,
                suggestion="Converter will attempt to parse as arrays",
            )

    def _is_valid_date_format(self, value: Any) -> bool:
        """Check if a value looks like a valid date."""
        if not isinstance(value, str):
            return False

        # Simple regex patterns for common date formats
        import re

        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",
            r"\d{2}/\d{2}/\d{4}",
            r"\d{4}/\d{2}/\d{2}",
            r"\d{2}-\d{2}-\d{4}",
        ]

        return any(re.search(pattern, value) for pattern in date_patterns)

    def _validate_common_issues(
        self, dataset: Dataset, config: DatasetConfig, result: ValidationResult
    ):
        """Check for common dataset issues."""

        # Check for duplicate document IDs
        id_fields = [k for k, v in config.column_mapping.items() if v == "document_id"]
        if id_fields:
            id_field = id_fields[0]
            if id_field in dataset.column_names:
                ids = dataset[id_field]
                unique_ids = set(ids)

                if len(unique_ids) < len(ids):
                    duplicates = len(ids) - len(unique_ids)
                    result.add_issue(
                        ValidationLevel.WARNING,
                        "data_integrity",
                        f"Found {duplicates} duplicate document IDs",
                        field=id_field,
                        suggestion="Consider deduplication or ID generation strategy",
                    )

        # Check for inconsistent text lengths in main content
        text_fields = [k for k, v in config.column_mapping.items() if v == "full_text"]
        if text_fields:
            text_field = text_fields[0]
            if text_field in dataset.column_names:
                texts = [t for t in dataset[text_field] if isinstance(t, str)]
                if texts:
                    avg_length = sum(len(t) for t in texts) / len(texts)
                    very_short = sum(1 for t in texts if len(t) < avg_length * 0.1)

                    if very_short > len(texts) * 0.05:  # More than 5% very short
                        result.add_issue(
                            ValidationLevel.WARNING,
                            "content_consistency",
                            f"Found {very_short} texts much shorter than average",
                            field=text_field,
                            suggestion="Review content extraction quality",
                        )

    def _estimate_resources(
        self, dataset_name: str, config: DatasetConfig, result: ValidationResult
    ):
        """Estimate resource requirements for processing."""
        try:
            # Get dataset size estimates
            total_rows = result.total_rows

            # Estimate processing time (rough calculation)
            estimated_processing_time = total_rows * 0.01  # 10ms per row baseline
            if config.text_fields:
                estimated_processing_time *= 2  # Text processing overhead

            # Estimate memory requirements
            avg_row_size = 1024  # 1KB per row estimate
            estimated_memory_mb = (total_rows * avg_row_size) / (1024 * 1024)

            # Estimate storage requirements
            # Embeddings add significant storage
            embedding_storage_mb = total_rows * 0.003  # ~3KB per document for embeddings

            result.resource_estimates = {
                "estimated_processing_time_minutes": estimated_processing_time / 60,
                "estimated_memory_mb": estimated_memory_mb,
                "estimated_storage_mb": embedding_storage_mb,
                "recommended_batch_size": min(32, max(8, 1000000 // total_rows)),
            }

            # Add warnings for large datasets
            if estimated_processing_time > 3600:  # More than 1 hour
                result.add_issue(
                    ValidationLevel.WARNING,
                    "resource_requirements",
                    f"Estimated processing time: {estimated_processing_time / 3600:.1f} hours",
                    suggestion="Consider processing in smaller batches or using more powerful hardware",
                )

            if estimated_memory_mb > 8192:  # More than 8GB
                result.add_issue(
                    ValidationLevel.WARNING,
                    "resource_requirements",
                    f"Estimated memory requirement: {estimated_memory_mb:.1f} MB",
                    suggestion="Consider streaming processing or increasing available memory",
                )

        except Exception as e:
            result.add_issue(
                ValidationLevel.WARNING,
                "resource_estimation",
                f"Could not estimate resource requirements: {str(e)}",
            )
