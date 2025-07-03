"""
Robust data type conversion with error recovery and validation.
"""

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from juddges.data.dataset_registry import DatasetConfig


@dataclass
class ConversionResult:
    """Result of data conversion with success status and errors."""

    success: bool
    converted_data: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


class RobustDataConverter:
    """Handle data type conversion with comprehensive error recovery."""

    # Common date formats to try
    DATE_FORMATS = [
        "%Y-%m-%d",  # 2023-12-25
        "%Y/%m/%d",  # 2023/12/25
        "%d/%m/%Y",  # 25/12/2023
        "%m/%d/%Y",  # 12/25/2023
        "%Y-%m-%d %H:%M:%S",  # 2023-12-25 14:30:00
        "%Y-%m-%dT%H:%M:%S",  # 2023-12-25T14:30:00
        "%Y-%m-%dT%H:%M:%SZ",  # 2023-12-25T14:30:00Z
        "%d %B %Y",  # 25 December 2023
        "%B %d, %Y",  # December 25, 2023
        "%d-%m-%Y",  # 25-12-2023
        "%m-%d-%Y",  # 12-25-2023
    ]

    def __init__(self):
        """Initialize the converter."""
        self.conversion_stats = {
            "total_rows": 0,
            "successful_rows": 0,
            "failed_rows": 0,
            "field_errors": {},
            "field_warnings": {},
        }

    def convert_dataset_batch(
        self, batch: Dict[str, List[Any]], config: DatasetConfig
    ) -> ConversionResult:
        """Convert a batch of data according to dataset configuration."""
        if "document_id" in batch:
            # Ensure document_id is a list for consistency
            logger.debug(f"Converting batch of {len(batch.get('document_id', []))} rows")
        elif "judgment_id" in batch:
            logger.debug(
                f"Converting batch of {len(batch.get('judgment_id', []))} rows using judgment_id"
            )
        elif "id" in batch:
            logger.debug(f"Converting batch of {len(batch.get('id', []))} rows using id")
        else:
            raise Exception("Batch must contain 'document_id', 'judgment_id', or 'id' field")

        converted_batch = {}
        all_errors = []
        all_warnings = []

        # Initialize output arrays
        batch_size = len(next(iter(batch.values())))
        for target_field in config.column_mapping.values():
            converted_batch[target_field] = [None] * batch_size

        # Apply default values
        for target_field, default_value in config.default_values.items():
            if target_field not in converted_batch:
                converted_batch[target_field] = [default_value] * batch_size
            else:
                for i in range(batch_size):
                    if converted_batch[target_field][i] is None:
                        converted_batch[target_field][i] = default_value

        # Convert each field
        for source_field, target_field in config.column_mapping.items():
            if source_field not in batch:
                warning = f"Source field '{source_field}' not found in batch"
                all_warnings.append(warning)
                continue

            source_values = batch[source_field]
            converted_values, field_errors, field_warnings = self._convert_field_batch(
                source_values, source_field, target_field, config
            )

            converted_batch[target_field] = converted_values
            all_errors.extend(field_errors)
            all_warnings.extend(field_warnings)

        # Update statistics
        self.conversion_stats["total_rows"] += batch_size
        if not all_errors:
            self.conversion_stats["successful_rows"] += batch_size
        else:
            self.conversion_stats["failed_rows"] += batch_size

        return ConversionResult(
            success=len(all_errors) == 0,
            converted_data=converted_batch,
            errors=all_errors,
            warnings=all_warnings,
        )

    def convert_row(self, row: Dict[str, Any], config: DatasetConfig) -> ConversionResult:
        """Convert a single row according to dataset configuration."""
        converted = {}
        errors = []
        warnings = []

        # Apply column mapping
        for source_field, target_field in config.column_mapping.items():
            try:
                value = row.get(source_field)
                converted_value, field_errors, field_warnings = self._convert_field_value(
                    value, source_field, target_field, config
                )

                if converted_value is not None:
                    converted[target_field] = converted_value

                errors.extend(field_errors)
                warnings.extend(field_warnings)

            except Exception as e:
                error_msg = f"Unexpected error converting {source_field}->{target_field}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)

        # Apply default values for missing fields
        for target_field, default_value in config.default_values.items():
            if target_field not in converted or converted[target_field] is None:
                converted[target_field] = default_value

        return ConversionResult(
            success=len(errors) == 0, converted_data=converted, errors=errors, warnings=warnings
        )

    def _convert_field_batch(
        self, values: List[Any], source_field: str, target_field: str, config: DatasetConfig
    ) -> Tuple[List[Any], List[str], List[str]]:
        """Convert a batch of values for a single field."""
        converted_values = []
        errors = []
        warnings = []

        for i, value in enumerate(values):
            try:
                converted_value, field_errors, field_warnings = self._convert_field_value(
                    value, source_field, target_field, config
                )
                converted_values.append(converted_value)
                errors.extend([f"Row {i}: {err}" for err in field_errors])
                warnings.extend([f"Row {i}: {warn}" for warn in field_warnings])
            except Exception as e:
                error_msg = f"Row {i}: Unexpected error converting {source_field}: {e}"
                errors.append(error_msg)
                converted_values.append(None)

        return converted_values, errors, warnings

    def _convert_field_value(
        self, value: Any, source_field: str, target_field: str, config: DatasetConfig
    ) -> Tuple[Any, List[str], List[str]]:
        """Convert individual field value with type-specific logic."""
        errors = []
        warnings = []

        # Handle None/null values
        if value is None or (isinstance(value, float) and np.isnan(value)):
            default_value = config.default_values.get(target_field)
            return default_value, errors, warnings

        # Handle empty strings
        if isinstance(value, str) and value.strip() == "":
            if target_field in config.required_fields:
                errors.append(f"Required field {target_field} is empty")
                return None, errors, warnings
            else:
                return None, errors, warnings

        # Apply field-specific conversions
        try:
            if target_field in config.date_fields:
                converted_value = self._convert_date(value)
                if converted_value is None:
                    errors.append(f"Could not parse date from '{value}'")
            elif target_field in config.array_fields:
                converted_value = self._convert_array(value)
            elif target_field in config.json_fields:
                converted_value = self._convert_json(value)
            elif target_field in config.text_fields:
                converted_value = self._convert_text(value)
            else:
                converted_value = self._convert_generic(value, target_field)

            return converted_value, errors, warnings

        except Exception as e:
            errors.append(f"Conversion failed for {source_field}->{target_field}: {e}")
            return None, errors, warnings

    def _convert_date(self, value: Any) -> Optional[str]:
        """Convert various date formats to ISO format string."""
        if isinstance(value, (datetime, date)):
            return value.isoformat()

        if not isinstance(value, str):
            value = str(value)

        # Clean the string
        value = value.strip()

        # Try each date format
        for date_format in self.DATE_FORMATS:
            try:
                parsed_date = datetime.strptime(value, date_format)
                return parsed_date.isoformat()
            except ValueError:
                continue

        # Try parsing with dateutil if available
        try:
            from dateutil import parser

            parsed_date = parser.parse(value)
            return parsed_date.isoformat()
        except (ImportError, ValueError):
            pass

        # Extract date using regex patterns
        date_match = self._extract_date_with_regex(value)
        if date_match:
            return date_match

        return None

    def _extract_date_with_regex(self, text: str) -> Optional[str]:
        """Extract date using regex patterns."""
        patterns = [
            r"(\d{4})-(\d{1,2})-(\d{1,2})",  # YYYY-MM-DD
            r"(\d{1,2})/(\d{1,2})/(\d{4})",  # MM/DD/YYYY or DD/MM/YYYY
            r"(\d{4})/(\d{1,2})/(\d{1,2})",  # YYYY/MM/DD
            r"(\d{1,2})-(\d{1,2})-(\d{4})",  # MM-DD-YYYY or DD-MM-YYYY
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 3:
                        # Try different interpretations
                        for date_format in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
                            try:
                                if pattern.startswith(r"(\d{4})"):
                                    # Year first
                                    date_str = f"{groups[0]}-{groups[1]:0>2}-{groups[2]:0>2}"
                                    parsed = datetime.strptime(date_str, "%Y-%m-%d")
                                else:
                                    # Try both MM/DD/YYYY and DD/MM/YYYY
                                    date_str = f"{groups[0]}/{groups[1]}/{groups[2]}"
                                    parsed = datetime.strptime(date_str, date_format)
                                return parsed.isoformat()
                            except ValueError:
                                continue
                except Exception:
                    continue

        return None

    def _convert_array(self, value: Any) -> Optional[List[Any]]:
        """Convert various formats to array/list."""
        if isinstance(value, list):
            return value

        if isinstance(value, str):
            # Try parsing as JSON array
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

            # Split by common delimiters
            for delimiter in [";", ",", "|", "\n"]:
                if delimiter in value:
                    items = [item.strip() for item in value.split(delimiter)]
                    return [item for item in items if item]  # Remove empty items

            # Single item becomes single-element array
            return [value.strip()] if value.strip() else []

        # Convert single value to array
        return [value] if value is not None else []

    def _convert_json(self, value: Any) -> Optional[str]:
        """Convert value to JSON string."""
        if isinstance(value, str):
            # Try to parse and re-serialize to ensure valid JSON
            try:
                parsed = json.loads(value)
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                # If it's not valid JSON, treat as regular string
                return json.dumps(value, ensure_ascii=False)

        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)

        # Convert other types to JSON
        return json.dumps(value, ensure_ascii=False) if value is not None else None

    def _convert_text(self, value: Any) -> Optional[str]:
        """Convert value to clean text string."""
        if value is None:
            return None

        if isinstance(value, str):
            # Clean whitespace and normalize
            cleaned = re.sub(r"\s+", " ", value.strip())
            return cleaned if cleaned else None

        # Convert other types to string
        return str(value)

    def _convert_generic(self, value: Any, target_field: str) -> Any:
        """Generic conversion for unspecified field types."""
        # Keep native type for simple types
        if isinstance(value, (str, int, float, bool)):
            return value

        # Convert complex types to JSON strings
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)

        # Convert everything else to string
        return str(value) if value is not None else None

    def validate_converted_data(
        self, converted_data: Dict[str, Any], config: DatasetConfig
    ) -> Tuple[bool, List[str]]:
        """Validate converted data against requirements."""
        errors = []

        # Check required fields
        for required_field in config.required_fields:
            if required_field not in converted_data:
                errors.append(f"Required field {required_field} is missing")
            elif converted_data[required_field] is None:
                errors.append(f"Required field {required_field} is null")

        # Validate data types for known fields
        if "document_id" in converted_data:
            doc_id = converted_data["document_id"]
            if not isinstance(doc_id, str) or not doc_id.strip():
                errors.append("document_id must be a non-empty string")

        # Validate text fields
        for text_field in config.text_fields:
            if text_field in converted_data:
                value = converted_data[text_field]
                if value is not None and not isinstance(value, str):
                    errors.append(f"Text field {text_field} must be a string")

        # Validate array fields
        for array_field in config.array_fields:
            if array_field in converted_data:
                value = converted_data[array_field]
                if value is not None and not isinstance(value, list):
                    errors.append(f"Array field {array_field} must be a list")

        return len(errors) == 0, errors

    def get_conversion_statistics(self) -> Dict[str, Any]:
        """Get conversion statistics."""
        total = self.conversion_stats["total_rows"]
        success_rate = (self.conversion_stats["successful_rows"] / total * 100) if total > 0 else 0

        return {
            "total_rows_processed": total,
            "successful_rows": self.conversion_stats["successful_rows"],
            "failed_rows": self.conversion_stats["failed_rows"],
            "success_rate_percent": round(success_rate, 2),
            "field_error_counts": dict(self.conversion_stats["field_errors"]),
            "field_warning_counts": dict(self.conversion_stats["field_warnings"]),
        }

    def reset_statistics(self) -> None:
        """Reset conversion statistics."""
        self.conversion_stats = {
            "total_rows": 0,
            "successful_rows": 0,
            "failed_rows": 0,
            "field_errors": {},
            "field_warnings": {},
        }
