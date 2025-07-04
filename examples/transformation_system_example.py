#!/usr/bin/env python3
"""
Example showing a pluggable transformation system for data processing.
This demonstrates how to handle court_id -> court_name mappings and other transformations.
"""

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class DataTransformer(ABC):
    """Base class for data transformations."""

    @abstractmethod
    def transform(self, value: Any, context: Dict[str, Any] = None) -> Any:
        """Transform a single value."""
        pass

    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Validate if the value can be transformed."""
        pass


class LookupTableTransformer(DataTransformer):
    """Transform values using a lookup table (e.g., court_id -> court_name)."""

    def __init__(self, lookup_table: Dict[str, str], default_value: Optional[str] = None):
        self.lookup_table = lookup_table
        self.default_value = default_value

    @classmethod
    def from_file(
        cls, file_path: Path, key_field: str, value_field: str, default_value: Optional[str] = None
    ):
        """Create transformer from JSON file."""
        with open(file_path) as f:
            data = json.load(f)

        lookup_table = {}
        for item in data:
            if key_field in item and value_field in item:
                lookup_table[str(item[key_field])] = str(item[value_field])

        return cls(lookup_table, default_value)

    def transform(self, value: Any, context: Dict[str, Any] = None) -> Any:
        """Transform using lookup table."""
        key = str(value) if value is not None else ""
        return self.lookup_table.get(key, self.default_value or value)

    def validate(self, value: Any) -> bool:
        """Check if value exists in lookup table or has default."""
        return str(value) in self.lookup_table or self.default_value is not None


class DateParserTransformer(DataTransformer):
    """Parse dates in various formats."""

    def __init__(self, date_formats: List[str] = None, output_format: str = "%Y-%m-%d"):
        self.date_formats = date_formats or [
            "%Y-%m-%d",
            "%d.%m.%Y",
            "%d/%m/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%d.%m.%Y %H:%M:%S",
        ]
        self.output_format = output_format

    def transform(self, value: Any, context: Dict[str, Any] = None) -> Any:
        """Parse date string to standardized format."""
        if value is None:
            return None

        # If already datetime object
        if isinstance(value, datetime):
            return value.strftime(self.output_format)

        # Try parsing with different formats
        date_str = str(value).strip()
        if not date_str:
            return None

        for fmt in self.date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return parsed_date.strftime(self.output_format)
            except ValueError:
                continue

        # If no format worked, return original value
        return value

    def validate(self, value: Any) -> bool:
        """Check if value can be parsed as date."""
        if value is None or isinstance(value, datetime):
            return True

        date_str = str(value).strip()
        if not date_str:
            return True

        for fmt in self.date_formats:
            try:
                datetime.strptime(date_str, fmt)
                return True
            except ValueError:
                continue
        return False


class JsonSerializerTransformer(DataTransformer):
    """Serialize complex objects to JSON strings."""

    def transform(self, value: Any, context: Dict[str, Any] = None) -> Any:
        """Serialize value to JSON string."""
        if value is None:
            return None

        # If already a string, assume it's JSON
        if isinstance(value, str):
            try:
                # Validate it's valid JSON
                json.loads(value)
                return value
            except json.JSONDecodeError:
                # Not valid JSON, wrap in array
                return json.dumps([value])

        # If it's a list or dict, serialize
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False, default=str)

        # For other types, convert to string and wrap in array
        return json.dumps([str(value)], ensure_ascii=False)

    def validate(self, value: Any) -> bool:
        """Always valid - can serialize anything."""
        return True


class ValueMappingTransformer(DataTransformer):
    """Map specific values to other values."""

    def __init__(self, value_mappings: Dict[str, str], case_sensitive: bool = False):
        self.value_mappings = value_mappings
        self.case_sensitive = case_sensitive

        if not case_sensitive:
            # Create lowercase mapping for case-insensitive matching
            self.lowercase_mappings = {k.lower(): v for k, v in value_mappings.items()}

    def transform(self, value: Any, context: Dict[str, Any] = None) -> Any:
        """Transform using value mapping."""
        if value is None:
            return None

        value_str = str(value)

        if self.case_sensitive:
            return self.value_mappings.get(value_str, value)
        else:
            return self.lowercase_mappings.get(value_str.lower(), value)

    def validate(self, value: Any) -> bool:
        """Always valid - returns original if no mapping found."""
        return True


class RegexTransformer(DataTransformer):
    """Transform values using regex patterns."""

    def __init__(self, patterns: List[tuple], default_value: Optional[str] = None):
        """
        patterns: List of (regex_pattern, replacement) tuples
        """
        self.patterns = [(re.compile(pattern), replacement) for pattern, replacement in patterns]
        self.default_value = default_value

    def transform(self, value: Any, context: Dict[str, Any] = None) -> Any:
        """Transform using regex patterns."""
        if value is None:
            return self.default_value

        value_str = str(value)

        for pattern, replacement in self.patterns:
            if pattern.search(value_str):
                return pattern.sub(replacement, value_str)

        return self.default_value or value

    def validate(self, value: Any) -> bool:
        """Check if any pattern matches."""
        if value is None:
            return self.default_value is not None

        value_str = str(value)
        return any(pattern.search(value_str) for pattern, _ in self.patterns)


class TransformationEngine:
    """Manages and applies data transformations."""

    def __init__(self):
        self.transformers: Dict[str, DataTransformer] = {}
        self._register_default_transformers()

    def _register_default_transformers(self):
        """Register commonly used transformers."""
        self.transformers.update(
            {
                "parse_date": DateParserTransformer(),
                "parse_datetime": DateParserTransformer(
                    [
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%d",
                        "%d.%m.%Y %H:%M:%S",
                        "%d.%m.%Y",
                    ]
                ),
                "serialize_json": JsonSerializerTransformer(),
            }
        )

    def register_transformer(self, name: str, transformer: DataTransformer):
        """Register a custom transformer."""
        self.transformers[name] = transformer

    def register_lookup_table(
        self, name: str, lookup_table: Dict[str, str], default_value: Optional[str] = None
    ):
        """Register a lookup table transformer."""
        self.transformers[name] = LookupTableTransformer(lookup_table, default_value)

    def register_lookup_from_file(
        self,
        name: str,
        file_path: Path,
        key_field: str,
        value_field: str,
        default_value: Optional[str] = None,
    ):
        """Register a lookup table transformer from file."""
        self.transformers[name] = LookupTableTransformer.from_file(
            file_path, key_field, value_field, default_value
        )

    def register_value_mapping(
        self, name: str, value_mappings: Dict[str, str], case_sensitive: bool = False
    ):
        """Register a value mapping transformer."""
        self.transformers[name] = ValueMappingTransformer(value_mappings, case_sensitive)

    def transform(self, value: Any, transformer_name: str, context: Dict[str, Any] = None) -> Any:
        """Apply a transformation."""
        if transformer_name not in self.transformers:
            raise ValueError(f"Unknown transformer: {transformer_name}")

        transformer = self.transformers[transformer_name]
        return transformer.transform(value, context)

    def validate(self, value: Any, transformer_name: str) -> bool:
        """Validate if value can be transformed."""
        if transformer_name not in self.transformers:
            return False

        transformer = self.transformers[transformer_name]
        return transformer.validate(value)

    def list_transformers(self) -> List[str]:
        """List all registered transformers."""
        return list(self.transformers.keys())


# Example configurations for specific legal domains
def setup_polish_legal_transformations(engine: TransformationEngine, data_dir: Path):
    """Setup transformations specific to Polish legal documents."""

    # Court ID to court name mapping
    court_mapping_file = data_dir / "court_mappings.json"
    if court_mapping_file.exists():
        engine.register_lookup_from_file(
            "normalize_court_name",
            court_mapping_file,
            "court_id",
            "court_name",
            default_value="Unknown Court",
        )

    # Tax category normalization
    tax_categories = {
        "VAT": "value_added_tax",
        "PIT": "personal_income_tax",
        "CIT": "corporate_income_tax",
        "Podatek VAT": "value_added_tax",
        "Podatek dochodowy": "income_tax",
    }
    engine.register_value_mapping("normalize_tax_category", tax_categories)

    # Court type standardization
    court_types = {
        "Sąd Okręgowy": "district_court",
        "Sąd Apelacyjny": "appellate_court",
        "Sąd Najwyższy": "supreme_court",
        "NSA": "supreme_administrative_court",
    }
    engine.register_value_mapping("normalize_court_type", court_types)

    # Case number standardization
    case_number_patterns = [
        (r"(\d+)\s+([A-Z]+)\s+(\d+/\d+)", r"\1 \2 \3"),  # Standardize spacing
        (r"Sygn\.\s*akt\s*(\S+)", r"\1"),  # Remove "Sygn. akt" prefix
    ]
    engine.register_transformer("normalize_case_number", RegexTransformer(case_number_patterns))


# Example usage
if __name__ == "__main__":
    # Create transformation engine
    engine = TransformationEngine()

    # Setup Polish legal transformations
    data_dir = Path("data/mappings")
    setup_polish_legal_transformations(engine, data_dir)

    # Example transformations
    print("Available transformers:", engine.list_transformers())

    # Date parsing
    date_result = engine.transform("15.03.2024", "parse_date")
    print(f"Date parsing: '15.03.2024' -> '{date_result}'")

    # JSON serialization
    json_result = engine.transform(["judge1", "judge2"], "serialize_json")
    print(f"JSON serialization: ['judge1', 'judge2'] -> '{json_result}'")

    # Value mapping (if registered)
    if "normalize_tax_category" in engine.list_transformers():
        tax_result = engine.transform("VAT", "normalize_tax_category")
        print(f"Tax category: 'VAT' -> '{tax_result}'")
