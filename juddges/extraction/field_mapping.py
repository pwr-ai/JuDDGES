"""Field mapping and transformation between extraction schema and Weaviate properties.

This module handles the conversion of extracted data to Weaviate-compatible formats,
including JSON serialization, array parsing, and special case handling (e.g., keywords).
"""

import json
from typing import Any, Dict, Set

from loguru import logger


# Field mapping from extraction schema to Weaviate properties
# All properties already exist in Weaviate - no need for "extracted_" prefix!
# NOTE: document_number, document_type, and date_issued are excluded as they're always already valid
EXTRACTION_TO_WEAVIATE_MAPPING = {
    # Direct TEXT mappings (existing properties in Weaviate)
    "title": "title",
    "summary": "summary",
    "thesis": "thesis",
    # TEXT_ARRAY (existing property - native array support)
    "keywords": "keywords",
    # NEW properties (added to Weaviate schema)
    "factual_state": "factual_state",
    "legal_state": "legal_state",
    # TEXT (JSON) properties (existing - need JSON serialization)
    "legal_references": "legal_references",
    "legal_concepts": "legal_concepts",
    "parties": "parties",
    "outcome": "outcome",
    "legal_analysis": "legal_analysis",
    "judgment_specific": "judgment_specific",
    "tax_interpretation_specific": "tax_interpretation_specific",
}


# Fields that need JSON serialization (stored as TEXT in Weaviate)
JSON_SERIALIZED_FIELDS: Set[str] = {
    "legal_references",
    "legal_concepts",
    "parties",
    "outcome",
    "legal_analysis",
    "judgment_specific",
    "tax_interpretation_specific",
}


class FieldMapper:
    """Handles field transformations between extraction results and Weaviate properties."""

    def __init__(
        self,
        field_mapping: Dict[str, str] = EXTRACTION_TO_WEAVIATE_MAPPING,
        json_fields: Set[str] = JSON_SERIALIZED_FIELDS,
    ):
        """Initialize field mapper.

        Args:
            field_mapping: Dictionary mapping extraction field names to Weaviate property names
            json_fields: Set of field names that require JSON serialization
        """
        self.field_mapping = field_mapping
        self.json_fields = json_fields

    def build_update_payload(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform extracted data to Weaviate property update payload.

        Handles:
        - Direct TEXT fields (no transformation)
        - TEXT_ARRAY fields (keywords - native array support)
        - TEXT (JSON) fields (need JSON serialization for lists/objects)
        - Special cases: keywords as JSON string or comma-separated values

        Args:
            extracted_data: Dictionary with extracted fields from LLM

        Returns:
            Dictionary with Weaviate properties ready for PATCH/PUT request
        """
        payload = {}

        for extracted_field, weaviate_property in self.field_mapping.items():
            value = extracted_data.get(extracted_field)

            # Skip empty/null values
            if value is None or value == "":
                continue

            # Handle list fields
            if isinstance(value, list):
                transformed = self._transform_list_field(extracted_field, value)
                if transformed is not None:
                    payload[weaviate_property] = transformed

            # Handle object/dict fields (judgment_specific, tax_interpretation_specific, etc.)
            elif isinstance(value, dict):
                transformed = self._transform_dict_field(extracted_field, value)
                if transformed is not None:
                    payload[weaviate_property] = transformed

            # Handle string fields
            elif isinstance(value, str):
                transformed = self._transform_string_field(extracted_field, value)
                if transformed is not None:
                    payload[weaviate_property] = transformed

            else:
                # Default: direct assignment for other types (numbers, booleans, etc.)
                payload[weaviate_property] = value

        return payload

    def _transform_list_field(self, field_name: str, value: list) -> Any:
        """Transform list field to appropriate Weaviate format.

        Args:
            field_name: Name of the extraction field
            value: List value to transform

        Returns:
            Transformed value or None if empty
        """
        # Filter out empty strings and None values
        cleaned_list = [v for v in value if v and str(v).strip()]
        if not cleaned_list:
            return None

        # keywords is TEXT_ARRAY - use directly without JSON serialization
        if field_name == "keywords":
            return cleaned_list

        # Other lists need JSON serialization
        if field_name in self.json_fields:
            return json.dumps(cleaned_list, ensure_ascii=False)

        # Default: direct assignment
        return cleaned_list

    def _transform_dict_field(self, field_name: str, value: dict) -> Any:
        """Transform dict field to appropriate Weaviate format.

        Args:
            field_name: Name of the extraction field
            value: Dict value to transform

        Returns:
            Transformed value or None if empty
        """
        # Only include if dict has meaningful content
        if not value or all(v is None or v == "" for v in value.values()):
            return None

        # Most dict fields need JSON serialization
        if field_name in self.json_fields:
            return json.dumps(value, ensure_ascii=False)

        # Default: direct assignment (shouldn't happen for legal docs)
        return value

    def _transform_string_field(self, field_name: str, value: str) -> Any:
        """Transform string field to appropriate Weaviate format.

        Special handling for keywords field which might be:
        - JSON string: '["keyword1", "keyword2"]'
        - Comma-separated: "keyword1, keyword2, keyword3"
        - Single keyword: "keyword"

        Args:
            field_name: Name of the extraction field
            value: String value to transform

        Returns:
            Transformed value or None if empty
        """
        # Special case: keywords field might be a JSON string or comma-separated string
        if field_name == "keywords":
            return self._parse_keywords_field(value)

        # Fields that need JSON wrapping (outcome, legal_analysis, etc.)
        if field_name in self.json_fields:
            return json.dumps(value, ensure_ascii=False)

        # Direct TEXT fields (most common: title, summary, thesis, factual_state, legal_state)
        return value

    def _parse_keywords_field(self, value: str) -> Any:
        """Parse keywords field from various string formats.

        Handles three formats:
        1. JSON array string: '["keyword1", "keyword2"]'
        2. Comma-separated: "keyword1, keyword2, keyword3"
        3. Single keyword: "keyword"

        Args:
            value: String representation of keywords

        Returns:
            List of keywords or None if empty
        """
        # Try to parse as JSON array first
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed_list = json.loads(value)
                if isinstance(parsed_list, list):
                    cleaned_list = [v for v in parsed_list if v and str(v).strip()]
                    if cleaned_list:
                        return cleaned_list
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"Failed to parse keywords as JSON list: {value[:100]}")

        # If not JSON, try comma-separated string
        if "," in value or value.strip():  # Single keyword or comma-separated
            # Split by comma and clean up
            keywords_list = [k.strip() for k in value.split(",") if k.strip()]
            if keywords_list:
                return keywords_list

        # Empty or invalid
        return None


# Global instance for convenience
default_field_mapper = FieldMapper()


def build_update_payload(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function using default field mapper.

    Args:
        extracted_data: Dictionary with extracted fields from LLM

    Returns:
        Dictionary with Weaviate properties ready for PATCH request
    """
    return default_field_mapper.build_update_payload(extracted_data)
