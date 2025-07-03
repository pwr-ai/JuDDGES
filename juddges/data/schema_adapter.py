"""
Dynamic Weaviate schema adaptation for different datasets.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from loguru import logger

import weaviate.classes.config as wvcc
from juddges.data.dataset_registry import DatasetConfig
from juddges.settings import VectorName


class WeaviateSchemaAdapter:
    """Dynamically adapt Weaviate schema for different datasets."""

    # Core schema properties that should always be present
    CORE_PROPERTIES = [
        wvcc.Property(
            name="document_id",
            data_type=wvcc.DataType.TEXT,
            description="Unique identifier for the document",
            skip_vectorization=True,
        ),
        wvcc.Property(
            name="document_type",
            data_type=wvcc.DataType.TEXT,
            description="Type of legal document (judgment, tax interpretation, etc.)",
            skip_vectorization=False,
        ),
        wvcc.Property(
            name="language",
            data_type=wvcc.DataType.TEXT,
            description="Document language",
            skip_vectorization=True,
        ),
        wvcc.Property(
            name="country",
            data_type=wvcc.DataType.TEXT,
            description="Country of origin",
            skip_vectorization=True,
        ),
        wvcc.Property(
            name="ingestion_date",
            data_type=wvcc.DataType.DATE,
            description="When document was ingested (ISO format datetime)",
            skip_vectorization=True,
        ),
    ]

    def create_adaptive_schema(
        self, config: DatasetConfig, sample_data: Optional[Dict[str, Any]] = None
    ) -> List[wvcc.Property]:
        """Create schema that adapts to dataset structure."""
        logger.info(f"Creating adaptive schema for dataset: {config.name}")

        # Start with core properties
        properties = self.CORE_PROPERTIES.copy()
        existing_names = {prop.name for prop in properties}

        # Add properties based on dataset configuration
        mapped_fields = set(config.column_mapping.values())

        for target_field in mapped_fields:
            if target_field in existing_names:
                continue

            prop = self._create_property_from_config(target_field, config, sample_data)
            if prop:
                properties.append(prop)
                existing_names.add(prop.name)

        # Add any additional properties from sample data
        if sample_data:
            for field_name, value in sample_data.items():
                if field_name not in existing_names:
                    prop = self._infer_property_from_sample(field_name, value, config)
                    if prop:
                        properties.append(prop)
                        existing_names.add(prop.name)

        logger.info(f"Created schema with {len(properties)} properties")
        return properties

    def _create_property_from_config(
        self, field_name: str, config: DatasetConfig, sample_data: Optional[Dict] = None
    ) -> Optional[wvcc.Property]:
        """Create a Weaviate property based on dataset configuration."""

        # Determine data type based on field configuration
        if field_name in config.date_fields:
            data_type = wvcc.DataType.DATE
            skip_vectorization = True
        elif field_name in config.array_fields:
            # Determine if it's text array or object array
            if sample_data and field_name in sample_data:
                sample_value = sample_data[field_name]
                if isinstance(sample_value, list) and sample_value:
                    if isinstance(sample_value[0], str):
                        data_type = wvcc.DataType.TEXT_ARRAY
                    else:
                        data_type = wvcc.DataType.OBJECT_ARRAY
                else:
                    data_type = wvcc.DataType.TEXT_ARRAY  # Default
            else:
                data_type = wvcc.DataType.TEXT_ARRAY
            skip_vectorization = field_name not in config.text_fields
        elif field_name in config.json_fields:
            data_type = wvcc.DataType.TEXT  # Store JSON as text
            skip_vectorization = True
        elif field_name in config.text_fields:
            data_type = wvcc.DataType.TEXT
            skip_vectorization = False
        else:
            # Try to infer from sample data
            if sample_data and field_name in sample_data:
                data_type = self._infer_weaviate_type(sample_data[field_name])
                skip_vectorization = not self._should_vectorize(
                    field_name, sample_data[field_name], config
                )
            else:
                data_type = wvcc.DataType.TEXT  # Safe default
                skip_vectorization = True

        return wvcc.Property(
            name=field_name,
            data_type=data_type,
            description=f"Field from dataset {config.name}",
            skip_vectorization=skip_vectorization,
        )

    def _infer_property_from_sample(
        self, field_name: str, value: Any, config: DatasetConfig
    ) -> Optional[wvcc.Property]:
        """Infer property configuration from sample data."""
        data_type = self._infer_weaviate_type(value)
        should_vectorize = self._should_vectorize(field_name, value, config)

        return wvcc.Property(
            name=field_name,
            data_type=data_type,
            description=f"Auto-generated from dataset field {field_name}",
            skip_vectorization=not should_vectorize,
        )

    def _infer_weaviate_type(self, value: Any) -> wvcc.DataType:
        """Infer Weaviate data type from Python value."""
        if value is None:
            return wvcc.DataType.TEXT  # Safe default

        if isinstance(value, bool):
            return wvcc.DataType.BOOLEAN
        elif isinstance(value, int):
            return wvcc.DataType.INT
        elif isinstance(value, float):
            return wvcc.DataType.NUMBER
        elif isinstance(value, str):
            # Check if it looks like a date
            if self._looks_like_date(value):
                return wvcc.DataType.DATE
            return wvcc.DataType.TEXT
        elif isinstance(value, (datetime, date)):
            return wvcc.DataType.DATE
        elif isinstance(value, list):
            if not value:
                return wvcc.DataType.TEXT_ARRAY  # Default for empty list

            first_item = value[0]
            if isinstance(first_item, str):
                return wvcc.DataType.TEXT_ARRAY
            elif isinstance(first_item, (int, float)):
                return wvcc.DataType.NUMBER_ARRAY
            elif isinstance(first_item, bool):
                return wvcc.DataType.BOOLEAN_ARRAY
            else:
                return wvcc.DataType.OBJECT_ARRAY
        elif isinstance(value, dict):
            return wvcc.DataType.OBJECT
        else:
            return wvcc.DataType.TEXT  # Default fallback

    def _looks_like_date(self, value: str) -> bool:
        """Check if a string value looks like a date."""
        if len(value) < 8:  # Too short to be a meaningful date
            return False

        # Check for common date patterns
        import re

        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
            r"\d{2}/\d{2}/\d{4}",  # MM/DD/YYYY
            r"\d{4}/\d{2}/\d{2}",  # YYYY/MM/DD
            r"\d{2}-\d{2}-\d{4}",  # MM-DD-YYYY
        ]

        return any(re.search(pattern, value) for pattern in date_patterns)

    def _should_vectorize(self, field_name: str, value: Any, config: DatasetConfig) -> bool:
        """Determine if a field should be vectorized."""
        # Always vectorize text fields specified in config
        if field_name in config.text_fields:
            return True

        # Don't vectorize IDs, dates, or small text fields
        if field_name.endswith("_id") or field_name.endswith("_date") or "id" in field_name.lower():
            return False

        # For string values, vectorize if they're substantial text
        if isinstance(value, str):
            # Vectorize longer text content
            if len(value) > 100:
                return True
            # Vectorize fields that look like content
            content_indicators = ["text", "content", "summary", "description", "abstract"]
            if any(indicator in field_name.lower() for indicator in content_indicators):
                return True

        return False

    def create_vectorizer_config(self, text_fields: List[str]) -> List[wvcc.Configure.NamedVectors]:
        """Create vectorizer configuration for multiple named vectors."""
        # Determine which fields to vectorize
        vectorize_fields = ["full_text"]  # Default

        # Add other text fields if full_text is not available
        if "full_text" not in text_fields and text_fields:
            vectorize_fields = text_fields[:3]  # Limit to first 3 text fields

        # Create named vectors for different use cases
        return [
            wvcc.Configure.NamedVectors.text2vec_transformers(
                name=VectorName.BASE,
                vectorize_collection_name=False,
                source_properties=vectorize_fields,
                vector_index_config=wvcc.Configure.VectorIndex.hnsw(),
            ),
            wvcc.Configure.NamedVectors.text2vec_transformers(
                name=VectorName.DEV,
                vectorize_collection_name=False,
                source_properties=vectorize_fields,
                vector_index_config=wvcc.Configure.VectorIndex.hnsw(),
            ),
            wvcc.Configure.NamedVectors.text2vec_transformers(
                name=VectorName.FAST,
                vectorize_collection_name=False,
                source_properties=vectorize_fields,
                vector_index_config=wvcc.Configure.VectorIndex.flat(),  # Faster for small datasets
            ),
        ]

    def create_chunk_schema(self, config: DatasetConfig) -> List[wvcc.Property]:
        """Create schema for document chunks collection."""
        chunk_properties = [
            wvcc.Property(
                name="document_id",
                data_type=wvcc.DataType.TEXT,
                description="ID of the parent document",
                skip_vectorization=True,
            ),
            wvcc.Property(
                name="document_type",
                data_type=wvcc.DataType.TEXT,
                description="Type of document",
                skip_vectorization=True,
            ),
            wvcc.Property(
                name="language",
                data_type=wvcc.DataType.TEXT,
                description="Language of the document chunk",
                skip_vectorization=True,
            ),
            wvcc.Property(
                name="chunk_id",
                data_type=wvcc.DataType.NUMBER,
                description="Chunk identifier",
                skip_vectorization=True,
            ),
            wvcc.Property(
                name="chunk_text",
                data_type=wvcc.DataType.TEXT,
                description="Text content of the chunk",
                skip_vectorization=False,
            ),
            wvcc.Property(
                name="position",
                data_type=wvcc.DataType.NUMBER,
                description="Order in document",
                skip_vectorization=True,
            ),
            wvcc.Property(
                name="chunk_size",
                data_type=wvcc.DataType.NUMBER,
                description="Size of chunk in characters",
                skip_vectorization=True,
            ),
            wvcc.Property(
                name="segment_type",
                data_type=wvcc.DataType.TEXT,
                description="Type of segment (if available)",
                skip_vectorization=False,
            ),
        ]

        return chunk_properties

    def validate_schema_compatibility(
        self, existing_schema: List[wvcc.Property], new_schema: List[wvcc.Property]
    ) -> Dict[str, Any]:
        """Validate compatibility between existing and new schema."""
        existing_props = {prop.name: prop for prop in existing_schema}
        new_props = {prop.name: prop for prop in new_schema}

        compatibility_report = {
            "compatible": True,
            "issues": [],
            "new_properties": [],
            "type_conflicts": [],
            "recommendations": [],
        }

        # Check for new properties
        for name, prop in new_props.items():
            if name not in existing_props:
                compatibility_report["new_properties"].append(name)

        # Check for type conflicts
        for name, new_prop in new_props.items():
            if name in existing_props:
                existing_prop = existing_props[name]
                if existing_prop.data_type != new_prop.data_type:
                    compatibility_report["type_conflicts"].append(
                        {
                            "property": name,
                            "existing_type": existing_prop.data_type,
                            "new_type": new_prop.data_type,
                        }
                    )
                    compatibility_report["compatible"] = False

        # Generate recommendations
        if compatibility_report["new_properties"]:
            compatibility_report["recommendations"].append(
                "Consider adding new properties to existing schema"
            )

        if compatibility_report["type_conflicts"]:
            compatibility_report["recommendations"].append(
                "Resolve type conflicts before proceeding with ingestion"
            )

        return compatibility_report
