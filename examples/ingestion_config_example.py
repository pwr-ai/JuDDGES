#!/usr/bin/env python3
"""
Example showing how the streaming ingester should be configured for a standalone package.
This demonstrates the configuration-driven approach needed for external packaging.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FieldMapping:
    """Configuration for mapping dataset fields to Weaviate properties."""

    source_field: str
    target_field: str
    required: bool = False
    transform: Optional[str] = None  # Name of transformation function
    default_value: Optional[Any] = None


@dataclass
class DatasetConfig:
    """Complete configuration for a dataset ingestion."""

    name: str
    huggingface_path: str
    description: str = ""

    # Field mappings
    field_mappings: List[FieldMapping] = None

    # Processing settings
    chunk_size: int = 512
    chunk_overlap: int = 128
    batch_size: int = 32

    # Weaviate settings
    document_collection: str = "LegalDocument"
    chunk_collection: str = "DocumentChunk"

    # Data transformations
    transformations: Dict[str, Any] = None  # Custom transformation configs

    # Validation rules
    required_fields: List[str] = None
    text_fields: List[str] = None
    date_fields: List[str] = None

    def __post_init__(self):
        if self.field_mappings is None:
            self.field_mappings = []
        if self.transformations is None:
            self.transformations = {}
        if self.required_fields is None:
            self.required_fields = ["document_id", "full_text"]
        if self.text_fields is None:
            self.text_fields = ["full_text"]
        if self.date_fields is None:
            self.date_fields = []


# Example configurations for different legal datasets
POLISH_COURT_CONFIG = DatasetConfig(
    name="pl-court-raw",
    huggingface_path="JuDDGES/pl-court-raw",
    description="Polish court judgments from JuDDGES",
    field_mappings=[
        FieldMapping("judgment_id", "document_id", required=True),
        FieldMapping("judgment_date", "date_issued", transform="parse_date"),
        FieldMapping("court_name", "issuing_body", transform="normalize_court_name"),
        FieldMapping("judgment_type", "document_type"),
        FieldMapping("full_text", "full_text", required=True),
        FieldMapping("legal_bases", "metadata.legal_bases", transform="serialize_json"),
        FieldMapping("judges", "metadata.judges", transform="serialize_json"),
        FieldMapping("keywords", "metadata.keywords", transform="serialize_json"),
        FieldMapping("country", "country", default_value="Poland"),
    ],
    transformations={
        "court_name_mapping": {
            "type": "lookup_table",
            "source_file": "data/court_mappings.json",
            "key_field": "court_id",
            "value_field": "court_name",
        },
        "date_formats": ["YYYY-MM-DD", "DD.MM.YYYY", "DD/MM/YYYY"],
    },
    required_fields=["document_id", "full_text", "court_name"],
    text_fields=["full_text", "summary"],
    date_fields=["judgment_date"],
)

TAX_INTERPRETATION_CONFIG = DatasetConfig(
    name="tax-interpretations",
    huggingface_path="AI-Tax/tax-interpretations",
    description="Polish tax interpretations",
    field_mappings=[
        FieldMapping("id", "document_id", required=True),
        FieldMapping("SYG", "document_number"),
        FieldMapping("DT_WYD", "date_issued", transform="parse_date"),
        FieldMapping("TEZA", "title"),
        FieldMapping("TRESC_INTERESARIUSZ", "full_text", required=True),
        FieldMapping("KATEGORIA_INFORMACJI", "document_type"),
        FieldMapping("PRZEPISY", "metadata.tax_provisions", transform="serialize_json"),
        FieldMapping("SLOWA_KLUCZOWE", "metadata.keywords", transform="serialize_json"),
        FieldMapping("_fetched_at", "metadata.fetched_at", transform="parse_datetime"),
    ],
    transformations={
        "tax_category_mapping": {
            "type": "value_mapping",
            "mappings": {
                "VAT": "value_added_tax",
                "PIT": "personal_income_tax",
                "CIT": "corporate_income_tax",
            },
        }
    },
    required_fields=["document_id", "full_text"],
    text_fields=["full_text", "title"],
    date_fields=["DT_WYD", "_fetched_at"],
)

ENGLISH_APPEALS_CONFIG = DatasetConfig(
    name="en-appealcourt-coded",
    huggingface_path="JuDDGES/en-appealcourt-coded",
    description="English court appeals from JuDDGES",
    field_mappings=[
        FieldMapping("id", "document_id", required=True),
        FieldMapping("case_number", "document_number"),
        FieldMapping("judgment_date", "date_issued", transform="parse_date"),
        FieldMapping("court", "issuing_body"),
        FieldMapping("case_title", "title"),
        FieldMapping("full_text", "full_text", required=True),
        FieldMapping("judges", "metadata.judges", transform="serialize_json"),
        FieldMapping("legal_areas", "metadata.legal_areas", transform="serialize_json"),
        FieldMapping("outcome", "metadata.outcome"),
        FieldMapping("country", "country", default_value="UK"),
    ],
    required_fields=["document_id", "full_text"],
    text_fields=["full_text", "title"],
    date_fields=["judgment_date"],
)


class ConfigurationManager:
    """Manages dataset configurations for the ingestion package."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("configs/datasets")
        self.configs: Dict[str, DatasetConfig] = {}
        self._load_default_configs()

    def _load_default_configs(self):
        """Load default configurations."""
        self.configs.update(
            {
                "pl-court-raw": POLISH_COURT_CONFIG,
                "tax-interpretations": TAX_INTERPRETATION_CONFIG,
                "en-appealcourt-coded": ENGLISH_APPEALS_CONFIG,
            }
        )

    def get_config(self, dataset_name: str) -> Optional[DatasetConfig]:
        """Get configuration for a dataset."""
        return self.configs.get(dataset_name)

    def register_config(self, config: DatasetConfig):
        """Register a new dataset configuration."""
        self.configs[config.name] = config

    def save_config(self, config: DatasetConfig, file_path: Optional[Path] = None):
        """Save configuration to file."""
        if file_path is None:
            file_path = self.config_dir / f"{config.name}.json"

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(asdict(config), f, indent=2, default=str)

    def load_config(self, file_path: Path) -> DatasetConfig:
        """Load configuration from file."""
        with open(file_path) as f:
            data = json.load(f)

        # Convert field mappings back to FieldMapping objects
        if "field_mappings" in data and data["field_mappings"]:
            data["field_mappings"] = [FieldMapping(**mapping) for mapping in data["field_mappings"]]

        return DatasetConfig(**data)

    def list_configs(self) -> List[str]:
        """List all available dataset configurations."""
        return list(self.configs.keys())


# Example usage for standalone package
if __name__ == "__main__":
    config_manager = ConfigurationManager()

    # Get configuration for Polish courts
    config = config_manager.get_config("pl-court-raw")
    print(f"Config for {config.name}:")
    print(f"  HF path: {config.huggingface_path}")
    print(f"  Field mappings: {len(config.field_mappings)}")

    # Save configuration to file
    config_manager.save_config(config, Path("example_config.json"))

    # Load configuration from file
    loaded_config = config_manager.load_config(Path("example_config.json"))
    print(f"Loaded config: {loaded_config.name}")
