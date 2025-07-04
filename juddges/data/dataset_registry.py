"""
Dynamic dataset registry for HuggingFace dataset configurations.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

from juddges.settings import CONFIG_PATH


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset ingestion."""

    name: str
    column_mapping: Dict[str, str]
    required_fields: List[str]
    text_fields: List[str]  # Fields to use for embeddings
    date_fields: List[str]
    array_fields: List[str]
    json_fields: List[str]  # Fields that should be JSON serialized
    default_values: Dict[str, Any]
    document_type: str = "judgment"  # Default document type
    chunk_strategy: str = "recursive"  # How to chunk documents
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_path: Optional[str] = None  # Path to pre-computed embeddings
    chunks_path: Optional[str] = None  # Path to chunk embeddings
    num_proc: Optional[int] = None  # Number of processes for parallel processing
    batch_size: int = 1000  # Batch size for dataset operations

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        """Create from dictionary."""
        return cls(**data)


class DatasetRegistry:
    """Registry for managing dataset configurations."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize registry with configuration directory."""
        self.config_dir = config_dir or (CONFIG_PATH / "datasets")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.registry: Dict[str, DatasetConfig] = {}
        self._load_configs()

    def register_dataset(self, config: DatasetConfig) -> None:
        """Register a new dataset configuration."""
        logger.info(f"Registering dataset configuration: {config.name}")
        self.registry[config.name] = config
        self._save_config(config)

    def get_config(self, dataset_name: str) -> Optional[DatasetConfig]:
        """Get configuration for a dataset."""
        return self.registry.get(dataset_name)

    def list_datasets(self) -> List[str]:
        """List all registered datasets."""
        return list(self.registry.keys())

    def remove_dataset(self, dataset_name: str) -> bool:
        """Remove a dataset configuration."""
        if dataset_name in self.registry:
            del self.registry[dataset_name]
            config_file = self.config_dir / f"{dataset_name}.yaml"
            if config_file.exists():
                config_file.unlink()
            logger.info(f"Removed dataset configuration: {dataset_name}")
            return True
        return False

    def update_config(self, dataset_name: str, updates: Dict[str, Any]) -> bool:
        """Update existing dataset configuration."""
        if dataset_name not in self.registry:
            return False

        config = self.registry[dataset_name]
        config_dict = config.to_dict()
        config_dict.update(updates)

        try:
            updated_config = DatasetConfig.from_dict(config_dict)
            self.registry[dataset_name] = updated_config
            self._save_config(updated_config)
            logger.info(f"Updated dataset configuration: {dataset_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update configuration for {dataset_name}: {e}")
            return False

    def _load_configs(self) -> None:
        """Load all configuration files from disk."""
        if not self.config_dir.exists():
            return

        for config_file in self.config_dir.glob("*.yaml"):
            try:
                with open(config_file, "r") as f:
                    data = yaml.safe_load(f)
                config = DatasetConfig.from_dict(data)
                self.registry[config.name] = config
                logger.debug(f"Loaded dataset config: {config.name}")
            except Exception as e:
                logger.error(f"Failed to load config from {config_file}: {e}")

    def _save_config(self, config: DatasetConfig) -> None:
        """Save configuration to disk."""
        config_file = self.config_dir / f"{config.name.replace('/', '_')}.yaml"
        try:
            with open(config_file, "w") as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
            logger.debug(f"Saved config for {config.name} to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save config for {config.name}: {e}")

    def create_default_configs(self) -> None:
        """Create default configurations for known datasets."""
        # Polish court dataset
        pl_court_config = DatasetConfig(
            name="juddges/pl-court-raw",
            column_mapping={
                "judgment_id": "document_id",
                "docket_number": "document_number",
                "judgment_date": "date_issued",
                "publication_date": "publication_date",
                "court_id": "source_id",
                "judgment_type": "judgment_type",
                "excerpt": "summary",
                "xml_content": "raw_content",
                "full_text": "full_text",
                "court_name": "court_name",
                "country": "country",
                "thesis": "thesis",
                "keywords": "keywords",
                "judges": "judges",
                "legal_bases": "legal_bases",
            },
            required_fields=["document_id", "full_text"],
            text_fields=["full_text", "summary", "thesis"],
            date_fields=["date_issued", "publication_date"],
            array_fields=["keywords", "judges", "legal_bases"],
            json_fields=["extracted_legal_bases"],
            default_values={"language": "pl", "country": "Poland", "document_type": "judgment"},
        )

        # English court dataset
        en_court_config = DatasetConfig(
            name="juddges/en-court-raw",
            column_mapping={
                "judgment_id": "document_id",
                "category": "document_type",
                "content": "full_text",
                "issued_on": "date_issued",
                "case_number": "document_number",
                "lang": "language",
                "country": "country",
                "abstract": "summary",
                "main_point": "thesis",
                "tags": "keywords",
            },
            required_fields=["document_id", "full_text"],
            text_fields=["full_text", "summary", "thesis"],
            date_fields=["date_issued"],
            array_fields=["keywords"],
            json_fields=[],
            default_values={"language": "en", "country": "England", "document_type": "judgment"},
        )

        self.register_dataset(pl_court_config)
        self.register_dataset(en_court_config)
        logger.info("Created default dataset configurations")


# Global registry instance
_registry = None


def get_registry() -> DatasetRegistry:
    """Get global dataset registry instance."""
    global _registry
    if _registry is None:
        _registry = DatasetRegistry()
    return _registry
