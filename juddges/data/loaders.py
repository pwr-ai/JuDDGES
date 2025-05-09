"""
Dataset loaders for Weaviate ingestion.
"""

import multiprocessing
import tempfile

import polars as pl
from datasets import Dataset, load_dataset
from loguru import logger

from juddges.config import EmbeddingConfig

# Define DEFAULT_PROCESSING_PROC locally instead of importing from scripts
DEFAULT_PROCESSING_PROC = max(1, multiprocessing.cpu_count() - 2)


class DatasetLoader:
    """Utility class for loading datasets with embedding data."""

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize dataset loader with embedding configuration.

        Args:
            config: Embedding configuration with dataset paths
        """
        self.config = config
        self._check_embeddings_exist()

    def _check_embeddings_exist(self) -> None:
        """Check if embedding directories exist."""
        assert self.config.agg_embeddings_dir.exists(), (
            f"Embeddings directory {self.config.agg_embeddings_dir} does not exist"
        )
        assert self.config.chunk_embeddings_dir.exists(), (
            f"Embeddings directory {self.config.chunk_embeddings_dir} does not exist"
        )

    def load_chunk_dataset(self) -> Dataset:
        """
        Load the chunk embeddings dataset.

        Returns:
            Hugging Face dataset containing chunk embeddings

        Raises:
            ValueError: If the dataset is empty or not loaded correctly
            Exception: If dataset loading fails
        """
        logger.info(f"Loading chunk embeddings from {self.config.chunk_embeddings_dir}...")
        try:
            chunk_ds = load_dataset(
                "parquet",
                data_dir=self.config.chunk_embeddings_dir,
                num_proc=DEFAULT_PROCESSING_PROC,
            )
            if not chunk_ds or "train" not in chunk_ds or chunk_ds["train"].num_rows == 0:
                logger.error("Chunk embeddings dataset is empty or not loaded correctly.")
                raise ValueError("Empty chunk embeddings dataset")
            return chunk_ds["train"]
        except Exception as e:
            logger.error(f"Failed to load chunk embeddings dataset: {e}")
            raise

    def load_document_dataset(self) -> Dataset:
        """
        Load the document embeddings dataset.

        Returns:
            Hugging Face dataset containing document embeddings

        Raises:
            ValueError: If the dataset is empty or not loaded correctly
            Exception: If dataset loading fails
        """
        logger.info(
            f"Loading document dataset from {self.config.dataset_name} and {self.config.agg_embeddings_dir}..."
        )
        try:
            ds_polars = pl.scan_parquet(f"hf://datasets/{self.config.dataset_name}/data/*.parquet")
            agg_ds_polars = pl.scan_parquet(f"{self.config.agg_embeddings_dir}/*.parquet")

            logger.info("Preparing aggregated dataset (it may take a few minutes)...")
            with tempfile.NamedTemporaryFile(suffix=".parquet") as temp_file:
                agg_ds_polars.join(ds_polars, on="judgment_id", how="left").sink_parquet(
                    temp_file.name
                )
                agg_ds = load_dataset(
                    "parquet",
                    data_files=temp_file.name,
                    num_proc=DEFAULT_PROCESSING_PROC,
                )["train"]

            if not agg_ds or agg_ds.num_rows == 0:
                logger.error("Aggregated embeddings dataset is empty or not loaded correctly.")
                raise ValueError("Empty aggregated embeddings dataset")

            return agg_ds
        except Exception as e:
            logger.error(f"Failed to load aggregated embeddings dataset: {e}")
            raise
