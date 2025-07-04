"""
Configuration classes for Weaviate ingestion.
"""

import multiprocessing
import os
from dataclasses import dataclass

# Default values for normal operation
DEFAULT_INGEST_BATCH_SIZE = 32
DEFAULT_UPSERT = True

# Debug mode configuration
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
if DEBUG:
    # Debug defaults for faster testing
    DEFAULT_INGEST_BATCH_SIZE = 2
    DEFAULT_PROCESSING_PROC = 1
    DEFAULT_INGEST_PROC = 1
else:
    # Production defaults
    DEFAULT_PROCESSING_PROC = max(1, multiprocessing.cpu_count() - 2)
    DEFAULT_INGEST_PROC = max(1, int(DEFAULT_PROCESSING_PROC / 2))


@dataclass
class IngestConfig:
    """Configuration for ingestion process."""

    batch_size: int = DEFAULT_INGEST_BATCH_SIZE
    upsert: bool = DEFAULT_UPSERT
    processing_proc: int = DEFAULT_PROCESSING_PROC
    ingest_proc: int = DEFAULT_INGEST_PROC
