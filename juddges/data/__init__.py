"""Data package for legal documents."""

from juddges.data.dataset_mapper import DatasetToWeaviateMapper
from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase
from juddges.data.loaders import DATASET_COLUMN_MAPPINGS, DatasetLoader
from juddges.data.schemas import DocumentChunk, DocumentType, LegalDocument, SegmentType

__all__ = [
    "DatasetToWeaviateMapper",
    "WeaviateLegalDocumentsDatabase",
    "DatasetLoader",
    "DATASET_COLUMN_MAPPINGS",
    "LegalDocument",
    "DocumentChunk",
    "DocumentType",
    "SegmentType",
]
