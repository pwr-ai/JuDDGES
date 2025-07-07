from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class DocumentType(str, Enum):
    """Enumeration of supported legal document types."""

    JUDGMENT = "judgment"
    TAX_INTERPRETATION = "tax_interpretation"
    LEGAL_ACT = "legal_act"


class SegmentType(str, Enum):
    """Types of document segments in legal documents."""

    # Core segment types used in Weaviate
    CONTENT = "content"
    FACTS = "facts"
    LEGAL_BASIS = "legal_basis"
    REASONING = "reasoning"
    CONCLUSION = "conclusion"
    DECISION = "decision"
    INTERPRETATION = "interpretation"
    OTHER = "other"


class DocumentChunk(BaseModel):
    """A chunk of document text for vectorization and search.

    This model matches the Weaviate DocumentChunks collection schema.
    """

    document_id: str
    document_type: Optional[str] = Field(None, description="Type of document")
    language: Optional[str] = Field(None, description="Document language")
    chunk_id: int = Field(description="Numeric chunk identifier")
    chunk_text: str = Field(description="Text content of the chunk")
    segment_type: Optional[str] = Field(None, description="Semantic type of segment")
    position: Optional[int] = Field(None, description="Order in document")
    confidence_score: Optional[float] = Field(
        None, description="Confidence of segment classification"
    )
    cited_references: Optional[str] = Field(None, description="References cited in this chunk")
    tags: Optional[str] = Field(None, description="Custom semantic tags for this chunk")
    parent_segment_id: Optional[str] = Field(None, description="ID of parent segment")
    x: Optional[float] = None
    y: Optional[float] = None


class LegalDocument(BaseModel):
    """Base schema for all legal documents.

    This model matches the Weaviate LegalDocuments collection schema.
    """

    # Core document fields
    document_id: str = Field(description="Unique identifier")
    document_type: str = Field(description="Type of legal document")
    title: Optional[str] = Field(None, description="Document title/name")
    date_issued: Optional[str] = Field(
        None, description="When the document was issued, ISO format date"
    )
    document_number: Optional[str] = Field(None, description="Official reference number")
    language: Optional[str] = Field(None, description="Document language")
    country: Optional[str] = Field(None, description="Country of origin")

    # Content fields
    full_text: Optional[str] = Field(None, description="Raw full text")
    summary: Optional[str] = Field(None, description="Abstract or summary")
    thesis: Optional[str] = Field(None, description="Thesis or main point of the document")
    keywords: Optional[List[str]] = None

    # Issuing body and metadata
    issuing_body: Optional[str] = Field(
        None, description="Information about the body that issued the legal document"
    )
    ingestion_date: Optional[str] = Field(None, description="When document was ingested")
    last_updated: Optional[str] = Field(None, description="When document was last updated")
    processing_status: Optional[str] = Field(None, description="Processing status")
    source_url: Optional[str] = Field(None, description="Source URL of the document")

    # Legal-specific fields
    legal_references: Optional[str] = Field(
        None, description="References to legal acts, regulations, or previous cases"
    )
    parties: Optional[str] = Field(
        None, description="Parties involved in the legal case or interpretation"
    )
    outcome: Optional[str] = Field(None, description="The outcome or result of the legal document")

    # Additional document metadata
    metadata: Optional[str] = Field(None, description="System metadata for the document")
    publication_date: Optional[str] = Field(
        None, description="When the document was published, ISO format date"
    )
    raw_content: Optional[str] = Field(None, description="XML or raw content of the document")

    # Court-specific fields
    presiding_judge: Optional[str] = Field(None, description="Presiding judge information")
    judges: Optional[List[str]] = Field(None, description="All judges involved in the case")
    legal_bases: Optional[List[str]] = Field(None, description="Legal bases for the judgment")
    court_name: Optional[str] = Field(None, description="Name of the court")
    department_name: Optional[str] = Field(None, description="Name of the department")
    extracted_legal_bases: Optional[str] = Field(None, description="Extracted legal bases")
    references: Optional[List[str]] = Field(None, description="References in the document")

    # Position fields for vector embeddings
    x: Optional[float] = None
    y: Optional[float] = None

    @validator("language")
    def validate_language(cls, v):
        """Validate language code format."""
        if v is not None and len(v) not in [
            2,
            5,
        ]:  # ISO 639-1 (2 chars) or locale (5 chars like 'en-US')
            raise ValueError("language must be a valid ISO 639-1 code or locale")
        return v

    @validator("processing_status")
    def validate_processing_status(cls, v):
        """Validate processing status against known values."""
        if v is not None:
            valid_statuses = ["pending", "processing", "completed", "error", "skipped"]
            if v.lower() not in valid_statuses:
                raise ValueError(f'processing_status must be one of: {", ".join(valid_statuses)}')
        return v

    @validator("document_type")
    def validate_document_type(cls, v):
        """Validate document type against DocumentType enum."""
        if v not in [dt.value for dt in DocumentType]:
            raise ValueError(
                f'document_type must be one of: {", ".join([dt.value for dt in DocumentType])}'
            )
        return v

    def to_weaviate_object(self) -> Dict:
        """Convert the document to a format suitable for Weaviate ingestion."""
        data = self.dict(exclude_none=True)
        return data

    @classmethod
    def from_weaviate_object(cls, data: Dict) -> "LegalDocument":
        """Create a LegalDocument from Weaviate data."""
        return cls(**data)

    @classmethod
    def from_judgment(cls, judgment_data: Dict) -> "LegalDocument":
        """Convert a judgment document to a unified LegalDocument."""
        doc_data = {
            "document_id": judgment_data.get("judgment_id", ""),
            "document_type": DocumentType.JUDGMENT.value,
            "document_number": judgment_data.get("docket_number"),
            "date_issued": judgment_data.get("judgment_date"),
        }
        return cls(**doc_data)

    @classmethod
    def from_tax_interpretation(cls, tax_data: Dict) -> "LegalDocument":
        """Convert a tax interpretation document to a unified LegalDocument."""
        doc_data = {
            "document_id": tax_data.get("id", ""),
            "document_type": DocumentType.TAX_INTERPRETATION.value,
        }
        return cls(**doc_data)

    @classmethod
    def from_legal_act(cls, act_data: Dict) -> "LegalDocument":
        """Convert a legal act to a unified LegalDocument."""
        doc_data = {
            "document_id": act_data.get("id", ""),
            "document_type": DocumentType.LEGAL_ACT.value,
            "title": act_data.get("title"),
            "document_number": act_data.get("document_number"),
            "date_issued": act_data.get("enactment_date"),
        }
        return cls(**doc_data)
