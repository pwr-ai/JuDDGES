from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Enumeration of supported legal document types."""

    JUDGMENT = "judgment"
    TAX_INTERPRETATION = "tax_interpretation"
    LEGAL_ACT = "legal_act"  # New type for legislation


class SegmentType(str, Enum):
    """Types of document segments in legal documents."""

    # Common segment types
    FACTS = "facts"
    LEGAL_BASIS = "legal_basis"
    REASONING = "reasoning"
    CONCLUSION = "conclusion"
    DECISION = "decision"
    PROCEDURAL_HISTORY = "procedural_history"
    ARGUMENTS = "arguments"
    INTERPRETATION = "interpretation"
    DISSENTING_OPINION = "dissenting_opinion"
    LEGAL_QUESTION = "legal_question"
    EVIDENCE = "evidence"
    TAX_REGULATION = "tax_regulation"
    TAXPAYER_POSITION = "taxpayer_position"
    AUTHORITY_POSITION = "authority_position"
    ABSTRACT = "abstract"
    HEADNOTE = "headnote"
    PREAMBLE = "preamble"
    OTHER = "other"

    # Legal act specific segment types
    TITLE = "title"
    CHAPTER = "chapter"
    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    SUBPARAGRAPH = "subparagraph"
    POINT = "point"
    DEFINITION = "definition"
    PENALTY = "penalty"
    SCOPE = "scope"


class RelationshipType(str, Enum):
    """Types of relationships between legal documents."""

    CITES = "cites"  # Document A cites Document B
    AMENDS = "amends"  # Document A amends Document B
    IMPLEMENTS = "implements"  # Document A implements Document B
    INTERPRETS = "interprets"  # Document A interprets Document B
    APPLIES = "applies"  # Document A applies Document B
    OVERRULES = "overrules"  # Document A overrules Document B
    FOLLOWS = "follows"  # Document A follows precedent in Document B
    CONTRADICTS = "contradicts"  # Document A contradicts Document B
    REFERS_TO = "refers_to"  # Document A refers to Document B
    CONSOLIDATES = "consolidates"  # Document A consolidates multiple Documents
    PARENT_OF = "parent_of"  # Document A is the parent act of Document B


class IssuingBody(BaseModel):
    """Information about the body that issued the legal document."""

    name: str = Field(description="Name of issuing institution (court, tax authority, etc.)")
    jurisdiction: Optional[str] = Field(None, description="Geographical/legal jurisdiction")
    type: str = Field(description="Type of issuing body (e.g., 'court', 'tax authority')")


class LegalReference(BaseModel):
    """A reference to a legal act, regulation, or previous case."""

    ref_id: Optional[str] = None
    ref_type: str = Field(description="Type of reference (e.g., statute, regulation, precedent)")
    text: str
    normalized_citation: Optional[str] = Field(None, description="Standardized citation format")
    target_document_id: Optional[str] = Field(
        None, description="ID of the referenced document if available"
    )
    target_segment_id: Optional[str] = Field(
        None, description="ID of the specific segment being referenced"
    )
    address: Optional[str] = Field(
        None,
        description="Official publication reference (e.g., 'Dz. U. z 1964 r. Nr 43, poz. 296')",
    )
    art: Optional[str] = Field(None, description="Article references (e.g., 'art. 98;art. 99')")
    isap_id: Optional[str] = Field(None, description="ISAP identifier for Polish legal acts")
    title: Optional[str] = Field(None, description="Full title of the referenced legal act")


class LegalConcept(BaseModel):
    """A legal concept or topic discussed in the document."""

    concept_name: str
    concept_type: Optional[str] = None
    relevance_score: Optional[float] = Field(
        None, description="How relevant this concept is to the document"
    )


class Party(BaseModel):
    """A party involved in the legal case or interpretation."""

    party_id: Optional[str] = None
    party_type: str = Field(description="e.g., plaintiff, defendant, taxpayer, government")
    name: Optional[str] = Field(None, description="Anonymized if needed")
    role: Optional[str] = None


class Outcome(BaseModel):
    """The outcome or result of the legal document."""

    decision_type: str = Field(description="e.g., ruling, interpretation, opinion")
    decision_summary: str = Field(description="Short description of outcome")
    winning_party: Optional[str] = Field(None, description="If applicable")


class Judge(BaseModel):
    """Information about a judge involved in a judgment."""

    name: str
    role: Optional[str] = Field(None, description="e.g., chief judge, dissenting")


class DissentingOpinion(BaseModel):
    """A dissenting opinion in a judgment."""

    judge: str
    text: str


class JudgmentSpecific(BaseModel):
    """Fields specific to judgment documents."""

    court_level: Optional[str] = Field(None, description="e.g., supreme, appellate, district")
    judges: Optional[List[Judge]] = None
    procedural_history: Optional[str] = None
    verdict: Optional[str] = None
    relief_granted: Optional[str] = None
    dissenting_opinions: Optional[List[DissentingOpinion]] = None
    judgment_type: Optional[str] = None


class TaxProvision(BaseModel):
    """A tax provision referenced in a tax interpretation."""

    provision_id: str
    provision_text: str
    code_section: Optional[str] = None


class EffectiveDates(BaseModel):
    """Effective dates for a tax interpretation or legal act."""

    start_date: Optional[str] = None  # ISO format date
    end_date: Optional[str] = None  # ISO format date, might be null if ongoing


class TaxInterpretationSpecific(BaseModel):
    """Fields specific to tax interpretation documents."""

    tax_area: Optional[str] = Field(None, description="e.g., income tax, VAT, corporate tax")
    tax_provisions: Optional[List[TaxProvision]] = None
    taxpayer_type: Optional[str] = Field(
        None, description="e.g., individual, corporation, partnership"
    )
    interpretation_scope: Optional[str] = Field(
        None, description="e.g., general, specific to scenario"
    )
    effective_dates: Optional[EffectiveDates] = None


class AmendmentInfo(BaseModel):
    """Information about an amendment to a legal act."""

    amendment_date: str  # ISO format date
    amending_act_id: Optional[str] = None
    description: str
    affected_segments: Optional[List[str]] = None  # IDs of affected segments


class LegalActSpecific(BaseModel):
    """Fields specific to legal acts (statutes, regulations, etc.)."""

    act_type: str = Field(description="Type of act (e.g., statute, regulation, directive)")
    enactment_date: Optional[str] = None
    effective_dates: Optional[EffectiveDates] = None
    jurisdiction_level: Optional[str] = Field(None, description="e.g., federal, state, EU")
    legislative_body: Optional[str] = None
    amendment_history: Optional[List[AmendmentInfo]] = None
    current_status: Optional[str] = Field(None, description="e.g., in force, repealed, amended")
    official_publication: Optional[str] = Field(
        None, description="Official publication information"
    )
    isap_id: Optional[str] = Field(None, description="ID in the Polish legal acts system")
    parent_act_id: Optional[str] = Field(
        None, description="ID of the parent act if this is subordinate legislation"
    )
    codification: Optional[str] = Field(
        None, description="Codification information (e.g., 'Civil Code')"
    )


class KeyFinding(BaseModel):
    """A key finding from the legal analysis."""

    finding_id: str
    text: str
    importance: Optional[int] = Field(None, description="1-5 scale")


class ReasoningPattern(BaseModel):
    """A reasoning pattern used in the legal analysis."""

    pattern_type: str = Field(description="e.g., statutory interpretation, precedent application")
    text: str


class PolicyConsideration(BaseModel):
    """A policy consideration mentioned in the legal analysis."""

    consideration_type: str
    text: str


class LegalAnalysis(BaseModel):
    """Analysis elements common across document types."""

    key_findings: Optional[List[KeyFinding]] = None
    reasoning_patterns: Optional[List[ReasoningPattern]] = None
    policy_considerations: Optional[List[PolicyConsideration]] = None


class DocumentMetadata(BaseModel):
    """System metadata for the document."""

    ingestion_date: Optional[str] = None  # ISO format datetime
    last_updated: Optional[str] = None  # ISO format datetime
    processing_status: Optional[str] = None
    source_url: Optional[str] = None
    license: Optional[str] = None
    confidence_score: Optional[float] = Field(None, description="For data extracted via ML")


class EntityMention(BaseModel):
    """A named entity mentioned in the text."""

    entity_type: str = Field(description="Type of entity (e.g., person, organization, statute)")
    text: str = Field(description="Text of the entity mention")
    start_char: Optional[int] = Field(None, description="Start character index in text")
    end_char: Optional[int] = Field(None, description="End character index in text")
    normalized_value: Optional[str] = Field(None, description="Normalized/canonical form of entity")
    confidence: Optional[float] = Field(None, description="Confidence score for entity extraction")


class TextSegment(BaseModel):
    """A semantic segment of document text."""

    segment_id: str
    segment_type: SegmentType = Field(description="Type of segment (facts, reasoning, etc.)")
    text: str
    position: int = Field(description="Order in document")

    # Enhanced features for better semantic chunking
    confidence_score: Optional[float] = Field(
        None, description="Confidence of segment classification"
    )
    cited_references: Optional[List[str]] = Field(
        None, description="References cited in this segment"
    )
    tags: Optional[List[str]] = Field(None, description="Custom semantic tags for this segment")
    parent_segment_id: Optional[str] = Field(
        None, description="ID of parent segment if hierarchical"
    )
    start_char_index: Optional[int] = Field(
        None, description="Start character index in full document"
    )
    end_char_index: Optional[int] = Field(None, description="End character index in full document")
    entities: Optional[List[EntityMention]] = Field(
        None, description="Named entities found in segment"
    )
    section_heading: Optional[str] = Field(
        None, description="Heading or title of this section if any"
    )

    # Legal act specific fields
    number: Optional[str] = Field(None, description="Number for articles, paragraphs, etc.")
    identifier: Optional[str] = Field(None, description="Full identifier including parent elements")
    is_amended: Optional[bool] = Field(None, description="Whether this segment has been amended")
    amendment_info: Optional[Dict[str, Any]] = Field(None, description="Details about amendments")


class DocumentStructure(BaseModel):
    """Structured representation of document content."""

    sections: List[TextSegment]
    semantic_graph: Optional[Dict[str, Any]] = Field(
        None, description="Relationships between sections"
    )

    # Information about how document was segmented
    segmentation_method: Optional[str] = Field(None, description="Method used for segmentation")
    segmentation_model: Optional[str] = Field(
        None, description="Model used for semantic segmentation"
    )
    segmentation_date: Optional[str] = Field(None, description="When segmentation was performed")


class DocumentRelationship(BaseModel):
    """A relationship between two legal documents."""

    source_id: str = Field(description="ID of the source document")
    target_id: str = Field(description="ID of the target document")
    relationship_type: RelationshipType = Field(description="Type of relationship")
    description: Optional[str] = Field(None, description="Description of the relationship")
    context_segment_id: Optional[str] = Field(
        None, description="ID of the segment containing the relationship"
    )
    confidence_score: Optional[float] = Field(
        None, description="Confidence score for the relationship"
    )
    bidirectional: bool = Field(
        False, description="Whether the relationship applies in both directions"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata about the relationship"
    )
    creation_date: Optional[str] = Field(None, description="When the relationship was created")


class DocumentChunk(BaseModel):
    """A chunk of document text for vectorization and search."""

    document_id: str
    document_type: DocumentType
    chunk_id: int
    chunk_text: str
    date_issued: Optional[datetime] = Field(None, description="Date of the document")
    publication_date: Optional[datetime] = Field(None, description="Date of publication")
    segment_type: Optional[str] = Field(None, description="Semantic type of segment")
    position: Optional[int] = Field(None, description="Order in document")
    confidence_score: Optional[float] = Field(
        None, description="Confidence of segment classification"
    )
    cited_references: Optional[str] = Field(
        None, description="JSON string: References cited in this chunk"
    )
    tags: Optional[str] = Field(
        None, description="JSON string: Custom semantic tags for this chunk"
    )
    parent_segment_id: Optional[str] = Field(None, description="ID of parent segment")
    x: Optional[float] = None
    y: Optional[float] = None


class LegalDocument(BaseModel):
    """Base schema for all legal documents."""

    document_id: str = Field(description="Unique identifier")
    document_type: str = Field(description="Type of legal document")
    title: Optional[str] = Field(None, description="Document title/name")
    date_issued: Optional[datetime] = Field(
        None, description="When the document was published, ISO format date"
    )
    publication_date: Optional[datetime] = Field(
        None, description="When the document was published, ISO format date"
    )
    issuing_body: Optional[str] = Field(
        None, description="JSON string: Information about the body that issued the legal document"
    )
    language: Optional[str] = Field(None, description="Document language")
    document_number: Optional[str] = Field(None, description="Official reference number")
    country: Optional[str] = Field(None, description="Country of origin")
    full_text: Optional[str] = Field(None, description="Raw full text")
    summary: Optional[str] = Field(None, description="Abstract or summary")
    thesis: Optional[str] = Field(None, description="Thesis or main point of the document")
    keywords: Optional[List[str]] = None
    x: Optional[float] = None
    y: Optional[float] = None
    ingestion_date: Optional[str] = None
    last_updated: Optional[str] = None
    processing_status: Optional[str] = None
    source_url: Optional[str] = None
    confidence_score: Optional[float] = None
    legal_references: Optional[str] = Field(
        None, description="JSON string: References to legal acts, regulations, or previous cases"
    )
    legal_concepts: Optional[str] = Field(
        None, description="JSON string: Legal concepts or topics discussed in the document"
    )
    parties: Optional[str] = Field(
        None, description="JSON string: Parties involved in the legal case or interpretation"
    )
    outcome: Optional[str] = Field(
        None, description="JSON string: The outcome or result of the legal document"
    )
    judgment_specific: Optional[str] = Field(
        None, description="JSON string: Fields specific to judgment documents"
    )
    tax_interpretation_specific: Optional[str] = Field(
        None, description="JSON string: Fields specific to tax interpretation documents"
    )
    legal_act_specific: Optional[str] = Field(
        None, description="JSON string: Fields specific to legal acts (statutes, regulations, etc.)"
    )
    relationships: Optional[str] = Field(
        None, description="JSON string: Relationships to other documents"
    )
    legal_analysis: Optional[str] = Field(
        None, description="JSON string: Analysis elements common across document types"
    )
    structured_content: Optional[str] = Field(
        None, description="JSON string: Structured representation of document content"
    )
    section_embeddings: Optional[str] = Field(
        None, description="JSON string: Vector embeddings for each section for semantic search"
    )
    metadata: Optional[str] = Field(
        None, description="JSON string: System metadata for the document"
    )

    def to_weaviate_object(self) -> Dict:
        """Convert the document to a format suitable for Weaviate ingestion."""
        # This is a simplified version - in practice, this would need to handle
        # nested objects and custom transformations
        return self.dict(exclude_none=True)

    @classmethod
    def from_judgment(cls, judgment_data: Dict) -> "LegalDocument":
        """Convert a judgment document to a unified LegalDocument."""
        # Implementation would map from judgment-specific fields to the unified schema
        doc = LegalDocument(
            document_id=judgment_data.get("judgment_id", ""),
            document_type=DocumentType.JUDGMENT,
            document_number=judgment_data.get("docket_number"),
            date_issued=judgment_data.get("judgment_date"),
            # Map other fields accordingly
        )
        return doc

    @classmethod
    def from_tax_interpretation(cls, tax_data: Dict) -> "LegalDocument":
        """Convert a tax interpretation document to a unified LegalDocument."""
        # Implementation would map from tax interpretation fields to the unified schema
        doc = LegalDocument(
            document_id=tax_data.get("id", ""),
            document_type=DocumentType.TAX_INTERPRETATION,
            # Example mapping for fields in the example tax interpretation
            title=next(
                (
                    field["value"]
                    for field in tax_data.get("dokument", {}).get("fields", [])
                    if field.get("key") == "TEZA"
                ),
                None,
            ),
            date_issued=next(
                (
                    field["value"]
                    for field in tax_data.get("dokument", {}).get("fields", [])
                    if field.get("key") == "DT_WYD"
                ),
                None,
            ),
            document_number=next(
                (
                    field["value"]
                    for field in tax_data.get("dokument", {}).get("fields", [])
                    if field.get("key") == "SYG"
                ),
                None,
            ),
            # Map other fields accordingly
        )

        # Extract full text from the "TRESC_INTERESARIUSZ" field
        full_text = next(
            (
                field["value"]
                for field in tax_data.get("dokument", {}).get("fields", [])
                if field.get("key") == "TRESC_INTERESARIUSZ"
            ),
            None,
        )
        if full_text:
            doc.full_text = full_text

        return doc

    @classmethod
    def from_legal_act(cls, act_data: Dict) -> "LegalDocument":
        """Convert a legal act to a unified LegalDocument."""
        # This is a simplified implementation
        doc = LegalDocument(
            document_id=act_data.get("id", ""),
            document_type=DocumentType.LEGAL_ACT,
            title=act_data.get("title"),
            document_number=act_data.get("document_number"),
            date_issued=act_data.get("enactment_date"),
            legal_act_specific=LegalActSpecific(
                act_type=act_data.get("act_type", ""),
                enactment_date=act_data.get("enactment_date"),
                legislative_body=act_data.get("legislative_body"),
                jurisdiction_level=act_data.get("jurisdiction_level"),
                current_status=act_data.get("current_status"),
                effective_dates=EffectiveDates(
                    start_date=act_data.get("effective_date"),
                    end_date=act_data.get("expiration_date"),
                )
                if act_data.get("effective_date")
                else None,
                isap_id=act_data.get("isap_id"),
            ),
        )

        # Add issuing body information if available
        if act_data.get("issuing_body"):
            doc.issuing_body = IssuingBody(
                name=act_data.get("issuing_body"),
                type="legislative",
                jurisdiction=act_data.get("jurisdiction"),
            )

        return doc


class DocumentQuery:
    """Query methods for legal documents."""

    @staticmethod
    def find_by_segment_type(
        document: LegalDocument, segment_type: SegmentType
    ) -> List[TextSegment]:
        """Return all segments of specified type from the document."""
        if not document.structured_content or not document.structured_content.sections:
            return []
        return [s for s in document.structured_content.sections if s.segment_type == segment_type]

    @staticmethod
    def find_related_documents(
        document: LegalDocument, relationship_type: Optional[RelationshipType] = None
    ) -> List[str]:
        """Find IDs of documents related to this document, optionally filtered by relationship type."""
        if not document.relationships:
            return []

        if relationship_type:
            return [
                rel.target_id
                for rel in document.relationships
                if rel.relationship_type == relationship_type
            ]

        return [rel.target_id for rel in document.relationships]

    @staticmethod
    def find_citing_legal_acts(document: LegalDocument) -> List[str]:
        """Find legal acts cited in this document."""
        return DocumentQuery.find_related_documents(document, RelationshipType.CITES)
