import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


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
    """Base schema for all legal documents.
    
    Schema version: 2.0.0
    - v1.0.0: Initial schema with core fields
    - v2.0.0: Added Weaviate-specific fields, datetime converters, and typed accessors
    """

    # Schema versioning
    schema_version: str = Field(default="2.0.0", description="Schema version for compatibility tracking")
    
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
    
    # Additional fields from Weaviate schema
    source_id: Optional[str] = Field(None, description="Court ID or source identifier")
    judgment_type: Optional[str] = Field(None, description="Type of judgment")
    raw_content: Optional[str] = Field(None, description="XML or raw content of the document")
    presiding_judge: Optional[str] = Field(None, description="Presiding judge information")
    judges: Optional[str] = Field(None, description="All judges involved in the case")
    legal_bases: Optional[str] = Field(None, description="Legal bases for the judgment")
    publisher: Optional[str] = Field(None, description="Publisher of the document")
    recorder: Optional[str] = Field(None, description="Recorder of the document")
    reviser: Optional[str] = Field(None, description="Reviser of the document")
    num_pages: Optional[int] = Field(None, description="Number of pages in the document")
    volume_number: Optional[str] = Field(None, description="Volume number")
    volume_type: Optional[str] = Field(None, description="Volume type")
    court_name: Optional[str] = Field(None, description="Name of the court")
    department_name: Optional[str] = Field(None, description="Name of the department")
    extracted_legal_bases: Optional[str] = Field(None, description="Extracted legal bases")
    references: Optional[str] = Field(None, description="References in the document")
    court_type: Optional[str] = Field(None, description="Type of court")

    @validator('num_pages')
    def validate_num_pages(cls, v):
        """Validate that num_pages is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError('num_pages must be positive')
        return v

    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        """Validate that confidence_score is between 0 and 1 if provided."""
        if v is not None and not (0 <= v <= 1):
            raise ValueError('confidence_score must be between 0 and 1')
        return v

    @validator('language')
    def validate_language(cls, v):
        """Validate language code format."""
        if v is not None and len(v) not in [2, 5]:  # ISO 639-1 (2 chars) or locale (5 chars like 'en-US')
            raise ValueError('language must be a valid ISO 639-1 code or locale')
        return v

    @validator('court_type')
    def validate_court_type(cls, v):
        """Validate court type against known values."""
        if v is not None:
            valid_types = ['supreme', 'appellate', 'district', 'administrative', 'constitutional', 'tax', 'regional', 'local']
            if v.lower() not in valid_types:
                raise ValueError(f'court_type must be one of: {", ".join(valid_types)}')
        return v

    @validator('processing_status')
    def validate_processing_status(cls, v):
        """Validate processing status against known values."""
        if v is not None:
            valid_statuses = ['pending', 'processing', 'completed', 'error', 'skipped']
            if v.lower() not in valid_statuses:
                raise ValueError(f'processing_status must be one of: {", ".join(valid_statuses)}')
        return v

    @validator('document_type')
    def validate_document_type(cls, v):
        """Validate document type against DocumentType enum."""
        if v not in [dt.value for dt in DocumentType]:
            raise ValueError(f'document_type must be one of: {", ".join([dt.value for dt in DocumentType])}')
        return v

    def to_weaviate_object(self) -> Dict:
        """Convert the document to a format suitable for Weaviate ingestion."""
        data = self.dict(exclude_none=True)
        
        # Convert datetime objects to ISO format strings for Weaviate
        for field_name in ['date_issued', 'publication_date']:
            if field_name in data and isinstance(data[field_name], datetime):
                data[field_name] = data[field_name].isoformat()
        
        return data

    @classmethod
    def from_weaviate_object(cls, data: Dict) -> "LegalDocument":
        """Create a LegalDocument from Weaviate data."""
        # Convert ISO strings back to datetime objects
        for field_name in ['date_issued', 'publication_date']:
            if field_name in data and isinstance(data[field_name], str):
                try:
                    data[field_name] = datetime.fromisoformat(data[field_name])
                except (ValueError, TypeError):
                    # If conversion fails, keep as string or set to None
                    data[field_name] = None
        
        return cls(**data)

    # Typed accessors for JSON-serialized complex objects
    def get_issuing_body(self) -> Optional[IssuingBody]:
        """Get the issuing body as a typed object."""
        if self.issuing_body:
            try:
                data = json.loads(self.issuing_body)
                return IssuingBody(**data)
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        return None

    def set_issuing_body(self, issuing_body: IssuingBody) -> None:
        """Set the issuing body from a typed object."""
        self.issuing_body = json.dumps(issuing_body.dict())

    def get_legal_references(self) -> Optional[List[LegalReference]]:
        """Get legal references as a list of typed objects."""
        if self.legal_references:
            try:
                data = json.loads(self.legal_references)
                if isinstance(data, list):
                    return [LegalReference(**ref) for ref in data]
                return None
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        return None

    def set_legal_references(self, references: List[LegalReference]) -> None:
        """Set legal references from a list of typed objects."""
        self.legal_references = json.dumps([ref.dict() for ref in references])

    def get_legal_concepts(self) -> Optional[List[LegalConcept]]:
        """Get legal concepts as a list of typed objects."""
        if self.legal_concepts:
            try:
                data = json.loads(self.legal_concepts)
                if isinstance(data, list):
                    return [LegalConcept(**concept) for concept in data]
                return None
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        return None

    def set_legal_concepts(self, concepts: List[LegalConcept]) -> None:
        """Set legal concepts from a list of typed objects."""
        self.legal_concepts = json.dumps([concept.dict() for concept in concepts])

    def get_parties(self) -> Optional[List[Party]]:
        """Get parties as a list of typed objects."""
        if self.parties:
            try:
                data = json.loads(self.parties)
                if isinstance(data, list):
                    return [Party(**party) for party in data]
                return None
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        return None

    def set_parties(self, parties: List[Party]) -> None:
        """Set parties from a list of typed objects."""
        self.parties = json.dumps([party.dict() for party in parties])

    def get_outcome(self) -> Optional[Outcome]:
        """Get outcome as a typed object."""
        if self.outcome:
            try:
                data = json.loads(self.outcome)
                return Outcome(**data)
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        return None

    def set_outcome(self, outcome: Outcome) -> None:
        """Set outcome from a typed object."""
        self.outcome = json.dumps(outcome.dict())

    def get_judgment_specific(self) -> Optional[JudgmentSpecific]:
        """Get judgment-specific fields as a typed object."""
        if self.judgment_specific:
            try:
                data = json.loads(self.judgment_specific)
                return JudgmentSpecific(**data)
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        return None

    def set_judgment_specific(self, judgment_specific: JudgmentSpecific) -> None:
        """Set judgment-specific fields from a typed object."""
        self.judgment_specific = json.dumps(judgment_specific.dict())

    def get_tax_interpretation_specific(self) -> Optional[TaxInterpretationSpecific]:
        """Get tax interpretation-specific fields as a typed object."""
        if self.tax_interpretation_specific:
            try:
                data = json.loads(self.tax_interpretation_specific)
                return TaxInterpretationSpecific(**data)
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        return None

    def set_tax_interpretation_specific(self, tax_specific: TaxInterpretationSpecific) -> None:
        """Set tax interpretation-specific fields from a typed object."""
        self.tax_interpretation_specific = json.dumps(tax_specific.dict())

    def get_legal_act_specific(self) -> Optional[LegalActSpecific]:
        """Get legal act-specific fields as a typed object."""
        if self.legal_act_specific:
            try:
                data = json.loads(self.legal_act_specific)
                return LegalActSpecific(**data)
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        return None

    def set_legal_act_specific(self, legal_act_specific: LegalActSpecific) -> None:
        """Set legal act-specific fields from a typed object."""
        self.legal_act_specific = json.dumps(legal_act_specific.dict())

    def get_structured_content(self) -> Optional[DocumentStructure]:
        """Get structured content as a typed object."""
        if self.structured_content:
            try:
                data = json.loads(self.structured_content)
                return DocumentStructure(**data)
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        return None

    def set_structured_content(self, structured_content: DocumentStructure) -> None:
        """Set structured content from a typed object."""
        self.structured_content = json.dumps(structured_content.dict())

    def get_metadata(self) -> Optional[DocumentMetadata]:
        """Get metadata as a typed object."""
        if self.metadata:
            try:
                data = json.loads(self.metadata)
                return DocumentMetadata(**data)
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        return None

    def set_metadata(self, metadata: DocumentMetadata) -> None:
        """Set metadata from a typed object."""
        self.metadata = json.dumps(metadata.dict())

    @classmethod
    def migrate_from_v1(cls, v1_data: Dict) -> "LegalDocument":
        """Migrate from schema version 1.0.0 to 2.0.0."""
        # Add default values for new fields
        v2_data = v1_data.copy()
        v2_data['schema_version'] = '2.0.0'
        
        # Map any renamed fields if necessary
        # (In this case, no field renaming occurred, just additions)
        
        return cls(**v2_data)

    def is_compatible_with_version(self, target_version: str) -> bool:
        """Check if this document is compatible with a target schema version."""
        current_major = int(self.schema_version.split('.')[0])
        target_major = int(target_version.split('.')[0])
        
        # Major version compatibility: same major version
        return current_major == target_major

    @classmethod
    def from_judgment(cls, judgment_data: Dict) -> "LegalDocument":
        """Convert a judgment document to a unified LegalDocument."""
        # Implementation would map from judgment-specific fields to the unified schema
        doc_data = {
            "document_id": judgment_data.get("judgment_id", ""),
            "document_type": DocumentType.JUDGMENT,
            "document_number": judgment_data.get("docket_number"),
            "date_issued": judgment_data.get("judgment_date"),
            # Map other fields accordingly
        }
        return cls(**doc_data)

    @classmethod
    def from_tax_interpretation(cls, tax_data: Dict) -> "LegalDocument":
        """Convert a tax interpretation document to a unified LegalDocument."""
        # Implementation would map from tax interpretation fields to the unified schema
        doc_data = {
            "document_id": tax_data.get("id", ""),
            "document_type": DocumentType.TAX_INTERPRETATION,
            # Example mapping for fields in the example tax interpretation
            "title": next(
                (
                    field["value"]
                    for field in tax_data.get("dokument", {}).get("fields", [])
                    if field.get("key") == "TEZA"
                ),
                None,
            ),
            "date_issued": next(
                (
                    field["value"]
                    for field in tax_data.get("dokument", {}).get("fields", [])
                    if field.get("key") == "DT_WYD"
                ),
                None,
            ),
            "document_number": next(
                (
                    field["value"]
                    for field in tax_data.get("dokument", {}).get("fields", [])
                    if field.get("key") == "SYG"
                ),
                None,
            ),
        }

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
            doc_data["full_text"] = full_text

        return cls(**doc_data)

    @classmethod
    def from_legal_act(cls, act_data: Dict) -> "LegalDocument":
        """Convert a legal act to a unified LegalDocument."""
        # This is a simplified implementation
        doc_data = {
            "document_id": act_data.get("id", ""),
            "document_type": DocumentType.LEGAL_ACT,
            "title": act_data.get("title"),
            "document_number": act_data.get("document_number"),
            "date_issued": act_data.get("enactment_date"),
        }

        # Create legal act specific data as JSON string
        legal_act_specific_data = {
            "act_type": act_data.get("act_type", ""),
        }
        
        # Add optional fields if they exist
        for field in ["enactment_date", "legislative_body", "jurisdiction_level", "current_status", "isap_id"]:
            if act_data.get(field):
                legal_act_specific_data[field] = act_data[field]
        
        # Add effective dates if available
        if act_data.get("effective_date"):
            legal_act_specific_data["effective_dates"] = EffectiveDates(
                start_date=act_data.get("effective_date"),
                end_date=act_data.get("expiration_date"),
            ).dict()
        
        legal_act_specific = LegalActSpecific(**legal_act_specific_data)
        doc_data["legal_act_specific"] = json.dumps(legal_act_specific.dict())

        # Add issuing body information if available
        if act_data.get("issuing_body"):
            issuing_body = IssuingBody(
                name=act_data.get("issuing_body", ""),
                type="legislative",
                jurisdiction=act_data.get("jurisdiction"),
            )
            doc_data["issuing_body"] = json.dumps(issuing_body.dict())

        return cls(**doc_data)


class DocumentQuery:
    """Query methods for legal documents."""

    @staticmethod
    def find_by_segment_type(
        document: LegalDocument, segment_type: SegmentType
    ) -> List[TextSegment]:
        """Return all segments of specified type from the document."""
        structured_content = document.get_structured_content()
        if not structured_content or not structured_content.sections:
            return []
        return [s for s in structured_content.sections if s.segment_type == segment_type]

    @staticmethod
    def find_related_documents(
        document: LegalDocument, relationship_type: Optional[RelationshipType] = None
    ) -> List[str]:
        """Find IDs of documents related to this document, optionally filtered by relationship type."""
        if not document.relationships:
            return []

        try:
            relationships_data = json.loads(document.relationships)
            if not isinstance(relationships_data, list):
                return []
            
            relationships = [DocumentRelationship(**rel) for rel in relationships_data]
        except (json.JSONDecodeError, TypeError, ValueError):
            return []

        if relationship_type:
            return [
                rel.target_id
                for rel in relationships
                if rel.relationship_type == relationship_type
            ]

        return [rel.target_id for rel in relationships]

    @staticmethod
    def find_citing_legal_acts(document: LegalDocument) -> List[str]:
        """Find legal acts cited in this document."""
        return DocumentQuery.find_related_documents(document, RelationshipType.CITES)
