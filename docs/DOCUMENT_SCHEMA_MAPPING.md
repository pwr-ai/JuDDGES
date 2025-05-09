# Legal Document Schema Mapping

This document explains the unified schema for legal documents in the JuDDGES, AI-Tax, and other legal-AI@pwr systems. This schema supports multiple document types, including court judgments, tax interpretations, and legal acts.

## Schema Design

The unified schema is designed to:
1. Store standard metadata across all document types
2. Provide type-specific fields for specialized information
3. Support advanced search and retrieval
4. Enable analysis across different document types
5. Allow semantic segmentation and tagging of document content
6. Represent relationships between legal documents
7. Support legislative content like statutes and regulations

## Supported Document Types

The system currently supports the following document types:

| Document Type      | Description                                     | Example                       |
| ------------------ | ----------------------------------------------- | ----------------------------- |
| JUDGMENT           | Court decisions and rulings                     | Supreme Court judgment        |
| TAX_INTERPRETATION | Tax authority interpretations                   | Individual tax interpretation |
| LEGAL_ACT          | Legislative texts (statutes, regulations, etc.) | Civil Code, Tax Ordinance     |

## Field Mapping Table

The table below maps properties between the unified schema, judgment-specific fields, and tax interpretation fields:

| Unified Schema                                   | Judgment Fields    | Tax Interpretation Fields | Legal Act Fields   | Description                              |
| ------------------------------------------------ | ------------------ | ------------------------- | ------------------ | ---------------------------------------- |
| **Common Fields**                                |
| document_id                                      | judgment_id        | id                        | id                 | Unique identifier                        |
| document_type                                    | "judgment"         | "tax_interpretation"      | "legal_act"        | Type of document                         |
| title                                            | -                  | TEZA                      | title              | Document title/name                      |
| date_issued                                      | judgment_date      | DT_WYD                    | enactment_date     | When document was issued                 |
| document_number                                  | docket_number      | SYG                       | document_number    | Official reference number                |
| language                                         | -                  | -                         | language           | Document language                        |
| country                                          | country            | -                         | jurisdiction       | Country of origin                        |
| **Issuing Body**                                 |
| issuing_body.name                                | court_name         | -                         | issuing_body       | Name of issuing institution              |
| issuing_body.jurisdiction                        | -                  | -                         | jurisdiction       | Geographical/legal jurisdiction          |
| issuing_body.type                                | court_type         | -                         | "legislative"      | Type of issuing body                     |
| **Text Content**                                 |
| full_text                                        | full_text          | TRESC_INTERESARIUSZ       | full_text          | Raw document text                        |
| summary                                          | excerpt            | -                         | -                  | Abstract or summary                      |
| **Enhanced Semantic Structure**                  |
| structured_content.sections[]                    | -                  | -                         | -                  | Segmented document content               |
| structured_content.sections[].segment_id         | -                  | -                         | -                  | Unique segment identifier                |
| structured_content.sections[].segment_type       | -                  | -                         | -                  | Semantic type (facts, legal basis, etc.) |
| structured_content.sections[].text               | -                  | -                         | -                  | Segment text content                     |
| structured_content.sections[].position           | -                  | -                         | -                  | Ordering in document                     |
| structured_content.sections[].confidence_score   | -                  | -                         | -                  | Classification confidence                |
| structured_content.sections[].cited_references   | -                  | -                         | -                  | References cited in segment              |
| structured_content.sections[].tags               | -                  | -                         | -                  | Custom semantic tags                     |
| structured_content.sections[].parent_segment_id  | -                  | -                         | -                  | Hierarchical parent ID                   |
| structured_content.sections[].entities           | -                  | -                         | -                  | Named entities in segment                |
| structured_content.sections[].section_heading    | -                  | -                         | -                  | Section heading/title                    |
| structured_content.sections[].number             | -                  | -                         | -                  | Number (article, paragraph)              |
| structured_content.sections[].identifier         | -                  | -                         | -                  | Full identifier with parent elements     |
| structured_content.sections[].is_amended         | -                  | -                         | -                  | Whether segment has been amended         |
| structured_content.semantic_graph                | -                  | -                         | -                  | Relationships between sections           |
| structured_content.segmentation_method           | -                  | -                         | -                  | How document was segmented               |
| **Legal References**                             |
| legal_references[].ref_id                        | -                  | -                         | -                  | Reference identifier                     |
| legal_references[].ref_type                      | -                  | -                         | -                  | Type of reference                        |
| legal_references[].text                          | legal_bases[]      | PRZEPISY[]                | -                  | Reference text                           |
| legal_references[].normalized_citation           | -                  | -                         | -                  | Standardized citation                    |
| legal_references[].target_document_id            | -                  | -                         | -                  | ID of referenced document                |
| **Document Relationships**                       |
| relationships[].source_id                        | -                  | -                         | -                  | Source document ID                       |
| relationships[].target_id                        | -                  | -                         | -                  | Target document ID                       |
| relationships[].relationship_type                | -                  | -                         | -                  | Type of relationship                     |
| relationships[].description                      | -                  | -                         | -                  | Relationship description                 |
| relationships[].context_segment_id               | -                  | -                         | -                  | Segment containing the relationship      |
| **Judgment-Specific**                            |
| judgment_specific.court_level                    | -                  | -                         | -                  | Level of court                           |
| judgment_specific.judges[].name                  | judges[]           | -                         | -                  | Judge names                              |
| judgment_specific.judges[].role                  | presiding_judge    | -                         | -                  | Judge roles                              |
| judgment_specific.procedural_history             | -                  | -                         | -                  | History of the case                      |
| judgment_specific.verdict                        | decision           | -                         | -                  | Final verdict                            |
| judgment_specific.dissenting_opinions            | dissenting_opinion | -                         | -                  | Opposing opinions                        |
| **Tax Interpretation-Specific**                  |
| tax_interpretation_specific.tax_area             | -                  | KATEGORIA_INFORMACJI      | -                  | Tax area covered                         |
| tax_interpretation_specific.tax_provisions       | -                  | PRZEPISY[]                | -                  | Tax provisions referenced                |
| tax_interpretation_specific.taxpayer_type        | -                  | -                         | -                  | Type of taxpayer                         |
| tax_interpretation_specific.interpretation_scope | -                  | -                         | -                  | Scope of interpretation                  |
| **Legal Act-Specific**                           |
| legal_act_specific.act_type                      | -                  | -                         | act_type           | Type of act (statute, regulation)        |
| legal_act_specific.enactment_date                | -                  | -                         | enactment_date     | When the act was enacted                 |
| legal_act_specific.effective_dates               | -                  | -                         | effective_date     | When the act takes/took effect           |
| legal_act_specific.jurisdiction_level            | -                  | -                         | jurisdiction_level | Federal, state, EU level, etc.           |
| legal_act_specific.legislative_body              | -                  | -                         | legislative_body   | Body that enacted the legislation        |
| legal_act_specific.amendment_history             | -                  | -                         | -                  | History of amendments                    |
| legal_act_specific.current_status                | -                  | -                         | current_status     | In force, repealed, amended              |
| legal_act_specific.isap_id                       | -                  | -                         | isap_id            | ID in Polish legal acts system           |
| **Metadata**                                     |
| metadata.ingestion_date                          | -                  | _fetched_at               | -                  | When document was ingested               |
| metadata.last_updated                            | last_update        | -                         | -                  | When document was updated                |
| metadata.source_url                              | -                  | -                         | -                  | Source of document                       |
| **Additional Fields**                            |
| thesis                                           | thesis             | TEZA                      | -                  | Main point of document                   |
| keywords                                         | keywords           | SLOWA_KLUCZOWE[]          | -                  | Keywords                                 |

## Semantic Segment Types

The system supports the following segment types for semantic document chunking:

| Segment Type       | Description                                   | Typically Found In         |
| ------------------ | --------------------------------------------- | -------------------------- |
| FACTS              | Factual background of the case                | Judgments, interpretations |
| LEGAL_BASIS        | Legal provisions and regulations cited        | All document types         |
| REASONING          | Court's or authority's reasoning and analysis | All document types         |
| CONCLUSION         | Final conclusion reached                      | All document types         |
| DECISION           | Official decision or verdict                  | Judgments                  |
| PROCEDURAL_HISTORY | History of the case through courts            | Judgments                  |
| ARGUMENTS          | Arguments presented by parties                | Judgments                  |
| INTERPRETATION     | Specific interpretation of law                | Tax interpretations        |
| DISSENTING_OPINION | Opinion disagreeing with majority             | Judgments                  |
| LEGAL_QUESTION     | Question of law being addressed               | All document types         |
| EVIDENCE           | Evidence presented in the case                | Judgments                  |
| TAX_REGULATION     | Specific tax regulations being addressed      | Tax interpretations        |
| TAXPAYER_POSITION  | Position or argument of the taxpayer          | Tax interpretations        |
| AUTHORITY_POSITION | Position of the tax authority                 | Tax interpretations        |
| ABSTRACT           | Brief summary or abstract                     | All document types         |
| HEADNOTE           | Editorial summary of key points               | Judgments                  |
| PREAMBLE           | Introductory section                          | All document types         |
| TITLE              | Title section of a legal act                  | Legal acts                 |
| CHAPTER            | Chapter subdivision                           | Legal acts                 |
| ARTICLE            | Article subdivision                           | Legal acts                 |
| PARAGRAPH          | Paragraph subdivision                         | Legal acts                 |
| SUBPARAGRAPH       | Subparagraph subdivision                      | Legal acts                 |
| POINT              | Point/item in a list                          | Legal acts                 |
| DEFINITION         | Definition section                            | Legal acts                 |
| PENALTY            | Penalties or sanctions section                | Legal acts                 |
| SCOPE              | Applicability and scope section               | Legal acts                 |
| OTHER              | Other content not fitting above categories    | All document types         |

## Document Relationship Types

The system supports the following relationship types between legal documents:

| Relationship Type | Description                                | Example                                           |
| ----------------- | ------------------------------------------ | ------------------------------------------------- |
| CITES             | Document A cites Document B                | Judgment cites Civil Code                         |
| AMENDS            | Document A amends Document B               | Amendment Act amends Tax Act                      |
| IMPLEMENTS        | Document A implements Document B           | Regulation implements Statute                     |
| INTERPRETS        | Document A interprets Document B           | Tax interpretation interprets Tax Act             |
| APPLIES           | Document A applies Document B              | Judgment applies legal provision                  |
| OVERRULES         | Document A overrules Document B            | Supreme Court judgment overrules earlier judgment |
| FOLLOWS           | Document A follows precedent in Document B | Judgment follows established precedent            |
| CONTRADICTS       | Document A contradicts Document B          | Judgment contradicts earlier ruling               |
| REFERS_TO         | Document A refers to Document B            | General reference without specific context        |
| CONSOLIDATES      | Document A consolidates multiple Documents | Consolidated version of an act                    |
| PARENT_OF         | Document A is the parent act of Document B | Tax Act is parent of Tax Regulation               |

## Example: Legal Act Document

```python
legal_act = LegalDocument(
    document_id="act_2004_11_54_535",
    document_type=DocumentType.LEGAL_ACT,
    title="Ustawa o podatku od towarów i usług",
    document_number="Dz.U. 2004 nr 54 poz. 535",
    date_issued="2004-03-11",
    issuing_body=IssuingBody(
        name="Sejm Rzeczypospolitej Polskiej",
        jurisdiction="Poland",
        type="legislative"
    ),
    legal_act_specific=LegalActSpecific(
        act_type="statute",
        enactment_date="2004-03-11",
        effective_dates=EffectiveDates(
            start_date="2004-05-01",
            end_date=None  # Still in force
        ),
        jurisdiction_level="national",
        legislative_body="Sejm",
        current_status="in force with amendments",
        isap_id="WDU20040540535"
    ),
    structured_content=DocumentStructure(
        sections=[
            TextSegment(
                segment_id="art_1",
                segment_type=SegmentType.ARTICLE,
                text="Art. 1. 1. Ustawa reguluje opodatkowanie podatkiem od towarów i usług...",
                position=1,
                section_heading="Art. 1",
                number="1",
                identifier="art_1"
            ),
            # More sections...
        ]
    ),
    # Relationships to other documents
    relationships=[
        DocumentRelationship(
            source_id="act_2004_11_54_535",
            target_id="act_1997_08_29_Dz_U_137_926",  # Tax Ordinance
            relationship_type=RelationshipType.REFERS_TO,
            description="References to Tax Ordinance procedures"
        )
    ]
)
```

## Example: Document Relationship Query

```python
# Find all tax interpretations of a specific legal act
interpretations = await db.get_document_relationships(
    document_id="act_2004_11_54_535",  # VAT Act
    relationship_type="INTERPRETS",
    as_source=False,
    as_target=True
)

# Find all judgments citing a specific document
citing_judgments = await db.get_citing_documents(
    document_id="act_2004_11_54_535",  # VAT Act
)
```

## Benefits of the Enhanced Schema

1. **Semantic Understanding**: Identify specific parts of documents by their legal function
2. **Fine-grained Search**: Search within specific document segments rather than whole documents
3. **Targeted Retrieval**: Return precisely the relevant segments in response to queries
4. **Cross-Document Analysis**: Compare similar segments across different document types
5. **Hierarchical Structure**: Preserve the document's hierarchical organization
6. **Rich Metadata**: Track confidence scores and semantic relationships
7. **Context Preservation**: Understand how segments relate to each other
8. **Custom Tagging**: Apply domain-specific tags for specialized filtering
9. **Document Relationships**: Model the complex web of relationships between legal documents
10. **Legislative Support**: Full support for legal acts with their unique structure and lifecycle 