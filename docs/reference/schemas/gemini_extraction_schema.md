# Gemini LLM Extraction Schema for Legal Documents

## Overview

This document defines the comprehensive extraction schema for augmenting Weaviate LegalDocuments collection fields using Gemini LLM extraction. The schema maps Weaviate property names to extraction instructions and field specifications.

## Schema Design Principles

1. **Field names match Weaviate schema** - Use exact property names from `LegalDocuments` collection
2. **JSON storage for complex objects** - Complex fields (parties, legal_references, etc.) stored as JSON strings
3. **ISO 8601 dates** - All date fields use ISO format (YYYY-MM-DD)
4. **Language-aware extraction** - Maintain original document language for extracted content
5. **Factual extraction only** - Extract explicitly stated information without interpretation

## Core Document Fields

### Document Identification & Metadata

```python
{
    "document_id": "string, unique identifier extracted from document header or metadata",
    "document_number": "string, official case/document reference number (e.g., 'I ACa 123/24', 'DIS.0101.2.2024')",
    "document_type": "string, type of legal document - must be one of: 'judgment', 'tax_interpretation', 'legal_act'",
    "title": "string, official document title or generated descriptive title",
    "date_issued": "date ISO 8601 (YYYY-MM-DD), date when document was officially issued/published",
    "publication_date": "date ISO 8601 (YYYY-MM-DD), date when document was made public",
    "language": "string, ISO 639-1 language code (e.g., 'pl', 'en')",
    "country": "string, ISO 3166-1 alpha-2 country code (e.g., 'PL', 'CH', 'US')",
}
```

**Extraction Instructions:**

- `document_number`: Look for official case numbers in header (format varies by court/jurisdiction)
- `document_type`: Infer from document structure and terminology (wyrok=judgment, interpretacja podatkowa=tax interpretation)
- `date_issued`: Primary date of document issuance, prefer over hearing/decision dates
- Extract language from document metadata or infer from content

---

## High-Priority LLM Augmentation Fields

### 1. Summary (Priority: HIGH)

```python
{
    "summary": "string, concise 3-5 sentence summary of the legal document covering: (1) document type and issuing body, (2) main legal issue or question, (3) key facts, (4) decision/outcome, (5) primary legal basis"
}
```

**Extraction Instructions:**

- Generate a balanced summary covering all key aspects
- For judgments: Include parties, legal issue, court decision, and reasoning basis
- For tax interpretations: Include applicant, tax question, authority's position, legal basis
- Maintain factual tone without legal conclusions
- Length: 150-300 words maximum
- Use document's original language

**Example (Judgment):**

```
"Wyrok Sądu Okręgowego w Warszawie z dnia 15.01.2024 r., sygn. V C 1234/23.
Sprawa dotyczyła roszczenia konsumenta o zwrot nienależnie pobranych opłat
bankowych za prowadzenie rachunku. Sąd ustalił, że bank pobierał opłaty
niezgodnie z postanowieniami umowy, co stanowiło bezpodstawne wzbogacenie.
Powództwo uwzględniono w całości na podstawie art. 410 k.c. w zw. z art. 405 k.c."
```

---

### 2. Thesis (Priority: HIGH)

```python
{
    "thesis": "string, main legal principle, rule, or holding established by the document; the key takeaway or precedential value"
}
```

**Extraction Instructions:**

- Extract the core legal principle or rule
- For judgments: The precedential legal conclusion or interpretation
- For tax interpretations: The authority's legal position on the tax matter
- Should be 1-3 sentences expressing the legal rule clearly
- Focus on the "what does this document establish" question

**Example:**

```
"Pobieranie przez bank opłat za czynności bankowe niezgodnych z postanowieniami
umowy rachunku bankowego stanowi bezpodstawne wzbogacenie w rozumieniu art. 405
k.c. i obliguje do zwrotu nienależnie pobranych kwot."
```

---

### 3. Keywords (Priority: HIGH)

```python
{
    "keywords": [
        "string, relevant legal keyword or concept",
        "string, area of law",
        "string, specific legal institution or mechanism"
    ]
}
```

**Extraction Instructions:**

- Generate 5-15 relevant keywords covering:
  - Legal domains (prawo cywilne, prawo podatkowe)
  - Legal institutions (bezpodstawne wzbogacenie, interpretacja indywidualna)
  - Specific concepts (umowa kredytu, klauzule abuzywne)
  - Document-specific terms
- Use normalized forms (singular, canonical)
- Prioritize legal terminology over general terms
- Include both broad and specific terms

**Example:**

```json
["prawo cywilne", "bezpodstawne wzbogacenie", "umowa bankowa", "opłaty bankowe",
"roszczenia konsumenckie", "art. 405 k.c.", "ochrona konsumenta"]
```

---

### 4. Outcome (Priority: HIGH)

```python
{
    "outcome": {
        "decision_type": "string, type of decision (e.g., 'uwzględniono', 'oddalono', 'uchylono', 'stanowisko pozytywne', 'stanowisko negatywne')",
        "decision_summary": "string, brief summary of what was decided",
        "awarded_amounts": [
            {
                "type": "string, type of award (e.g., 'kwota główna', 'odsetki', 'koszty')",
                "amount": "number, monetary amount if applicable",
                "currency": "string, currency code (e.g., 'PLN', 'EUR')",
                "recipient": "string, who receives the amount"
            }
        ],
        "legal_effect": "string, legal effect or consequence of the decision"
    }
}
```

**Extraction Instructions:**

- **decision_type**: Classify the outcome (for judgments: granted/dismissed/modified; for interpretations: favorable/unfavorable)
- **decision_summary**: 2-3 sentences explaining what the authority decided
- **awarded_amounts**: Extract all monetary awards, including main amounts, interest, and costs
- **legal_effect**: Describe the practical legal consequence (e.g., "Umowa nieważna", "Zastosowanie stawki 23% VAT")
- Store as JSON string in Weaviate

---

### 5. Legal Concepts (Priority: MEDIUM)

```python
{
    "legal_concepts": [
        {
            "concept_name": "string, name of legal concept (e.g., 'bezpodstawne wzbogacenie', 'force majeure')",
            "legal_area": "string, area of law (e.g., 'prawo cywilne', 'tax law')",
            "definition_context": "string, brief context of how concept is used in this document",
            "relevance": "string, how central this concept is ('primary', 'secondary', 'mentioned')"
        }
    ]
}
```

**Extraction Instructions:**

- Identify all significant legal concepts discussed
- Focus on concepts that are:
  - Central to the legal reasoning
  - Interpreted or defined in the document
  - Part of the legal basis for the decision
- Include both domestic and international legal concepts
- Classify by relevance to the document's core issue
- Store as JSON string in Weaviate

---

### 6. Legal References (Priority: HIGH)

```python
{
    "legal_references": [
        {
            "type": "string, type of reference ('statute', 'article', 'case_law', 'regulation', 'directive', 'treaty')",
            "title": "string, title or name of the legal source",
            "article": "string, specific article/section/paragraph referenced",
            "jurisdiction": "string, jurisdiction (e.g., 'Poland', 'EU', 'Switzerland')",
            "citation": "string, full citation as it appears in document",
            "context": "string, how this reference is used in the document"
        }
    ]
}
```

**Extraction Instructions:**

- Extract ALL legal citations including:
  - Domestic statutes (e.g., "art. 405 k.c.", "ustawa o VAT")
  - EU directives and regulations
  - Court decisions (own or referenced)
  - International treaties
- Parse structured citations to extract components
- Include context: "invoked", "distinguished", "followed", "criticized"
- Normalize common abbreviations (k.c., k.p.c., k.k.)
- Store as JSON string in Weaviate

**Example:**

```json
[
    {
        "type": "statute",
        "title": "Kodeks cywilny",
        "article": "art. 405",
        "jurisdiction": "Poland",
        "citation": "art. 405 k.c.",
        "context": "invoked as legal basis for claim"
    }
]
```

---

### 7. Parties (Priority: MEDIUM)

```python
{
    "parties": [
        {
            "party_type": "string, role in proceeding ('plaintiff', 'defendant', 'appellant', 'appellee', 'applicant', 'third_party')",
            "party_name": "string, name of party (anonymized if necessary)",
            "party_category": "string, category ('natural_person', 'company', 'public_entity', 'ngo')",
            "representation": "string, legal representative info if available",
            "identification": "string, any relevant identifiers (company registration, tax ID)"
        }
    ]
}
```

**Extraction Instructions:**

- For judgments: Extract all parties from case header
- Preserve party anonymization (use initials if document uses them)
- Identify party roles correctly (powód=plaintiff, pozwany=defendant)
- Include legal representatives (attorneys, counsel)
- For tax interpretations: Extract applicant information
- Store as JSON string in Weaviate

---

### 8. Legal Analysis (Priority: MEDIUM)

```python
{
    "legal_analysis": {
        "facts_summary": "string, summary of key factual findings",
        "legal_issues": ["string, main legal question or issue"],
        "reasoning": "string, court's/authority's reasoning process",
        "precedents_applied": ["string, previous cases or interpretations applied"],
        "statutory_interpretation": "string, how statutes were interpreted",
        "conclusion": "string, final legal conclusion"
    }
}
```

**Extraction Instructions:**

- Extract structured analysis from judgment reasoning or interpretation body
- **facts_summary**: Key facts established by court (Stan faktyczny)
- **legal_issues**: Main legal questions to be resolved
- **reasoning**: Logical flow of legal reasoning (Sąd uznał, że...)
- **precedents_applied**: References to previous decisions influencing this one
- **statutory_interpretation**: How laws were interpreted/applied
- Store as JSON string in Weaviate

---

### 9. Structured Content (Priority: MEDIUM)

```python
{
    "structured_content": {
        "sections": [
            {
                "section_type": "string, type ('header', 'facts', 'reasoning', 'decision', 'dissent')",
                "section_title": "string, section heading",
                "content_summary": "string, brief summary of section content",
                "key_points": ["string, key points from this section"]
            }
        ],
        "document_structure": {
            "has_dissent": "boolean, whether document contains dissenting opinion",
            "has_appendices": "boolean, whether document has appendices",
            "page_count": "integer, number of pages"
        }
    }
}
```

**Extraction Instructions:**

- Parse document structure into logical sections
- Identify standard sections (sentencja, uzasadnienie, etc.)
- Extract key points from each major section
- Note presence of special elements (dissenting opinions, appendices)
- Store as JSON string in Weaviate

---

## Document Type-Specific Fields

### Judgment-Specific Fields

```python
{
    "judgment_specific": {
        "court_name": "string, full name of the court",
        "court_type": "string, type of court ('district', 'regional', 'appeal', 'supreme', 'administrative')",
        "department_name": "string, court department or division",
        "case_number": "string, full case signature",
        "hearing_date": "date ISO 8601, date of hearing if different from judgment date",
        "judgment_type": "string, type of judgment ('wyrok', 'postanowienie', 'zarządzenie')",
        "proceeding_type": "string, type of proceeding ('civil', 'criminal', 'administrative')",
        "judges": [
            {
                "name": "string, judge name",
                "role": "string, role ('presiding', 'panel_member', 'rapporteur')"
            }
        ],
        "legal_bases": ["string, legal basis for jurisdiction/decision"],
        "appeal_basis": "string, grounds for appeal if applicable",
        "is_final": "boolean, whether judgment is final/binding"
    }
}
```

**Extraction Instructions:**

- Extract from judgment header (Sąd ... w składzie:)
- **court_name**: Full official court name
- **judges**: Parse judge list with roles (Przewodniczący, Sędziowie)
- **judgment_type**: Distinguish wyrok (judgment) from postanowienie (order)
- **legal_bases**: Articles establishing court jurisdiction
- Store as JSON string in Weaviate

---

### Tax Interpretation-Specific Fields

```python
{
    "tax_interpretation_specific": {
        "interpretation_type": "string, type ('individual', 'general')",
        "tax_authority": "string, issuing tax authority name",
        "authority_level": "string, level ('local', 'regional', 'national')",
        "applicant_type": "string, type of applicant ('natural_person', 'company', 'ngo')",
        "tax_matter": "string, specific tax matter/question",
        "tax_type": "string, type of tax (VAT, CIT, PIT, etc.)",
        "fiscal_year": "string, relevant fiscal year if applicable",
        "interpretation_status": "string, status ('binding', 'non-binding', 'revoked')",
        "validity_period": {
            "start_date": "date ISO 8601",
            "end_date": "date ISO 8601"
        }
    }
}
```

**Extraction Instructions:**

- Extract from interpretation metadata
- **interpretation_type**: Individual (indywidualna) vs general (ogólna)
- **tax_authority**: Extract from document header
- **tax_matter**: Brief description of the tax question addressed
- **interpretation_status**: Whether interpretation is binding on taxpayer
- Store as JSON string in Weaviate

---

## Issuing Body Information

```python
{
    "issuing_body": {
        "name": "string, name of issuing institution (court, tax authority, etc.)",
        "type": "string, type of body ('court', 'tax_authority', 'regulatory_agency')",
        "jurisdiction": "string, territorial jurisdiction",
        "authority_level": "string, hierarchical level ('first_instance', 'appellate', 'supreme')",
        "contact_info": {
            "address": "string, institutional address",
            "website": "string, official website URL"
        }
    }
}
```

---

## Document Relationships

```python
{
    "relationships": [
        {
            "relationship_type": "string, type of relationship ('appeals', 'overturns', 'follows', 'distinguishes', 'cites')",
            "related_document_id": "string, ID of related document",
            "related_document_number": "string, official number of related document",
            "description": "string, description of relationship"
        }
    ]
}
```

**Extraction Instructions:**

- Identify relationships to other documents
- For appeals: Link to lower instance decisions
- For references: Link to cited precedents
- Store as JSON string in Weaviate

---

## Metadata Fields (Auto-Generated)

These fields are typically auto-generated by the system but can be augmented:

```python
{
    "source_url": "string, URL where document was sourced",
    "ingestion_date": "datetime ISO 8601, when document was ingested",
    "last_updated": "datetime ISO 8601, last modification timestamp",
    "processing_status": "string, status ('raw', 'processed', 'enriched', 'verified')",
    "confidence_score": "number (0.0-1.0), confidence in extracted information",
    "metadata": {
        "extractor_model": "string, model used for extraction (e.g., 'gemini-2.5-pro')",
        "extraction_date": "datetime ISO 8601",
        "extraction_version": "string, version of extraction schema",
        "quality_flags": ["string, any quality issues or warnings"]
    }
}
```

---

## Usage Examples

### Example 1: Judgment Extraction Schema

```python
from juddges.extraction import GeminiExtractionChain, ExtractionSchema, DocumentType

schema = ExtractionSchema(
    fields={
        # Core fields
        "document_number": "string, official case number",
        "date_issued": "date ISO 8601, judgment date",
        "court_name": "string, full court name",

        # High-priority augmentation fields
        "summary": "string, 3-5 sentence summary covering parties, issue, decision, and basis",
        "thesis": "string, main legal principle established",
        "keywords": "List[string], 5-15 relevant legal keywords",

        # Outcome
        "outcome": "JSON object with decision_type, decision_summary, awarded_amounts, legal_effect",

        # Legal content
        "legal_references": "JSON array of legal citations with type, title, article, context",
        "legal_concepts": "JSON array of legal concepts with name, area, definition_context",
        "parties": "JSON array of parties with type, name, category, representation",

        # Judgment-specific
        "judgment_specific": "JSON object with court_name, judges, case_number, legal_bases",
    },
    instructions="""
    Focus on extracting factual information from the judgment.
    For dates, use ISO 8601 format (YYYY-MM-DD).
    For complex objects (outcome, parties, legal_references), return valid JSON.
    Maintain original language (Polish) for all extracted content.
    If information is not found, use empty string or empty array.
    For legal references, extract ALL citations including article numbers.
    """,
    language="polish"
)

chain = GeminiExtractionChain(model_name="gemini-2.5-flash")
result = chain.extract(
    document_type=DocumentType.JUDGMENT,
    text=judgment_text,
    schema=schema
)
```

### Example 2: Tax Interpretation Extraction Schema

```python
schema = ExtractionSchema(
    fields={
        "document_number": "string, interpretation reference number",
        "date_issued": "date ISO 8601, interpretation date",
        "tax_authority": "string, issuing authority name",

        "summary": "string, summary of tax question and authority position",
        "thesis": "string, main tax law principle",
        "keywords": "List[string], tax-related keywords",

        "outcome": "JSON object with interpretation decision and legal effect",
        "legal_references": "JSON array of tax law citations",

        "tax_interpretation_specific": "JSON object with interpretation_type, applicant_type, tax_matter, tax_type",
    },
    instructions="""
    Extract tax interpretation content focusing on the legal question and authority's position.
    Identify the specific tax type and relevant tax law provisions.
    """,
    language="polish"
)
```

---

## Field Coverage Analysis

Based on Weaviate schema analysis, these fields have the **highest priority for LLM augmentation** (sorted by emptiness):

| Priority | Field | Coverage | Description |
|----------|-------|----------|-------------|
| 1 | `summary` | 0-10% | Concise document summary |
| 2 | `thesis` | 0-10% | Main legal principle |
| 3 | `keywords` | 0-15% | Relevant keywords/tags |
| 4 | `outcome` | 0-20% | Decision outcome and effects |
| 5 | `legal_concepts` | 0-20% | Legal concepts discussed |
| 6 | `legal_references` | 20-40% | Legal citations and references |
| 7 | `parties` | 30-50% | Party information |
| 8 | `legal_analysis` | 0-10% | Structured legal analysis |
| 9 | `structured_content` | 0-10% | Document structure |
| 10 | `judgment_specific` | 40-60% | Judgment metadata |

---

## Integration with Weaviate

After extraction, results should be stored in Weaviate using the appropriate data types:

```python
from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase
import json

# Extract information
extracted_data = chain.extract(document_type, text, schema)

# Prepare for Weaviate (convert complex objects to JSON strings)
weaviate_properties = {
    "document_id": extracted_data.get("document_id"),
    "summary": extracted_data.get("summary"),
    "thesis": extracted_data.get("thesis"),
    "keywords": extracted_data.get("keywords"),  # Already a list

    # Convert complex objects to JSON strings
    "outcome": json.dumps(extracted_data.get("outcome")) if extracted_data.get("outcome") else None,
    "legal_references": json.dumps(extracted_data.get("legal_references")) if extracted_data.get("legal_references") else None,
    "parties": json.dumps(extracted_data.get("parties")) if extracted_data.get("parties") else None,
    "judgment_specific": json.dumps(extracted_data.get("judgment_specific")) if extracted_data.get("judgment_specific") else None,
}

# Update Weaviate document
with WeaviateLegalDocumentsDatabase() as db:
    collection = db.legal_documents_collection
    collection.data.update(
        uuid=document_uuid,
        properties=weaviate_properties
    )
```

---

## Best Practices

1. **Incremental Extraction**: Start with high-priority fields (summary, thesis, keywords) before extracting complex structures
2. **Validation**: Verify extracted dates are valid ISO 8601, JSON objects are well-formed
3. **Caching**: Use LangChain SQLite cache to avoid re-extracting same documents
4. **Batch Processing**: Use `batch_extract()` for processing multiple documents efficiently
5. **Quality Control**: Review confidence scores and implement human-in-the-loop for low-confidence extractions
6. **Version Tracking**: Store extraction schema version in metadata for reproducibility

---

## See Also

- [Gemini Extraction Chain Implementation](../../juddges/extraction/gemini_chain.py)
- [Weaviate Schema Definition](../../juddges/data/documents_weaviate_db.py)
- [Field Coverage Analysis](../../scripts/embed/analyze_field_coverage.py)
- [Example Extraction Scripts](../../scripts/extraction/)
