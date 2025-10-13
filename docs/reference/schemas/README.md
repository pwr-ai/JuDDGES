# Extraction Schema Reference

This directory contains comprehensive validation and specification documents for the JuDDGES extraction schema used for Polish legal documents.

## Overview

The extraction schema defines the structured fields that the LLM extraction pipeline identifies and extracts from legal documents (court judgments and tax interpretations).

**Schema Implementation**: [`scripts/extraction/run_extraction_rest.py`](../../../scripts/extraction/run_extraction_rest.py)

**Last Updated**: 2025-10-11

---

## Documents in This Directory

### Main Reference

- **[schema_improvement_summary.md](schema_improvement_summary.md)** - Complete overview of schema validation findings
  - Executive summary of all improvements
  - Field-by-field analysis with recommendations
  - Implementation checklist
  - Expected impact assessment

### Detailed Verification Documents

Individual analysis documents for each schema component:

1. **[legal_references_verification.md](legal_references_verification.md)** - Legal citation types (8 types)
2. **[decision_types_verification.md](decision_types_verification.md)** - Court decision classifications (9 types)
3. **[court_types_verification.md](court_types_verification.md)** - Polish court hierarchy (8 types)
4. **[party_types_verification.md](party_types_verification.md)** - Legal party classifications (13 types)
5. **[tax_types_verification.md](tax_types_verification.md)** - Polish tax system types (15 types)
6. **[new_fields_specification.md](new_fields_specification.md)** - New required fields (factual_state, legal_state)

---

## Key Findings

### Schema Improvements Identified

| Component | Status | Recommendation |
|-----------|--------|----------------|
| Legal References | ❌ Incomplete | Expand from 4 to 8 types |
| Decision Types | ❌ Incomplete | Expand from 5 to 9 types |
| Court Types | ❌ English terms | Use Polish, expand to 8 types |
| Party Types | ❌ Too vague | Define 13 specific types |
| Tax Types | ❌ Incomplete | Define 15 specific types |
| factual_state field | ✅ Missing | Add new field for factual context |
| legal_state field | ✅ Missing | Add new field for legal framework |

### Impact

- **+2 new essential fields** capturing factual and legal context
- **Better enumeration coverage** for Polish legal system
- **Polish terminology** matching source documents
- **Expected coverage improvements**: 60-80% → 85-95%

---

## How to Use These Documents

### For Developers

When implementing or updating the extraction schema:

1. Read **schema_improvement_summary.md** for the complete picture
2. Reference specific verification documents for detailed field specifications
3. Use the recommended enumerations in Polish terminology
4. Follow the implementation checklist

### For Researchers

When analyzing extraction results:

1. Use these documents to understand schema design decisions
2. Reference the validation methodology (web search against Polish legal system)
3. Compare extraction coverage against expected benchmarks
4. Identify gaps or areas for further improvement

### For Data Scientists

When working with extracted data:

1. Understand the complete field structure and types
2. Use enumeration lists for data validation
3. Reference expected coverage metrics
4. Plan data cleaning strategies for missing fields

---

## Schema Structure

### Current Schema (16 fields)

```
Core Identification (4 fields)
├── document_number
├── document_type
├── title
└── date_issued

High-Priority Augmentation (3 fields)
├── summary
├── thesis
└── keywords

Factual & Legal Context (2 NEW fields)
├── factual_state ← NEW
└── legal_state ← NEW

Outcome (1 field)
└── outcome (with decision_type enumeration)

Legal Content (3 fields)
├── legal_references (with type enumeration)
├── legal_concepts
└── parties (with party_type enumeration)

Structured Content (1 field)
└── legal_analysis

Document-Specific (2 fields)
├── judgment_specific (with court_type enumeration)
└── tax_interpretation_specific (with tax_type enumeration)
```

---

## Related Documentation

### Implementation

- [Extraction How-To Guides](../../how-to/extraction/) - Practical guides for running extraction
- [Structured Output Implementation](../../explanation/structured-output-implementation.md) - Technical details of Gemini structured output

### Data Management

- [Weaviate Schema Management](../../how-to/weaviate-schema-management.md) - How to update Weaviate properties
- [Ingestion Integration](../../INGESTION_INTEGRATION_COMPLETE.md) - Ingesting extracted data back to Weaviate

---

## Validation Methodology

All schema improvements were validated through:

1. **Web Search**: Research on Polish legal system, court hierarchy, tax types
2. **Document Analysis**: Review of actual Polish legal documents
3. **Expert Consultation**: Legal terminology verification
4. **Coverage Analysis**: Field population rates in test extractions

**Quality Assurance**: Each enumeration is backed by authoritative sources from Polish legal system documentation.

---

## Next Steps

1. ✅ Schema validation complete
2. ⏳ Implement schema improvements
3. ⏳ Test on sample documents (5-10)
4. ⏳ Validate coverage improvements
5. ⏳ Roll out to production pipeline

---

_This is reference documentation for the JuDDGES extraction schema, validated 2025-10-11._
