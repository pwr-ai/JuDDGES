# Extraction Schema Reference

This directory contains comprehensive validation and specification documents for the JuDDGES extraction schema used for Polish legal documents.

## Overview

The extraction schema defines the structured fields that the LLM extraction pipeline identifies and extracts from legal documents (court judgments and tax interpretations).

**Schema Implementation**: [`scripts/extraction/run_extraction_rest.py`](../../../scripts/extraction/run_extraction_rest.py)

**Last Updated**: 2025-10-11

---

## Documents in This Directory

- **[DOCUMENT_SCHEMA_MAPPING.md](DOCUMENT_SCHEMA_MAPPING.md)** - Mapping between source documents and schema fields
- **[dataset_weaviate_mapping.md](dataset_weaviate_mapping.md)** - Mapping between dataset fields and Weaviate properties
- **[extraction_schema_judgments.md](extraction_schema_judgments.md)** - Extraction schema specification for court judgments
- **[extraction_schema_tax_interpretations.md](extraction_schema_tax_interpretations.md)** - Extraction schema specification for tax interpretations
- **[gemini_extraction_schema.md](gemini_extraction_schema.md)** - Gemini structured output schema definition
- **[llm_field_extraction_schema.yaml](llm_field_extraction_schema.yaml)** - YAML field extraction schema used by the LLM pipeline

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

1. Reference the per-document schema specifications for detailed field definitions
2. Use the recommended enumerations in Polish terminology

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
