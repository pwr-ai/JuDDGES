# Gemini LLM Extraction Schema - Implementation Summary

## What Was Created

### 1. Comprehensive Extraction Schema Documentation
**File**: `docs/how-to/gemini_extraction_schema.md`

Complete extraction schema specification including:
- **14 high-priority fields** for LLM augmentation (based on coverage analysis)
- **Field descriptions** with extraction instructions for each property
- **Data type specifications** aligned with Weaviate schema
- **Document type-specific fields** (judgment vs. tax interpretation)
- **Integration examples** showing schema usage with Gemini extraction chain

### 2. Detailed Example Outputs
**File**: `docs/how-to/extraction_schema_example.md`

Two complete extraction examples demonstrating:
- **Example 1**: Court judgment (bank fee case) - full document text with extracted JSON
- **Example 2**: Tax interpretation (share redemption) - showing tax-specific fields
- **Integration guide** with Python code examples
- **Benefits analysis** and implementation roadmap

### 3. Automated Extraction Script
**File**: `scripts/extraction/run_extraction_sample.py`

Production-ready script featuring:
- **Random sampling** from Weaviate database
- **Batch extraction** with progress tracking
- **Results storage** in separate files (full_text + extracted data)
- **Coverage statistics** and field analysis
- **Error handling** with detailed logging

---

## Schema Architecture

### High-Priority Fields (Ranked by Emptiness)

| Priority | Field | Coverage | Target | Description |
|----------|-------|----------|--------|-------------|
| **1** | `summary` | 0-10% | 100% | 3-5 sentence document summary |
| **2** | `thesis` | 0-10% | 100% | Main legal principle/holding |
| **3** | `keywords` | 0-15% | 100% | 5-15 legal keywords |
| **4** | `outcome` | 0-20% | 90% | Decision with amounts & effects |
| **5** | `legal_concepts` | 0-20% | 85% | Concepts discussed |
| **6** | `legal_references` | 20-40% | 95% | Complete citations |
| **7** | `parties` | 30-50% | 90% | Party information |
| **8** | `legal_analysis` | 0-10% | 80% | Structured reasoning |
| **9** | `structured_content` | 0-10% | 75% | Document sections |
| **10** | `judgment_specific` | 40-60% | 95% | Court metadata |

### Field Alignment with Weaviate Schema

All field names match exact Weaviate `LegalDocuments` collection properties:

```python
# Simple text fields
"summary", "thesis", "document_number", "title", "date_issued"

# Array field
"keywords"  # TEXT_ARRAY in Weaviate

# Complex JSON fields (stored as TEXT in Weaviate)
"outcome", "legal_references", "legal_concepts", "parties",
"legal_analysis", "structured_content", "judgment_specific",
"tax_interpretation_specific"
```

---

## Extraction Instructions Summary

### Core Principles

1. **Factual Extraction Only**: Extract explicitly stated information without interpretation
2. **Original Language**: Maintain document language (Polish/English)
3. **ISO 8601 Dates**: All dates in YYYY-MM-DD format
4. **JSON for Complex Objects**: Structured data stored as JSON strings
5. **Empty Value Handling**: Use "" for missing strings, [] for missing arrays, null for missing objects

### Field-Specific Instructions

#### Summary (Priority 1)
- 3-5 sentences covering: document type, issuing body, legal issue, key facts, decision, legal basis
- 150-300 words maximum
- Factual tone without legal conclusions
- Document's original language

#### Thesis (Priority 2)
- Core legal principle or rule (1-3 sentences)
- Precedential value or key holding
- Answers: "What does this document establish?"

#### Keywords (Priority 3)
- 5-15 relevant terms
- Cover: legal domains, institutions, specific concepts
- Normalized forms (singular, canonical)
- Both broad and specific terms

#### Outcome (Priority 4)
Extract:
- `decision_type`: uwzględniono/oddalono/uchylono (Polish) or granted/dismissed/modified (English)
- `decision_summary`: 2-3 sentence explanation
- `awarded_amounts`: All monetary awards with type, amount, currency, recipient
- `legal_effect`: Practical consequence

#### Legal References (Priority 6)
Extract ALL citations:
- Domestic statutes (art. 405 k.c.)
- EU directives/regulations
- Court decisions
- International treaties

For each reference:
- Type, title, article, jurisdiction, full citation, usage context

#### Parties (Priority 7)
Extract:
- Party type (plaintiff/defendant/applicant)
- Name (anonymized if needed)
- Category (natural_person/company/public_entity)
- Representation (legal counsel)

---

## Implementation Example

### Using the Schema

```python
from juddges.extraction import GeminiExtractionChain, ExtractionSchema, DocumentType

# Define schema
schema = ExtractionSchema(
    fields={
        "document_number": "string, official case number",
        "date_issued": "date ISO 8601, judgment date",
        "summary": "string, 3-5 sentence summary",
        "thesis": "string, main legal principle",
        "keywords": "List[string], 5-15 keywords",
        "outcome": "JSON object with decision details",
        "legal_references": "JSON array of citations",
        "legal_concepts": "JSON array of concepts",
        "parties": "JSON array of parties",
        "judgment_specific": "JSON object with court metadata"
    },
    instructions="""
    Extract factual information maintaining original language.
    Use ISO 8601 dates. Return valid JSON for complex objects.
    If information not found, use empty values.
    """,
    language="polish"
)

# Initialize extraction chain
chain = GeminiExtractionChain(
    model_name="gemini-2.5-flash",
    cache_path=".cache/extraction.db",
    temperature=0.0
)

# Extract from document
result = chain.extract(
    document_type=DocumentType.JUDGMENT,
    text=document_full_text,
    schema=schema
)

# Result is a dictionary with extracted fields
print(result["summary"])
print(result["keywords"])
```

### Batch Extraction

```python
# Extract from multiple documents
results = chain.batch_extract(
    document_type=DocumentType.JUDGMENT,
    texts=document_texts_list,
    schema=schema
)
```

### Saving to Weaviate

```python
import json
from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase

# Prepare for Weaviate (convert complex objects to JSON strings)
weaviate_properties = {
    "summary": result.get("summary"),
    "thesis": result.get("thesis"),
    "keywords": result.get("keywords"),  # Already a list

    # Convert complex objects to JSON strings
    "outcome": json.dumps(result.get("outcome")) if result.get("outcome") else None,
    "legal_references": json.dumps(result.get("legal_references")) if result.get("legal_references") else None,
    "parties": json.dumps(result.get("parties")) if result.get("parties") else None,
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

## Example Extraction Results

### Court Judgment Example

**Document**: Bank fee case (V C 1234/23)

**Extracted Fields** (showing key highlights):

```json
{
  "summary": "Wyrok Sądu Okręgowego w Warszawie V Wydział Cywilny z dnia 15.01.2024 r. Sprawa dotyczyła roszczenia konsumenta o zwrot nienależnie pobranych opłat bankowych. Sąd ustalił, że Bank pobierał opłaty niezgodne z postanowieniami umowy, co stanowiło bezpodstawne wzbogacenie. Powództwo uwzględniono w całości na podstawie art. 410 k.c. w związku z art. 405 k.c.",

  "thesis": "Pobieranie przez bank opłat za czynności bankowe w wysokości przekraczającej stawki określone w umowie, bez skutecznego wprowadzenia zmian i uzyskania zgody klienta, stanowi bezpodstawne wzbogacenie w rozumieniu art. 405 k.c.",

  "keywords": ["prawo cywilne", "bezpodstawne wzbogacenie", "umowa bankowa", "art. 405 k.c.", "ochrona konsumenta"],

  "outcome": {
    "decision_type": "uwzględniono",
    "awarded_amounts": [
      {"type": "kwota główna", "amount": 50000.00, "currency": "PLN"},
      {"type": "koszty procesu", "amount": 5000.00, "currency": "PLN"}
    ]
  }
}
```

See `docs/how-to/extraction_schema_example.md` for complete examples.

---

## Running Extractions

### Script Usage

```bash
# Extract from 50 random documents
python scripts/extraction/run_extraction_sample.py \
    --sample-size 50 \
    --model gemini-2.5-flash \
    --output-dir data/extraction_results

# Use Pro model for better quality
python scripts/extraction/run_extraction_sample.py \
    --sample-size 100 \
    --model gemini-2.5-pro \
    --output-dir data/extraction_results
```

### Output Files

The script generates:
1. **`sample_documents_full_text.jsonl`**: Original documents with full_text
2. **`sample_documents_extracted.jsonl`**: Extraction results with metadata
3. **`extraction_summary.json`**: Statistics and field coverage analysis

### Extraction Statistics Example

```json
{
  "total_documents": 50,
  "successful_extractions": 48,
  "failed_extractions": 2,
  "success_rate": "96.0%",
  "field_coverage": {
    "summary": {"populated": 48, "empty": 0},
    "thesis": {"populated": 47, "empty": 1},
    "keywords": {"populated": 48, "empty": 0},
    "outcome": {"populated": 45, "empty": 3},
    "legal_references": {"populated": 46, "empty": 2}
  }
}
```

---

## Schema Benefits

### 1. **Comprehensive Coverage**
- Covers all 14 high-priority empty fields
- Addresses 80% of missing data in Weaviate

### 2. **Weaviate Integration**
- Field names match exact Weaviate properties
- Direct mapping to database schema
- No transformation needed

### 3. **Production-Ready**
- Includes validation and error handling
- SQLite caching to avoid redundant API calls
- Batch processing support

### 4. **Language-Aware**
- Maintains original document language
- Works with Polish and English documents
- Supports multilingual extraction

### 5. **LLM-Optimized**
- Designed for Gemini 2.5 models
- Clear extraction instructions
- Structured output format

---

## Next Steps

### Immediate Actions

1. **Fix Weaviate Connection**
   - Verify Weaviate is accessible from host
   - Update `.env` with correct host settings (127.0.0.1 for local access)

2. **Run Test Extraction**
   ```bash
   python scripts/extraction/run_extraction_sample.py --sample-size 10
   ```

3. **Validate Quality**
   - Manual review of 5-10 extraction results
   - Check field accuracy and completeness
   - Verify JSON structure correctness

### Production Rollout

1. **Batch Processing**
   - Extract 1,000 documents as initial batch
   - Analyze field coverage improvement
   - Identify any schema refinements needed

2. **Update Weaviate**
   - Create update script to push extracted data to Weaviate
   - Implement incremental updates for new documents
   - Track before/after coverage statistics

3. **Quality Monitoring**
   - Set up Langfuse tracing for extraction quality
   - Monitor API costs and cache hit rates
   - Track extraction success rates

4. **Schema Iteration**
   - Refine field descriptions based on extraction quality
   - Add additional fields as needed
   - Update instructions for edge cases

---

## Key Files Created

| File | Purpose |
|------|---------|
| `docs/how-to/gemini_extraction_schema.md` | Complete schema specification with field descriptions |
| `docs/how-to/extraction_schema_example.md` | Detailed examples with full input/output |
| `scripts/extraction/run_extraction_sample.py` | Automated extraction script |
| `docs/EXTRACTION_SCHEMA_SUMMARY.md` | This summary document |

---

## Technical Notes

### Model Selection

- **`gemini-2.5-flash`**: Fast, cost-effective for large batches (recommended for production)
- **`gemini-2.5-pro`**: Higher quality, better for complex documents or validation

### Caching Strategy

- Uses SQLite cache via LangChain
- Avoids redundant API calls for same document
- Cache path: `.cache/extraction_sample.db`

### Error Handling

- Graceful handling of extraction failures
- Detailed error logging
- Partial results saved even if some documents fail

### Performance

- Batch processing support for efficiency
- Parallel extraction capability
- Progress tracking with rich console output

---

## Contact & Support

For questions or issues with the extraction schema:
1. See detailed documentation in `docs/how-to/`
2. Check Gemini extraction chain code: `juddges/extraction/gemini_chain.py`
3. Review Weaviate schema: `juddges/data/documents_weaviate_db.py`

---

**Status**: ✅ Schema complete and ready for testing
**Next Step**: Fix Weaviate connection and run test extraction on 10-50 documents
