# Tutorial: Advanced Information Extraction with Gemini

Master advanced information extraction techniques including complex schemas, multi-step pipelines, validation, and quality control for production legal document processing.

## Table of Contents

- [Learning Objectives](#learning-objectives)
- [Prerequisites](#prerequisites)
- [Step 1: Complex Extraction Schemas](#step-1-complex-extraction-schemas)
- [Step 2: Multi-Step Extraction Pipeline](#step-2-multi-step-extraction-pipeline)
- [Step 3: Validation and Quality Control](#step-3-validation-and-quality-control)
- [Step 4: Batch Processing at Scale](#step-4-batch-processing-at-scale)
- [Step 5: Production Deployment](#step-5-production-deployment)
- [Best Practices](#best-practices)
- [Summary](#summary)

---

## Learning Objectives

- ✅ Design complex extraction schemas with nested structures
- ✅ Build multi-step extraction pipelines
- ✅ Implement validation and quality control
- ✅ Process documents at scale
- ✅ Monitor and optimize production systems

**Estimated Time**: 45 minutes

---

## Prerequisites

- Completion of [Tutorial 1](./tutorial-01-first-legal-document-analysis.md)
- [Gemini Extraction Tutorial](./GEMINI_EXTRACTION.md) completed
- Google API key configured
- Familiarity with Pydantic models

---

## Step 1: Complex Extraction Schemas

### Multi-Level Nested Schema

```python
"""Define a comprehensive extraction schema for court judgments."""

from juddges.extraction.gemini_chain import ExtractionSchema, DocumentType
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

# Define structured output models
class Party(BaseModel):
    """Legal party information."""
    name: str
    role: str  # "plaintiff", "defendant", "appellant", etc.
    represented_by: Optional[str] = None

class LegalReference(BaseModel):
    """Reference to legal article or statute."""
    statute: str
    article: str
    paragraph: Optional[str] = None

class MonetaryAmount(BaseModel):
    """Monetary value with currency."""
    amount: float
    currency: str = "PLN"
    description: str

# Create extraction schema
complex_schema = ExtractionSchema(
    fields={
        # Basic fields
        "verdict_date": "date as ISO 8601, data wydania wyroku",
        "case_signature": "string, sygnatura sprawy",
        "court": "string, pełna nazwa sądu",

        # Complex nested fields
        "parties": "List[object], strony procesu. Dla każdej strony wyodrębnij: name (string), role (string), represented_by (string, optional)",

        "legal_basis": "List[object], podstawa prawna. Dla każdego przepisu: statute (string, nazwa ustawy), article (string, numer artykułu), paragraph (string, optional)",

        "monetary_amounts": "List[object], kwoty pieniężne. Dla każdej kwoty: amount (float), currency (string), description (string, opis czego dotyczy)",

        # Structured verdicts
        "verdict_items": "List[string], poszczególne punkty wyroku",

        # Boolean flags
        "appeal_allowed": "boolean, czy przysługuje apelacja",
        "is_final": "boolean, czy wyrok jest prawomocny",

        # Complex text fields
        "case_summary": "string, streszczenie stanu faktycznego (200-300 słów)",
        "reasoning": "string, uzasadnienie wyroku (kluczowe argumenty)",
    },
    instructions=(
        "Wyodrębnij wszystkie informacje zgodnie z podanymi typami. "
        "Dla list obiektów, każdy element musi zawierać wszystkie wymagane pola. "
        "Dla kwot: jeśli waluta nie jest podana, użyj PLN. "
        "Dla dat: zawsze format ISO 8601 (YYYY-MM-DD). "
        "Dla boolean: true tylko gdy wyraźnie potwierdzone w tekście."
    ),
    language="polish",
)

# Example usage
from juddges.extraction import GeminiExtractionChain

chain = GeminiExtractionChain(model_name="gemini-2.5-pro")  # Use Pro for complex schemas

result = chain.extract(
    document_type=DocumentType.JUDGMENT,
    text=judgment_text,
    schema=complex_schema,
)

# Result will have nested structure:
# {
#   "parties": [
#     {"name": "Jan Kowalski", "role": "plaintiff", "represented_by": "Adw. Anna Nowak"},
#     {"name": "XYZ Bank", "role": "defendant", "represented_by": "r.pr. Marek Wiśniewski"}
#   ],
#   "monetary_amounts": [
#     {"amount": 50000.0, "currency": "CHF", "description": "kapitał kredytu"},
#     {"amount": 12000.0, "currency": "PLN", "description": "koszty postępowania"}
#   ],
#   ...
# }
```

### Schema for Swiss Franc Loans

```python
"""Specialized schema for Swiss franc loan cases."""

swiss_franc_schema = ExtractionSchema(
    fields={
        # Loan details
        "loan_amount": "float, kwota kredytu w CHF",
        "loan_currency": "string, waluta kredytu (CHF)",
        "loan_date": "date as ISO 8601, data zawarcia umowy",
        "exchange_rate_clause": "string, dokładne brzmienie klauzuli waloryzacyjnej",

        # Bank information
        "bank_name": "string, nazwa banku",
        "bank_representatives": "List[string], przedstawiciele banku",

        # Legal claims
        "claim_type": "string, główne żądanie powoda (np. 'unieważnienie umowy', 'ustalenie nieważności')",
        "claim_amount": "float, optional, wysokość roszczenia w PLN",

        # Court decision
        "verdict_type": "string, typ rozstrzygnięcia (np. 'uwzględniono w całości', 'oddalono', 'uwzględniono częściowo')",
        "verdict_reasoning": "string, główne argumenty uzasadnienia (2-3 zdania)",

        # Specific findings
        "unfair_terms_found": "boolean, czy stwierdzono niedozwolone postanowienia",
        "contract_invalidated": "boolean, czy unieważniono umowę",

        # Referenced case law
        "cjeu_references": "List[string], odwołania do orzeczeń TSUE",
        "supreme_court_references": "List[string], odwołania do orzeczeń SN",
    },
    instructions=(
        "To jest sprawa dotycząca kredytu we frankach szwajcarskich. "
        "Szczególną uwagę zwróć na: "
        "1) Dokładne brzmienie klauzuli waloryzacyjnej "
        "2) Argumenty dotyczące niedozwolonych postanowień "
        "3) Odwołania do orzecznictwa TSUE i SN "
        "4) Kwoty pieniężne (kredyt, roszczenie, koszty)"
    ),
    language="polish",
)
```

---

## Step 2: Multi-Step Extraction Pipeline

### Build a Pipeline

```python
"""Multi-step extraction pipeline with progressive refinement."""

from typing import Dict, Any
from loguru import logger

class ExtractionPipeline:
    """Multi-stage extraction pipeline."""

    def __init__(self):
        self.chain = GeminiExtractionChain(
            model_name="gemini-2.5-pro",
            cache_path=".cache/pipeline.db",
        )

    def step1_basic_metadata(self, text: str) -> Dict[str, Any]:
        """Step 1: Extract basic metadata."""
        logger.info("Step 1: Extracting basic metadata")

        schema = ExtractionSchema(
            fields={
                "court": "string",
                "date": "date as ISO 8601",
                "case_number": "string",
                "court_type": "string (civil, criminal, administrative)",
            },
            language="polish",
        )

        return self.chain.extract(
            document_type=DocumentType.JUDGMENT,
            text=text[:5000],  # Use beginning for metadata
            schema=schema,
        )

    def step2_parties_and_representatives(self, text: str) -> Dict[str, Any]:
        """Step 2: Extract parties and their representatives."""
        logger.info("Step 2: Extracting parties")

        schema = ExtractionSchema(
            fields={
                "plaintiffs": "List[object], each with: name (string), represented_by (string, optional)",
                "defendants": "List[object], each with: name (string), represented_by (string, optional)",
            },
            instructions="Wyodrębnij wszystkie strony z sekcji nagłówkowej wyroku.",
            language="polish",
        )

        return self.chain.extract(
            document_type=DocumentType.JUDGMENT,
            text=text[:10000],  # Use beginning for parties
            schema=schema,
        )

    def step3_legal_analysis(self, text: str, metadata: Dict) -> Dict[str, Any]:
        """Step 3: Extract legal reasoning (context-aware)."""
        logger.info("Step 3: Extracting legal analysis")

        # Use metadata from step 1 to guide extraction
        court_type = metadata.get("court_type", "")

        schema = ExtractionSchema(
            fields={
                "legal_basis": "List[string], referenced laws and articles",
                "key_arguments": "List[string], main legal arguments (3-5 points)",
                "precedents": "List[string], referenced court decisions",
            },
            instructions=f"Wyrok z sądu {court_type}. Skup się na argumentacji prawnej.",
            language="polish",
        )

        return self.chain.extract(
            document_type=DocumentType.JUDGMENT,
            text=text,  # Use full text
            schema=schema,
        )

    def step4_verdict_details(self, text: str) -> Dict[str, Any]:
        """Step 4: Extract detailed verdict."""
        logger.info("Step 4: Extracting verdict details")

        schema = ExtractionSchema(
            fields={
                "verdict_items": "List[string], each point of the verdict",
                "costs": "float, optional, costs awarded in PLN",
                "appeal_allowed": "boolean",
                "appeal_deadline": "string, optional, deadline for appeal",
            },
            language="polish",
        )

        # Focus on verdict section (usually at end)
        verdict_section = text[-20000:]  # Last 20k chars

        return self.chain.extract(
            document_type=DocumentType.JUDGMENT,
            text=verdict_section,
            schema=schema,
        )

    def run_full_pipeline(self, text: str) -> Dict[str, Any]:
        """Run complete extraction pipeline."""
        logger.info("Starting full extraction pipeline")

        # Execute steps sequentially
        metadata = self.step1_basic_metadata(text)
        parties = self.step2_parties_and_representatives(text)
        legal = self.step3_legal_analysis(text, metadata)
        verdict = self.step4_verdict_details(text)

        # Combine results
        result = {
            **metadata,
            **parties,
            **legal,
            **verdict,
            "pipeline_version": "1.0",
        }

        logger.info("Pipeline complete")
        return result

# Usage
pipeline = ExtractionPipeline()
result = pipeline.run_full_pipeline(judgment_text)
```

---

## Step 3: Validation and Quality Control

### Implement Validation

```python
"""Validation layer for extraction results."""

from pydantic import BaseModel, validator, ValidationError
from datetime import datetime
from typing import List, Optional

class ValidatedExtraction(BaseModel):
    """Validated extraction result with constraints."""

    # Required fields
    verdict_date: str
    court: str
    case_signature: str

    # Optional fields
    parties: Optional[List[str]] = None
    verdict: Optional[str] = None
    legal_basis: Optional[List[str]] = None

    @validator("verdict_date")
    def validate_date(cls, v):
        """Ensure date is valid ISO 8601."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}")
        return v

    @validator("case_signature")
    def validate_signature(cls, v):
        """Ensure case signature matches expected pattern."""
        import re
        # Pattern: Roman numerals + letter + numbers/year
        pattern = r"^[IVX]+\s+[A-Z]+\s+\d+/\d{4}$"
        if not re.match(pattern, v.strip()):
            raise ValueError(f"Invalid case signature format: {v}")
        return v

    @validator("court")
    def validate_court(cls, v):
        """Ensure court name is not empty."""
        if not v or v.strip() == "":
            raise ValueError("Court name cannot be empty")
        return v

# Usage
def extract_with_validation(text: str, chain: GeminiExtractionChain) -> ValidatedExtraction:
    """Extract and validate."""
    # Extract
    result = chain.extract(
        document_type=DocumentType.JUDGMENT,
        text=text,
        schema=schema,
    )

    # Validate
    try:
        validated = ValidatedExtraction(**result)
        logger.info("✓ Validation passed")
        return validated
    except ValidationError as e:
        logger.error(f"✗ Validation failed: {e}")
        raise

# Run with validation
try:
    result = extract_with_validation(text, chain)
    print(f"Valid result: {result.dict()}")
except ValidationError as e:
    print(f"Invalid extraction: {e}")
```

### Quality Scoring

```python
"""Calculate quality score for extractions."""

def calculate_quality_score(result: Dict[str, Any]) -> float:
    """Score extraction quality (0-1)."""
    score = 0.0
    max_score = 0.0

    # Check completeness
    required_fields = ["verdict_date", "court", "case_signature"]
    for field in required_fields:
        max_score += 10
        if result.get(field) and result[field] != "":
            score += 10

    # Check field validity
    if result.get("verdict_date"):
        max_score += 10
        try:
            datetime.fromisoformat(result["verdict_date"])
            score += 10
        except:
            pass

    # Check richness (optional fields)
    optional_fields = ["parties", "legal_basis", "verdict"]
    for field in optional_fields:
        max_score += 5
        if result.get(field) and len(str(result[field])) > 10:
            score += 5

    return score / max_score if max_score > 0 else 0.0

# Usage
quality = calculate_quality_score(result)
if quality < 0.7:
    logger.warning(f"Low quality extraction: {quality:.2f}")
else:
    logger.info(f"Good quality extraction: {quality:.2f}")
```

---

## Step 4: Batch Processing at Scale

### Parallel Processing

```python
"""Process documents at scale with parallel execution."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from tqdm import tqdm
import json

def process_document(doc_id: str, text: str, chain: GeminiExtractionChain) -> Dict:
    """Process single document."""
    try:
        result = chain.extract(
            document_type=DocumentType.JUDGMENT,
            text=text,
            schema=schema,
        )
        result["_id"] = doc_id
        result["_status"] = "success"
        return result
    except Exception as e:
        return {
            "_id": doc_id,
            "_status": "error",
            "_error": str(e),
        }

def batch_process(
    dataset,
    output_file: str,
    max_workers: int = 10,
    batch_size: int = 100,
):
    """Process dataset in parallel."""
    chain = GeminiExtractionChain(
        model_name="gemini-2.5-flash",  # Flash for speed
        cache_path=".cache/batch_processing.db",
    )

    results = []
    total = len(dataset)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_document,
                doc.get("id", str(i)),
                doc["text"],
                chain
            ): i
            for i, doc in enumerate(dataset)
        }

        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=total, desc="Processing"):
            try:
                result = future.result(timeout=60)
                results.append(result)

                # Save intermediate results
                if len(results) % batch_size == 0:
                    with open(output_file, "w") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)

            except Exception as e:
                logger.error(f"Task failed: {e}")

    # Final save
    with open(output_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    successes = sum(1 for r in results if r.get("_status") == "success")
    failures = len(results) - successes

    logger.info(f"Processed {len(results)} documents: {successes} success, {failures} failures")

# Usage
dataset = load_dataset("JuDDGES/pl-court-raw-sample", split="train[:1000]")
batch_process(dataset, "extraction_results.json", max_workers=20)
```

---

## Step 5: Production Deployment

### Monitoring and Logging

```python
"""Production-ready extraction with monitoring."""

from langfuse.langchain import CallbackHandler
import time

class ProductionExtractionService:
    """Production extraction service with monitoring."""

    def __init__(self, langfuse_public_key: str, langfuse_secret_key: str):
        self.chain = GeminiExtractionChain(
            model_name="gemini-2.5-flash",
            cache_path=".cache/production.db",
        )

        # Set up Langfuse
        self.langfuse_handler = CallbackHandler(
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
        )

    def extract_with_monitoring(
        self,
        document_id: str,
        text: str,
        schema: ExtractionSchema,
    ) -> Dict[str, Any]:
        """Extract with full monitoring."""

        start_time = time.time()

        try:
            # Extract with Langfuse tracing
            result = self.chain.extract(
                document_type=DocumentType.JUDGMENT,
                text=text,
                schema=schema,
                langfuse_handler=self.langfuse_handler,
            )

            # Calculate quality
            quality_score = calculate_quality_score(result)

            # Log metrics
            duration = time.time() - start_time
            logger.info(
                f"Extracted doc {document_id}: "
                f"quality={quality_score:.2f}, "
                f"duration={duration:.2f}s"
            )

            # Add metadata
            result["_metadata"] = {
                "document_id": document_id,
                "quality_score": quality_score,
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat(),
            }

            return result

        except Exception as e:
            logger.error(f"Extraction failed for {document_id}: {e}")
            raise

# Usage
service = ProductionExtractionService(
    langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
)

result = service.extract_with_monitoring(
    document_id="12345",
    text=judgment_text,
    schema=complex_schema,
)
```

---

## Best Practices

### 1. Schema Design

- ✅ Start simple, add complexity gradually
- ✅ Use clear, specific field descriptions
- ✅ Provide examples in instructions
- ✅ Test on diverse documents

### 2. Error Handling

```python
# Always use try-except
try:
    result = chain.extract(...)
except Exception as e:
    logger.error(f"Extraction failed: {e}")
    # Retry with simpler schema or skip
```

### 3. Performance Optimization

- Use `gemini-2.5-flash` for speed
- Enable caching (default)
- Process in parallel (10-20 workers)
- Monitor API quotas

### 4. Quality Assurance

- Validate all results with Pydantic
- Calculate quality scores
- Review low-quality extractions
- Iterate on schema design

---

## Summary

You've mastered advanced information extraction!

### What You've Learned

✅ Complex nested schemas
✅ Multi-step pipelines
✅ Validation and quality control
✅ Batch processing at scale
✅ Production deployment with monitoring

### Next Steps

- [Tutorial 5: End-to-End Project](./tutorial-05-end-to-end-project.md)
- [Gemini API Troubleshooting](../how-to/troubleshooting/GEMINI_API_TROUBLESHOOTING.md)

---

**Last Updated**: 2025-10-11 | **Version**: 1.0 | **Status**: Published
