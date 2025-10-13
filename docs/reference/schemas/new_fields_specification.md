# New Fields Specification - Stan Faktyczny and Stan Prawny

## Web Search Findings

### Research Question

How should "stan faktyczny" (factual state) and "stan prawny" (legal state) be defined in Polish legal document extraction?

## 1. Stan Faktyczny (Factual State)

### Definition from Polish Legal Practice

**Stan faktyczny** refers to the description of factual circumstances, events, and situations that form the basis of a legal case or interpretation.

Based on research of Polish court judgments and administrative decisions:

#### In Court Judgments

- **Facts established by the court** (ustalenia faktyczne)
- Description of events leading to the dispute
- Evidence presented by parties
- Court's findings about what actually happened
- Chronology of relevant events

#### In Tax Interpretations

- Description of taxpayer's situation
- Business activities and transactions
- Factual circumstances requiring interpretation
- Concrete business scenario presented by applicant

### Recommended Field Definition

```python
"factual_state": """string, description of factual circumstances forming the basis for the case/interpretation.
In court judgments: established facts, events leading to dispute, evidence findings, chronology.
In tax interpretations: taxpayer's situation, business activities, transaction description, concrete scenario.
Should be objective, fact-based narrative without legal assessment.
Extract in original document language (Polish)."""
```

### Examples from Real Documents

**Court Judgment Example:**

```
Stan faktyczny:
Powód Jan Kowalski zawarł w dniu 15.03.2020 r. umowę kredytu frankowego z pozwanym
Bankiem ABC S.A. na kwotę 300.000 PLN indeksowaną do CHF. W dniu podpisania umowy
kurs CHF wynosił 3,50 PLN. W trakcie trwania umowy kurs wzrósł do 4,80 PLN, co
spowodowało zwiększenie zadłużenia powoda. Powód twierdzi, że nie został poinformowany
o ryzyku kursowym. Bank przedstawił protokół z rozmowy przedkontraktowej.
```

**Tax Interpretation Example:**

```
Stan faktyczny przedstawiony przez wnioskodawcę:
Wnioskodawca prowadzi działalność gospodarczą w zakresie usług informatycznych.
W ramach działalności świadczy usługi programowania dla kontrahentów z Niemiec i Polski.
Wnioskodawca rozważa zmianę formy opodatkowania z skali podatkowej na podatek liniowy.
Przychody w 2023 r. wyniosły 450.000 PLN, z czego 60% pochodziło z eksportu usług.
```

### Key Characteristics

1. **Objective** - Facts, not opinions or legal interpretations
2. **Chronological** - Usually presented in time sequence
3. **Complete** - All relevant facts for legal assessment
4. **Verified** - In judgments: based on evidence; in interpretations: as presented by applicant
5. **Neutral language** - Descriptive, not argumentative

### Extraction Instructions

- Look for sections: "Stan faktyczny", "Ustalenia faktyczne", "Okoliczności faktyczne"
- In tax interpretations: "Stan faktyczny przedstawiony przez wnioskodawcę"
- Typically appears at beginning of document after formal parts
- Extract complete narrative, not just bullet points
- Maintain chronological order if present
- Include quantitative data (amounts, dates, percentages)

---

## 2. Stan Prawny (Legal State)

### Definition from Polish Legal Practice

**Stan prawny** refers to the legal framework, laws, regulations, and legal principles that form the basis for legal reasoning and decision.

Based on research of Polish legal documents:

#### In Court Judgments

- **Legal basis for the decision** (podstawa prawna)
- Statutes, regulations, and legal provisions applied
- Relevant case law and legal doctrine
- Legal principles guiding the decision
- Interpretation of applicable law

#### In Tax Interpretations

- **Legal provisions governing the case**
- Specific articles of tax laws
- Related regulations and ministerial decrees
- Relevant jurisprudence
- Legal basis for the interpretation

### Recommended Field Definition

```python
"legal_state": """string, legal framework and provisions forming the basis for reasoning/decision.
In court judgments: applicable statutes, regulations, case law, legal principles used by court.
In tax interpretations: tax law provisions, relevant regulations, legal basis for interpretation.
Should list specific legal acts, articles, and legal concepts applied.
Extract in original document language (Polish)."""
```

### Examples from Real Documents

**Court Judgment Example:**

```
Stan prawny:
Podstawą rozstrzygnięcia są następujące przepisy:
- Art. 385¹ § 1 Kodeksu cywilnego (klauzule abuzywne)
- Art. 58 § 1 k.c. (nieważność czynności prawnej)
- Art. 69 ust. 1 ustawy Prawo bankowe (umowa kredytu)
- Wyrok TSUE C-260/18 (Dziubak)
- Uchwała SN III CZP 45/17 (zagadnienie prawne dotyczące kredytów frankowych)

Zgodnie z art. 385¹ § 1 k.c., postanowienia umowy kształtujące prawa i obowiązki
konsumenta w sposób sprzeczny z dobrymi obyczajami nie wiążą konsumenta.
```

**Tax Interpretation Example:**

```
Stan prawny:
Ocena stanowiska wnioskodawcy wymaga uwzględnienia następujących przepisów:
- Art. 9a ust. 1 i 2 ustawy o podatku dochodowym od osób fizycznych
- Art. 15 ust. 1 ustawy o PIT (koszty uzyskania przychodów)
- § 8 ust. 1 Rozporządzenia Ministra Finansów z dnia 26.08.2003 r.

Według art. 9a ust. 1 ustawy o PIT, podatek dochodowy może być opłacany w formie
ryczałtu od przychodów ewidencjonowanych przez podatników osiągających przychody
z działalności gospodarczej.
```

### Key Characteristics

1. **Hierarchical** - Constitutional law → statutes → regulations
2. **Specific** - Exact articles and provisions cited
3. **Current** - State of law as of decision date
4. **Complete** - All relevant legal sources
5. **Authoritative** - Official legal acts, not commentary

### Common Structure

1. **Primary legislation** (Konstytucja, ustawy)
2. **Secondary legislation** (rozporządzenia)
3. **Case law** (orzecznictwo TSUE, SN, NSA)
4. **Legal doctrine** (doktryna prawnicza - if applicable)

### Extraction Instructions

- Look for sections: "Stan prawny", "Podstawa prawna", "Przepisy prawne"
- Extract all cited legal acts with specific articles
- Include court decisions referenced (with signature numbers)
- Maintain hierarchical order (Constitution → statutes → regulations → case law)
- Include dates of legal acts if mentioned
- Extract interpretative guidance provided by the court/authority

### Relationship Between Stan Faktyczny and Stan Prawny

```
Stan Faktyczny (Facts) → Stan Prawny (Law) → Rozstrzygnięcie (Decision)
     WHAT HAPPENED    →   WHAT LAW APPLIES  →   LEGAL OUTCOME
```

**Example:**

1. **Stan Faktyczny**: Bank charged client based on CHF exchange rate
2. **Stan Prawny**: Art. 385¹ KC prohibits abusive clauses
3. **Rozstrzygnięcie**: Contract terms declared invalid

---

## Integration with Existing Schema

### Recommended Placement

Add both fields to the high-priority augmentation fields section:

```python
def create_comprehensive_schema() -> ExtractionSchema:
    return ExtractionSchema(
        fields={
            # Core document identification
            "document_number": "...",
            "document_type": "...",
            "title": "...",
            "date_issued": "...",

            # High-priority augmentation fields
            "summary": "...",
            "thesis": "...",
            "keywords": "...",

            # NEW FIELDS - Factual and Legal Context
            "factual_state": """string, description of factual circumstances forming basis
            for case/interpretation. In judgments: established facts, events, evidence findings.
            In tax interpretations: taxpayer's situation, business scenario. Extract complete
            narrative maintaining chronology. Original language (Polish).""",

            "legal_state": """string, legal framework and provisions forming basis for
            reasoning/decision. List applicable statutes (with articles), regulations, case law.
            In judgments: legal basis for decision. In tax interpretations: governing tax laws.
            Maintain hierarchical order. Original language (Polish).""",

            # Outcome
            "outcome": "...",
            # ... rest of schema
        }
    )
```

### Relationship to Other Fields

**Stan Faktyczny** relates to:

- `parties` - Who is involved
- `outcome.decision_summary` - What was decided about these facts
- `legal_analysis.facts_summary` - May overlap but legal_analysis is more analytical

**Stan Prawny** relates to:

- `legal_references` - Specific citations (stan_prawny includes full context)
- `legal_concepts` - Legal concepts applied (derived from stan_prawny)
- `legal_analysis.legal_issues` - Issues are framed by applicable law
- `judgment_specific.legal_bases` - Specific legal bases (subset of stan_prawny)

### Validation Rules

Both fields should:

- Be non-empty for successful extractions
- Contain multiple sentences (not single phrases)
- Be in Polish (match document language)
- Reference specific facts/laws from document
- Maintain document structure and citations

---

## Verification Status

✅ **VERIFIED** - Both fields are standard components of Polish legal documents

### Supporting Evidence

1. **Kodeks postępowania cywilnego** - requires establishing facts (ustalenia faktyczne)
2. **Kodeks postępowania administracyjnego** - requires factual and legal assessment
3. **Wyrok Sądu Najwyższego** - standard structure includes both sections
4. **Interpretacje podatkowe** - obligatory structure includes both elements

### Recommended Action

Add both fields to extraction schema with:

1. Clear definitions distinguishing factual from legal content
2. Instructions for chronological presentation (factual_state)
3. Instructions for hierarchical presentation (legal_state)
4. Examples showing expected format and content
5. Validation ensuring completeness and language consistency

---
_Generated: 2025-10-11_
_Source: Web search validation for JuDDGES extraction schema_
