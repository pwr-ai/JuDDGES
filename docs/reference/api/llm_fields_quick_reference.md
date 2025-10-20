# LLM Field Extraction - Quick Reference

## Single-Line Field Definitions

| Field | How to Extract | Why Extract | Priority | Coverage |
|-------|----------------|-------------|----------|----------|
| `thesis` | Extract core legal principle/holding from reasoning section or conclusions | Essential for legal research - enables quick understanding of precedential value | 1 | 17.3% |
| `title` | Generate from court_name + case_type + main_legal_issue or parties | Improves discoverability in search results and document lists | 2 | 15.5% |
| `outcome` | Extract final decision from dispositif section: granted/denied/remanded/dismissed | Critical for filtering by outcome and understanding practical impact | 3 | 69.6% |
| `keywords` | Extract 5-15 legal terms covering: legal areas, concepts, procedure, subject matter | Essential for search, filtering, topic-based navigation | 4 | 83.2% |
| `summary` | Generate 3-5 paragraph summary: facts, legal issue, arguments, reasoning, conclusion | Provides overview for relevance assessment without reading full text | 5 | 99.9% |
| `legal_concepts` | Identify legal doctrines, principles, tests mentioned (e.g., proportionality, res judicata) | Enables semantic search and concept-based legal research | 6 | 0% |
| `parties` | Extract appellant/respondent, plaintiff/defendant with roles and legal status | Enables party-based search and litigation pattern analysis | 7 | 99.2% |
| `tags` | Generate hashtag-formatted semantic tags (#tax-law, #landmark-case, #eu-law-reference) | Enables faceted search and flexible categorization | 8 | 0% |
| `legal_references` | Extract structured citations: statutes (article/section), EU directives, case law (court/number/date) | Critical for precedent research and citation network analysis | 9 | 100% |
| `issuing_body` | Extract court name + chamber + jurisdiction level from metadata or document header | Essential for understanding precedential weight and filtering by court | 10 | 15.5% |
| `legal_analysis` | Analyze interpretive methods used (textual/purposive/systematic), tests applied, precedent treatment | Supports advanced research - understanding judicial reasoning patterns | 11 | 0% |
| `structured_content` | Parse into semantic sections: facts, procedural history, legal issues, arguments, reasoning, decision | Enables section-specific search and targeted information extraction | 12 | 0% |

## Extraction Priority Phases

### Phase 1: Core Fields (High Impact, Low Coverage)
1. **thesis** - Main legal principle
2. **title** - Document naming
3. **outcome** - Decision result
4. **keywords** - Search/discovery

### Phase 2: Context Fields (Enhancement)
5. **summary** - Executive overview
6. **parties** - Parties involved
7. **legal_references** - Citations

### Phase 3: Advanced Fields (Deep Analysis)
8. **legal_concepts** - Thematic categorization
9. **tags** - Flexible tagging
10. **legal_analysis** - Reasoning analysis
11. **structured_content** - Section decomposition
12. **issuing_body** - Institutional info

## Document Types & Field Relevance

| Document Type | Key Fields to Extract | Special Considerations |
|---------------|----------------------|------------------------|
| **Judgment** | thesis, outcome, parties, legal_references, court_name, presiding_judge | Focus on precedential value, decision reasoning |
| **Tax Interpretation** | thesis, summary, legal_references, issuing_body | Focus on interpretive guidance, applicable law |
| **Legal Act** | title, summary, legal_references, structured_content | Focus on statutory structure, provisions |

## Language Handling

| Language | Extraction Approach | Output Language |
|----------|-------------------|-----------------|
| Polish (pl) | Extract using Polish legal terminology | Polish |
| English (en) | Extract using English legal terminology | English |
| Mixed | Detect primary language, extract accordingly | Primary language |

## Output Formats by Field

| Field | Format | Example |
|-------|--------|---------|
| `thesis` | Sentence (50-300 chars) | "Contracts signed under duress are voidable within reasonable time after duress ceases." |
| `title` | Title Case (30-150 chars) | "Supreme Court - VAT Deduction for Holding Companies (2024)" |
| `outcome` | Statement (50-200 chars) | "Appeal granted. Lower court decision reversed. Case remanded." |
| `keywords` | Array of lowercase strings | `["tax law", "vat deduction", "holding company"]` |
| `summary` | Paragraphs (500-1500 chars) | Multi-paragraph structured summary |
| `legal_concepts` | JSON array | `[{"concept": "economic activity", "context": "VAT law qualification"}]` |
| `parties` | JSON object | `{"appellant": "ABC Holdings", "respondent": "Tax Authority"}` |
| `tags` | Array of hashtags | `["#tax-law", "#landmark-case"]` |
| `legal_references` | JSON array | `[{"type": "eu_directive", "reference": "2006/112/EC Art 168"}]` |
| `issuing_body` | JSON object | `{"name": "Supreme Court", "type": "supreme_administrative_court"}` |
| `legal_analysis` | JSON object | `{"primary_reasoning": "purposive", "methods": ["textual", "eu_conform"]}` |
| `structured_content` | JSON array | `[{"section_type": "facts", "heading": "Background", "position": 1}]` |

## Quality Criteria

| Criteria | Requirement |
|----------|-------------|
| Accuracy | Extract only verifiable information from source document |
| Completeness | Cover all relevant aspects per field definition |
| Consistency | Maintain format and terminology across documents |
| Objectivity | No editorial comments or personal interpretation |
| Language | Match source document language |
| Legal Precision | Use proper legal terminology and citations |

## Common Extraction Challenges

| Challenge | Solution |
|-----------|----------|
| Anonymous parties | Preserve anonymization (e.g., "Taxpayer", "Company A") |
| Complex citations | Parse systematically: type → reference → context |
| Mixed languages | Extract in primary language, note secondary language references |
| Unclear outcome | Extract literal decision text if ambiguous |
| Missing sections | Return `null` rather than guessing |
| Inconsistent structure | Map to semantic section types regardless of document formatting |
