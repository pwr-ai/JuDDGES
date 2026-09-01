# JuDDGES-pl: Hundreds of Thousands of Structurally Enriched Polish Judgments from Common and Administrative Courts

> **Venue:** NeurIPS 2026 Evaluations & Datasets Track
> **Status:** Markdown skeleton — to be ported to LaTeX with the NeurIPS 2026 E&D template (`neurips_2026.sty`).
> **Page budget:** main 9 pages; references / appendix / NeurIPS Paper Checklist unlimited.
> **Bibliography:** reuse `shared/bibliography/main.bib`. Placeholder citation keys appear as `[@key]` and will be converted to `\citep{key}` on LaTeX port.
> **Silver-label framing** is load-bearing across §1, §4, §6, §7. Edits in any of these sections must keep this framing consistent.

---

## Authors

Łukasz Augustyniak\*, [TODO: co-authors with verified OpenReview profiles]
Wrocław University of Science and Technology
\* corresponding: lukasz.augustyniak@pwr.edu.pl

## Abstract

Civil-law jurisdictions remain severely underrepresented in legal NLP resources, where existing corpora are dominated by English-language common-law data and rarely include analytical structure beyond raw text. We release **JuDDGES-pl** (`https://huggingface.co/datasets/JuDDGES/juddges-pl`), a single large-scale, analytically enriched Polish court judgment corpus spanning the two disjoint branches of the Polish judiciary: a common-courts subset covering the Supreme Court, courts of appeal, regional and district courts, and an administrative-courts subset covering the Supreme Administrative Court and lower administrative courts. The corpus comprises hundreds of thousands of judgments across multiple years, each augmented with LLM-extracted analytical fields — structured factual state, legal state, summaries, theses, statutory bases, and keywords — generated through a uniform Google Gemini 2.5 Pro extraction pipeline, alongside court-hierarchy, chamber, and procedural metadata preserved from the source portals.

Our contribution is curatorial and infrastructural. We describe the acquisition, normalization, and large-scale LLM-based enrichment pipeline that converts heterogeneous court-publication formats into a uniform, analysis-ready resource; we characterize the corpus through descriptive analyses of coverage, document structure, citation patterns, court hierarchy, extraction-field statistics, and pseudonymization fidelity; and we document the resource using Croissant metadata with Responsible AI fields and an accompanying evaluation card. **We treat LLM-extracted fields explicitly as silver labels** — useful for analysis, model pretraining, and weak-supervision pipelines, but **not as gold annotations for benchmarking** — and document this distinction throughout.

JuDDGES-pl is designed to enable downstream evaluation of civil-law legal NLP — including structured information extraction, statute-grounded reasoning, cross-branch generalization between common and administrative courts, longitudinal analysis of judicial language, and fairness audits in jurisdictions outside the Anglophone common-law mainstream — without itself proposing new tasks or model comparisons. The corpus is openly released on Hugging Face under a permissive license, alongside the reproducible enrichment pipeline and validated Croissant metadata.

---

## 1. Introduction

**Goal of the section:** establish (i) the civil-law gap, (ii) what JuDDGES-pl is, (iii) what counts as our contribution, (iv) the silver-label framing as an explicit design commitment.

### 1.1 The civil-law gap in legal NLP

- LegalBench [@guhaLegalbenchCollaborativelyBuilt2023a], LexGLUE [@chalkidis-etal-2022-lexglue], CaseHOLD [@zheng2021does] dominate legal NLP evaluation; all are English-language and built almost exclusively on common-law sources.
- Civil-law systems differ substantively: judgments cite codified statutes rather than precedents; document structure is fixed (sentencja, uzasadnienie); courts are organized into separate hierarchies (common vs administrative) with disjoint procedural codes.
- Polish legal NLP has prior resources (PolEval, Bielik corpora, JuDDGES NeurIPS 2024 [@augustyniak2024juddges]), but none combines (a) two judicial branches in a unified release, (b) LLM-derived analytical structure at scale, (c) NeurIPS-grade evaluation documentation.

### 1.2 What we release

- **JuDDGES-pl** — unified Hugging Face dataset (`JuDDGES/juddges-pl`) with two configs: `pl-court` (common courts) and `pl-nsa` (administrative courts).
- **Hundreds of thousands** of judgments; multi-year coverage; documents preserved verbatim with court-published anonymization intact.
- **LLM-extracted analytical layer** generated uniformly with Google Gemini 2.5 Pro: factual state, legal state, summaries, theses, statutory bases, keywords.
- **Croissant metadata** (validated) with full Responsible AI fields; reproducible extraction pipeline.

### 1.3 Contributions

1. **Unified civil-law corpus** spanning two disjoint branches of Polish judiciary, addressing the common-law / Anglophone bias of legal NLP resources.
2. **Large-scale LLM-based enrichment** with documented prompts, schemas, model version, and per-field statistics.
3. **Silver-label framing as explicit design commitment** — see §1.4.
4. **Documentation infrastructure**: Datasheet, Evaluation Card, Croissant + RAI metadata, reproducibility artifacts.
5. **Descriptive characterization** of coverage, structure, citation patterns, and pseudonymization fidelity.

### 1.4 Silver labels by design — not by accident

> ⚠️ **Design commitment.** The analytical fields released with JuDDGES-pl are **silver labels**: produced by a single LLM under uniform prompting, without per-document human review. We make this commitment explicit because the alternative — calling LLM outputs "annotations" without qualification — has produced a wave of resources whose quality is silently overstated and whose downstream benchmark numbers are not reproducible across model versions.

- We do not call our fields *annotations* without qualification.
- We do not propose JuDDGES-pl as a benchmark with held-out test sets.
- We document expected error modes (§4.4, §7.1) and provide pointers to gold-label subsets in the JuDDGES family (e.g., `pl-swiss-franc-loans`) for tasks requiring validated labels.

### 1.5 Roadmap

§2 reviews related resources; §3 describes acquisition and licensing; §4 details the enrichment pipeline; §5 characterizes the corpora; §6 presents the Datasheet and Evaluation Card; §7 enumerates limitations and intended use; §8 covers release and infrastructure.

---

## 2. Related work

**Goal of the section:** position JuDDGES-pl against (a) existing legal NLP benchmarks and corpora, (b) civil-law / non-English legal resources, (c) the literature on LLM-as-annotator and silver labels.

### 2.1 Legal NLP benchmarks and resources

- LegalBench [@guhaLegalbenchCollaborativelyBuilt2023a]: 162 tasks, common-law focus, English.
- LexGLUE [@chalkidis-etal-2022-lexglue]: 7 tasks across EU + US/UK legal text; limited Polish.
- CaseHOLD [@zheng2021does], CUAD [@hendrycks2021cuad], BillSum [@kornilova2019billsum]: domain-specific English-language tasks.
- Multi-jurisdictional resources [TODO: identify 2-3 most relevant].

### 2.2 Polish legal NLP

- JuDDGES NeurIPS 2024 [@augustyniak2024juddges]: bilingual PL/UK judgment dataset; precursor.
- PolEval shared tasks (legal subset) [TODO: cite specific years].
- Bielik corpora [TODO: cite Bielik dataset paper / model paper].
- PLLuM corpora [TODO: cite].

### 2.3 LLM-as-annotator, weak supervision, silver labels

- LLM annotation studies [@gilardi2023chatgpt; TODO: add 2-3 contemporary citations on LLM annotation reliability].
- Weak supervision frameworks [@ratner2017snorkel].
- Critiques of LLM-derived labels: model-version dependence, prompt sensitivity, distributional bias [TODO: cite].

### 2.4 Documentation standards for ML datasets

- Datasheets for Datasets [@gebru2021datasheets].
- Data Cards / Model Cards [@pushkarna2022data; @mitchell2019model].
- Croissant [@akhtar2024croissant] and the NeurIPS Responsible AI extension.

---

## 3. Acquisition & licensing

**Goal of the section:** describe data sources, copyright basis, GDPR / privacy posture, and version pinning.

### 3.1 Source portals

- **Common courts** — `orzeczenia.ms.gov.pl`. Official API; XML payloads. Acquired up to [DATE]; oldest record [DATE].
- **Administrative courts** — `orzeczenia.nsa.gov.pl`. No API; HTML scraping with respectful rate limits and re-acquisition policy for retroactively-edited records (cf. JuDDGES 2024 §3 [@augustyniak2024juddges] for full procedure).

### 3.2 Copyright and licensing

- Polish copyright law, Art. 4 pkt 2 of the Ustawa o prawie autorskim i prawach pokrewnych: court judgments (*orzeczenia sądów*) are *dokumenty urzędowe* and not subject to copyright.
- Source raw text is therefore public-domain.
- Our **enrichment layer** (LLM-extracted fields, structured metadata, Croissant) is the authors' contribution and released under **CC BY 4.0**.

### 3.3 Privacy and GDPR / RODO

- Court-published judgments are pseudonymized at source (names of natural persons → initials/codes; sensitive data redacted).
- Names of public officials, judges, and corporate parties are typically retained as legally permitted.
- We perform no additional de-anonymization. Users redistributing derivatives must comply with RODO and must not attempt re-identification.

### 3.4 Versioning

- Snapshot date pinned: [DATE].
- HF dataset versions correspond to commit hashes; Croissant `version` field tracks releases.

---

## 4. Enrichment pipeline

**Goal of the section:** describe the LLM extraction pipeline rigorously enough that it is reproducible, including version pinning of the model. **This is the methodological core of the paper.**

### 4.1 Motivation: from raw text to analysis-ready

- Raw judgment text alone is hard to evaluate over: documents are long, structurally heterogeneous, and use specialized language.
- Two prior strategies have limitations: (a) human annotation — does not scale to millions of judgments; (b) rule-based extraction — brittle across courts and time.
- LLM-based extraction with uniform prompting trades exactness for scale, with known failure modes that we document.

### 4.2 Model and prompting

- **Model:** Google Gemini 2.5 Pro, accessed via the Google AI API. Version pinned at [exact API version string] for reproducibility.
- **Prompt design:** schema-conditioned extraction; one prompt per field family; few-shot examples drawn from a held-out development set.
- **Inference:** batched calls; deterministic decoding (`temperature=0`) where supported; full prompts in Appendix A.

### 4.3 Extracted fields

| Field | Type | Source |
|---|---|---|
| `factual_state` | string | LLM-extracted (Gemini 2.5 Pro) |
| `legal_state` | string | LLM-extracted |
| `extracted_summary` | string | LLM-extracted |
| `extracted_thesis` | string | LLM-extracted |
| `extracted_legal_bases` | list of statute references | LLM-extracted, regex-validated |
| `extracted_keywords` | list of strings | LLM-extracted |
| `extracted_title` | string | LLM-extracted |
| `extracted_date_issued` | date | LLM-extracted, parsed |
| `court_name`, `judges`, `signature`, `date`, `legalBases`, `themePhrases`, … | various | preserved from source portal |

### 4.4 Quality, cost, and runtime

- Total compute: [TODO] Gemini API calls; cost: [TODO] USD; wall-clock: [TODO].
- Sample-level inspection: [TODO: N] documents inspected by a Polish-speaking lawyer; observed error modes (truncated reasoning sections; missed citations to repealed statutes; over-summarization for short procedural judgments).
- We deliberately do **not** report aggregate accuracy: there is no validated gold reference at corpus scale, and reporting LLM-vs-LLM agreement would obscure the silver-label nature of the resource.

### 4.5 Silver labels: what this means in practice

> ⚠️ **Silver-label commitment.** The analytical fields above are produced uniformly by a single LLM. They have not been human-validated at corpus scale. Users:
>
> - **may** use them for exploratory analysis, descriptive statistics, model pretraining, weak supervision, retrieval indexing;
> - **must not** use them as gold labels for benchmarking, regulatory claims, or downstream legal decisions;
> - **must** cite the model version when reporting analyses, since LLM behavior shifts across model releases.
>
> A separate validated subset (`JuDDGES/pl-swiss-franc-loans`) provides gold annotations for tasks requiring high-fidelity labels.

---

## 5. Corpus characterization

**Goal of the section:** descriptive statistics that justify the claim "evaluation-relevant" without running models.

### 5.1 Coverage

- Total judgments per config (pl-court, pl-nsa).
- Distribution by court level (Supreme Court / appellate / regional / district; NSA / WSA).
- Distribution by year (figure: timeline plot).
- Distribution by case type / procedural category.

### 5.2 Document structure

- Length distribution (tokens, sections).
- Section-presence statistics (sentencja, uzasadnienie, dissent, glosa).
- Sentencja vs. uzasadnienie length ratio.

### 5.3 Citation patterns

- Statute-citation graph (figure: top 30 cited statutes).
- Case-citation patterns (rare in civil law; quantify).
- Cross-branch citation behavior (do administrative courts cite common-court judgments? rarely; quantify).

### 5.4 Extraction-field statistics

- Length distributions for each LLM-extracted field.
- Missingness / refusal rates per field.
- Deduplication-aware diversity metrics for `extracted_keywords`.

### 5.5 Pseudonymization fidelity

- Audit on a stratified sample: presence of personal-identifier patterns (PESEL regex, address patterns, full names).
- Variance across courts and years.
- Caveats: not a privacy guarantee, only an empirical characterization.

### 5.6 Cross-branch comparison

- Token-level vocabulary overlap between common and administrative subsets.
- Statute-citation overlap.
- Topic coverage by case type.

---

## 6. Datasheet and Evaluation Card

**Goal of the section:** condensed Datasheet [@gebru2021datasheets] + an Evaluation Card declaring what evaluative claims this resource is designed to support.

### 6.1 Datasheet

- Motivation, composition, collection, preprocessing, uses, distribution, maintenance — full template in Appendix B.

### 6.2 Evaluation Card

| Claim type | Supported? | Notes |
|---|---|---|
| Pretraining / continued pretraining of legal LMs in Polish | ✅ | Silver labels acceptable; raw text high-fidelity. |
| Weak-supervision training of extraction models | ✅ | Silver labels are the intended use. |
| Descriptive / longitudinal legal analytics | ✅ | Acknowledge model-version drift. |
| Cross-branch generalization studies | ✅ | Two disjoint branches enable this. |
| **Benchmarking** with held-out test sets on LLM-extracted fields | ❌ | Use validated subsets instead. |
| Privacy guarantees beyond court-applied pseudonymization | ❌ | Document anonymization is heterogeneous. |
| Individual legal decision-making | ❌ | Out of scope; not legal advice. |

### 6.3 Responsible AI metadata

- All Croissant RAI fields populated; see §8 and the released `juddges-pl.json`.

---

## 7. Limitations and intended use

> ⚠️ **This section is the silver-label commitment in operational form. Reviewers and downstream users should read it before depending on the resource.**

### 7.1 Limitations

- **Single-jurisdiction.** Poland only; no cross-jurisdictional claims without additional resources.
- **LLM-derived analytical fields are silver labels.** Not human-validated at corpus scale; subject to model-version drift; aggregate accuracy is intentionally not reported (see §4.4).
- **Court-applied pseudonymization is heterogeneous** across courts and time and does not constitute a uniform privacy guarantee.
- **Publication bias.** Only published judgments are included; selection by publishers is non-random.
- **Temporal coverage** is denser in recent years; pre-2000 records are sparse.
- **Monolingual.** Polish only; no machine translation included.

### 7.2 Intended uses

- Pretraining, weak supervision, retrieval indexing, descriptive analytics, fairness audits, civil-law NLP research, civic-tech tooling under appropriate disclaimers.

### 7.3 Out-of-scope uses

- Benchmarking on LLM-derived labels without independent validation.
- Privacy-sensitive applications without additional anonymization.
- Individual legal advice, regulatory decisions, or any high-stakes individual outcome.

### 7.4 Model-version disclosure

- Users analyzing extracted fields **must** cite the Gemini model version recorded in §4.2 and the dataset version, since LLM outputs are model-specific.

---

## 8. Release and infrastructure

### 8.1 Hugging Face release

- `JuDDGES/juddges-pl` — single unified release with two configs (`pl-court` and `pl-nsa`).
- Public access; no gating; long-term hosting commitment by Wrocław University of Science and Technology.
- Earlier component datasets (`JuDDGES/pl-court-raw-enriched`, `JuDDGES/pl-nsa-enriched`) served as upstream sources during pipeline development and remain available for provenance, but the **canonical release for this paper is `JuDDGES/juddges-pl`**.

### 8.2 Croissant metadata

- Validated `juddges-pl.json` (also per-component fallbacks) with full Responsible AI fields:
  `dataCollection`, `dataCollectionType`, `dataCollectionRawData`, `dataPreprocessingProtocol`, `dataAnnotationProtocol`, `dataAnnotationPlatform`, `dataAnnotationAnalysis`, `annotationsPerItem`, `annotatorDemographics`, `machineAnnotationTools`, `dataUseCases`, `dataBiases`, `personalSensitiveInformation`, `dataLimitations`, `dataReleaseMaintenancePlan`, `dataSocialImpact`.

### 8.3 Reproducibility artifacts

- Extraction prompts and field schemas (Appendix A).
- Croissant generator (`croissant/generate_croissant.py`).
- EDA notebooks (Appendix C; supplementary ZIP).

### 8.4 License

- **CC BY 4.0** for the enrichment layer; raw judgment text is public-domain under Polish copyright law (§3.2).

---

## 9. Conclusion

We presented JuDDGES-pl, a large-scale enriched Polish judgment corpus addressing the civil-law gap in legal NLP resources. By combining two disjoint judicial branches, applying a uniform LLM-based enrichment pipeline, and committing to **silver-label framing** as a first-class design choice, we offer a resource whose evaluative role is documented as carefully as its content. We invite the community to build gold-label subsets, validation studies, and downstream benchmarks atop this foundation.

---

## References

Use `shared/bibliography/main.bib`. New keys to add (verify existence first):

- `guha2023legalbench`, `chalkin2022lexglue`, `zheng2021does`, `hendrycks2021cuad`, `kornilova2019billsum`
- `augustyniak2024juddges` (NeurIPS 2024 paper — confirm key)
- `gilardi2023chatgpt`, `ratner2017snorkel`
- `gebru2021datasheets`, `pushkarna2022data`, `mitchell2019model`, `akhtar2024croissant`
- TODO: PolEval / Bielik / PLLuM citation keys

---

## Appendix

### A. Extraction prompts and field schemas

For each of `factual_state`, `legal_state`, `extracted_summary`, `extracted_thesis`, `extracted_legal_bases`, `extracted_keywords`, `extracted_title`, `extracted_date_issued`: full prompt text, few-shot examples (drawn from held-out development set), and parser/post-processor.

### B. Datasheet for Datasets (full template)

Following Gebru et al. [@gebru2021datasheets]: motivation, composition, collection process, preprocessing/cleaning/labeling, uses, distribution, maintenance.

### C. Sample records

[TODO: include 2-3 fully-anonymized example records per config, showing all fields side-by-side.]

### D. Detailed statistics tables

[TODO: full tables for §5 figures.]

### E. NeurIPS 2026 Paper Checklist

[Mandatory. Use the official LaTeX template from the NeurIPS 2026 site. Place inline at end of PDF before submission. Desk-reject if missing.]

---

## Open TODOs

- [ ] Confirm/insert exact dataset sizes (replace "hundreds of thousands" once final).
- [ ] Confirm date ranges per config.
- [ ] Confirm pinned Gemini API version string.
- [ ] Compute and insert all §5 statistics; render figures.
- [ ] Perform §5.5 pseudonymization audit on a stratified sample; insert numbers.
- [ ] Identify human-validated comparator (likely a sampled subset reviewed by a Polish lawyer); decide whether to include validation results in §4.4 (recommended) or only point to `pl-swiss-franc-loans`.
- [ ] Add 2-3 contemporary citations on LLM-as-annotator reliability (§2.3).
- [ ] Confirm bib keys; add missing entries to `shared/bibliography/main.bib`.
- [ ] Port to LaTeX with `neurips_2026.sty`; split into `sections/*.tex` mirroring the JuDDGES 2024 layout.
- [ ] Generate and insert NeurIPS 2026 Paper Checklist (mandatory).
- [ ] Re-run Croissant generator after `JuDDGES/juddges-pl` HF publish.
