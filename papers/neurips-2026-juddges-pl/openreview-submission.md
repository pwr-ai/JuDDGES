# NeurIPS 2026 E&D — OpenReview Submission Form (JuDDGES-pl)

**Use:** copy/paste each field directly into the OpenReview form.
**Deadlines (UTC):** Abstract registration May 5 11:59 AM · Full paper May 7 11:59 AM.
**Today:** 2026-05-05.

---

## 1. Title

```
JuDDGES-pl: Hundreds of Thousands of Structurally Enriched Polish Judgments from Common and Administrative Courts
```

Backup (shorter, if length is an issue):

```
JuDDGES-pl: Large-Scale Enriched Polish Court Judgment Corpora for Civil-Law Legal NLP
```

## 2. Authors

- Lukasz Augustyniak (already prefilled)
- TODO: add co-authors with verified OpenReview profiles before submission

## 3. Keywords (comma-separated)

```
legal NLP, Polish, civil law, court judgments, dataset, LLM-based annotation, silver labels, structured information extraction, low-resource languages, evaluation resources
```

## 4. TL;DR (one sentence)

```
Hundreds of thousands of Polish court judgments from common and administrative courts, enriched with LLM-extracted factual and legal state fields via Google Gemini 2.5 Pro, openly released to enable civil-law legal NLP evaluation.
```

## 5. Abstract

```
Civil-law jurisdictions remain severely underrepresented in legal NLP resources, where existing corpora are dominated by English-language common-law data and rarely include analytical structure beyond raw text. We release JuDDGES-pl (https://huggingface.co/datasets/JuDDGES/juddges-pl), a single large-scale, analytically enriched Polish court judgment corpus spanning the two disjoint branches of the Polish judiciary: a common-courts subset covering the Supreme Court, courts of appeal, regional and district courts, and an administrative-courts subset covering the Supreme Administrative Court and lower administrative courts. The corpus comprises hundreds of thousands of judgments across multiple years, each augmented with LLM-extracted analytical fields — structured factual state, legal state, summaries, theses, statutory bases, and keywords — generated through a uniform Google Gemini 2.5 Pro extraction pipeline, alongside court-hierarchy, chamber, and procedural metadata preserved from the source portals.

Our contribution is curatorial and infrastructural. We describe the acquisition, normalization, and large-scale LLM-based enrichment pipeline that converts heterogeneous court-publication formats into a uniform, analysis-ready resource; we characterize the corpus through descriptive analyses of coverage, document structure, citation patterns, court hierarchy, extraction-field statistics, and pseudonymization fidelity; and we document the resource using Croissant metadata with Responsible AI fields and an accompanying evaluation card. We treat LLM-extracted fields explicitly as silver labels — useful for analysis, model pretraining, and weak-supervision pipelines, but not as gold annotations for benchmarking — and document this distinction throughout.

JuDDGES-pl is designed to enable downstream evaluation of civil-law legal NLP — including structured information extraction, statute-grounded reasoning, cross-branch generalization between common and administrative courts, longitudinal analysis of judicial language, and fairness audits in jurisdictions outside the Anglophone common-law mainstream — without itself proposing new tasks or model comparisons. The corpus is openly released on Hugging Face under a permissive license, alongside the reproducible enrichment pipeline and validated Croissant metadata.
```

## 6. Review Mode

```
Single-blind (allowed ONLY for dataset-centered submissions)
```

**Justification:** the JuDDGES dataset family and Hugging Face organization
(`huggingface.co/JuDDGES`) are publicly attributed to the authors and to prior
peer-reviewed work (NeurIPS 2024 JuDDGES). The corpora cannot be released
anonymously without breaking dataset provenance and Croissant metadata. The
track explicitly permits single-blind for this scenario.

## 7. Dataset Submission

```
[x] This submission includes a dataset.
```

## 8. Dataset URL (primary)

Primary unified release (will be published on HF before paper-submission
deadline May 7; combines common-court and administrative-court material
as named configs / splits inside a single dataset):

```
https://huggingface.co/datasets/JuDDGES/juddges-pl
```

Org landing page:

```
https://huggingface.co/JuDDGES
```

⚠️ TODO before May 7: publish `JuDDGES/juddges-pl` on Hugging Face with
two configs (suggested names: `pl-court` and `pl-nsa`), README, and
license. The dataset **must be accessible to anonymous reviewers at
submission time** — non-compliance is grounds for desk rejection per
E&D track rules.

## 9. Dataset Large URL (sample for >4GB datasets)

Verify the on-disk Parquet size of `JuDDGES/juddges-pl` before the
deadline; if it exceeds 4GB, host a stratified sample:

```
https://huggingface.co/datasets/JuDDGES/juddges-pl-sample
```

TODO before May 7: confirm size via `huggingface-cli` or HF API; create
a stratified sample (e.g., 1k docs stratified by court level × year)
only if needed; add a sampling-procedure README to the sample dataset.

## 10. Croissant File

PENDING until `JuDDGES/juddges-pl` is published on HF.

```
papers/neurips-2026-juddges-pl/croissant/juddges-pl.json              (TODO: generate after HF publish)
papers/neurips-2026-juddges-pl/croissant/JuDDGES-pl-croissant.zip     <- upload this
```

After publishing `JuDDGES/juddges-pl` on Hugging Face, run:
```
cd papers/neurips-2026-juddges-pl/croissant
python generate_croissant.py
zip -j JuDDGES-pl-croissant.zip juddges-pl.json
```

The Croissant file must validate cleanly under `mlcroissant`
(target: 0 errors / 0 warnings at metadata level) and include the full
set of Responsible AI fields:
dataCollection, dataCollectionType, dataCollectionRawData,
dataPreprocessingProtocol, dataAnnotationProtocol, dataAnnotationPlatform,
dataAnnotationAnalysis, annotationsPerItem, annotatorDemographics,
machineAnnotationTools, dataUseCases, dataBiases,
personalSensitiveInformation, dataLimitations, dataReleaseMaintenancePlan,
dataSocialImpact.


## 11. Code URL

For a curatorial / dataset-centered submission with descriptive analyses
only (no model training), code release is **optional**. Two options:

- **(Recommended)** Provide the enrichment pipeline repo (anonymized if
  possible, or single-blind link otherwise). Strengthens reproducibility
  claims even though not strictly required.
- **(Minimal)** Skip code URL; fill the justification field below.

## 12. Code Submission Justification (if no code URL)

```
Our contribution is curatorial and analytical: we release a single
unified, enriched Polish court judgment corpus — produced by a uniform
Google Gemini 2.5 Pro extraction pipeline applied to court-published
source materials — and characterize it through descriptive statistics
on coverage, document structure, extraction-field distributions,
citation patterns, and pseudonymization fidelity. We do not introduce
new tasks, train new models, or propose a reusable evaluation
framework. The extraction prompts and per-field schemas are released
alongside the paper and the Croissant metadata; the analyses use
standard, openly available tooling described in the paper. Per the
E&D Call for Papers, code release is optional for analytical,
empirical, conceptual, or methodological contributions of this type.
```

Recommended alternative: provide the enrichment pipeline + EDA notebooks
in an anonymized repository, since reviewers will value reproducibility
of the Gemini-based extraction even when not strictly required.

## 13. Supplementary Material

Optional. If included, package descriptive-statistics notebooks (HTML
rendered, anonymized) as a single ZIP, ≤100MB, anonymized. Path:

```
papers/neurips-2026-juddges-pl/supplementary/eda-notebooks.zip
```

## 14. Financial Support

Leave blank unless a student co-author needs it. Format: `~First_Last1`.

## 15. Reviewer Nomination

Provide the OpenReview ID of at least one author with 2+ first-author OR
5+ co-authored archival publications at venues like NeurIPS / ICLR / ICML /
ACL / EMNLP / etc.

```
~Lukasz_Augustyniak1
```

(verify the exact ID format on OpenReview profile before submitting)

## 16. Primary Area (pick up to 2)

From the dropdown, pick the closest two. Most plausible matches for this paper:

1. **Datasets and benchmarks for natural language processing** (or "NLP" subcategory)
2. **Datasets and benchmarks for applied / domain-specific ML** (legal AI)

Backup if "fairness/responsibility" subcategory exists:

3. *Fairness, accountability, transparency, and safety datasets*

## 17. Contribution Type

```
Dataset
```

(if dropdown distinguishes Dataset / Benchmark / Evaluation methodology /
Tool / Analysis — pick **Dataset**; secondary if allowed: **Analysis**)

## 18. LLM Usage

Tick all that apply. Be honest. Common selections for this kind of paper:

- Writing assistance (drafting, polishing)
- Code assistance (enrichment pipeline, EDA scripts)

## 19. Author Acknowledgements (all required)

- [x] CONTRIBUTION ACKNOWLEDGEMENT — abstract above explicitly states
  evaluative role, assumptions, and limitations; final paper will include
  a dedicated "Intended Use & Limitations" section.
- [x] CHECKLIST CONFIRMATION — TODO before May 7: include the NeurIPS 2026
  paper checklist in the PDF (use provided LaTeX template).
- [x] RESPONSIBLE REVIEWING
- [x] ACADEMIC INTEGRITY
- [x] DECLARATION

## 20. License

Suggested:

```
CC BY 4.0
```

Rationale: Polish court judgments are public-domain materials under
Polish copyright law (*orzeczenia sądów* are *dokumenty urzędowe*,
not subject to copyright per Art. 4 pkt 2 of the Polish Copyright Act).
The **enrichment layer** (LLM-extracted factual state, legal state,
summaries, theses, statutory bases produced by Google Gemini 2.5 Pro)
is the authors' contribution and can be released under **CC BY 4.0**.
Confirm consistency with the existing license declared on the JuDDGES
Hugging Face datasets before final selection.

If existing HF license is more permissive (e.g., CC0) → match it.
If more restrictive (e.g., CC BY-SA, ODbL) → match it.

---

# IMMEDIATE TODO BEFORE ABSTRACT REGISTRATION (May 5, 11:59 AM UTC)

1. Verify OpenReview profile ID for `Lukasz_Augustyniak`.
2. Add any co-authors who must appear on the submission (Primary Area,
   Contribution Type, Reviewer Nomination cannot be changed after May 4 AOE
   per the form text — confirm whether this also applies to author list).
3. Pick exact Primary Area + Contribution Type from the dropdowns.
4. Paste Title + Abstract + TL;DR + Keywords; tick Single-blind + Dataset
   Submission; tick all 5 acknowledgements.

# TODO BEFORE FULL PAPER DEADLINE (May 7, 11:59 AM UTC)

1. **Publish `JuDDGES/juddges-pl` on Hugging Face** with two configs
   (`pl-court`, `pl-nsa`), README, license, and verify it is accessible
   to anonymous reviewers. This is the hard blocker.
2. Re-run `croissant/generate_croissant.py` after publish; re-zip; upload
   updated `JuDDGES-pl-croissant.zip`.
3. Confirm dataset size (Parquet on disk); create a stratified sample
   only if `JuDDGES/juddges-pl` >4GB; upload sample-dataset URL if needed.
4. Compute final corpus statistics to replace "hundreds of thousands" with
   exact numbers in the camera-ready paper (abstract can stay qualitative).
5. Write paper sections per agreed structure:
   Intro → Related work → Acquisition & licensing → Enrichment pipeline
   (Gemini 2.5 Pro) → Corpus characterization → Datasheet / Evaluation card →
   Limitations & intended use (silver-label disclaimer) →
   Release & Croissant.
6. Include NeurIPS 2026 paper checklist in the PDF (mandatory; desk
   reject if missing).
7. Re-validate Croissant via official validator (mlcroissant or web tool).
