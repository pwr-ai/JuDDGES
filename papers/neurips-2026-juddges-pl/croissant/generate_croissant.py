"""
Generate NeurIPS 2026 E&D-compliant Croissant metadata for JuDDGES-pl corpora.

Pulls auto-generated Croissant from Hugging Face, then augments with the
Responsible AI (RAI) fields that NeurIPS 2026 Evaluations & Datasets Track
requires (https://neurips.cc/Conferences/2026/EvaluationsDatasetsHosting).

Outputs validated, ready-to-upload JSON-LD next to this script.

Run:
    python generate_croissant.py

Validate output with the official Croissant validator:
    pip install mlcroissant
    python -m mlcroissant validate --jsonld pl-court-raw-enriched.json
    python -m mlcroissant validate --jsonld pl-nsa-enriched.json
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent

DATASETS = [
    # Primary unified release. Publish on HF before paper-submission deadline.
    # Once published, this entry will produce the canonical Croissant for the paper.
    {
        "hf_id": "JuDDGES/juddges-pl",
        "out": "juddges-pl.json",
        "title": "JuDDGES-pl: Enriched Polish Court Judgment Corpus for Civil-Law Legal NLP",
        "size_band": "1M+ judgments (combined)",
        "optional": True,  # may not exist yet at script-run time
    },
    # Component sub-corpora (already public). Croissants serve as fallback if
    # the unified release is not yet published when the paper PDF is built.
    {
        "hf_id": "JuDDGES/pl-court-raw-enriched",
        "out": "pl-court-raw-enriched.json",
        "title": "JuDDGES/pl-court-raw-enriched: Polish Common-Court Judgments, Enriched",
        "size_band": "100K - 1M judgments",
    },
    {
        "hf_id": "JuDDGES/pl-nsa-enriched",
        "out": "pl-nsa-enriched.json",
        "title": "JuDDGES/pl-nsa-enriched: Polish Administrative-Court Judgments, Enriched",
        "size_band": "1M - 10M judgments",
    },
]

# License for the enrichment layer. Polish judgments themselves are public-domain
# legal acts (Art. 4 pkt 2 Ustawy o prawie autorskim — orzeczenia są dokumentami
# urzędowymi i nie podlegają prawu autorskiemu). The added structured/extracted
# annotations are released under CC BY 4.0.
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

CITE_AS = (
    "@inproceedings{juddges-pl-2026,\n"
    "  title     = {JuDDGES-pl: Large-Scale Enriched Polish Court Judgment Corpora "
    "for Civil-Law Legal NLP},\n"
    "  author    = {Augustyniak, {\\L}ukasz and others},\n"
    "  booktitle = {NeurIPS 2026 Evaluations and Datasets Track},\n"
    "  year      = {2026}\n"
    "}"
)

# ----------------------------------------------------------------------------
# Responsible AI (RAI) fields. NeurIPS 2026 E&D requires these.
# https://docs.mlcommons.org/croissant/docs/croissant-rai-spec.html
# ----------------------------------------------------------------------------

RAI_COMMON = {
    "dataCollection": (
        "Source judgments were obtained from official Polish court publication "
        "portals — the Common Courts judgment portal (orzeczenia.ms.gov.pl) for "
        "pl-court-raw-enriched, and the Central Database of Administrative Court "
        "Judgments (orzeczenia.nsa.gov.pl) for pl-nsa-enriched. Both portals "
        "publish judgments as public-domain legal acts (Art. 4 pkt 2 of the "
        "Polish Copyright Act, dokumenty urzędowe). Documents were retrieved "
        "with court-published anonymization in place; no additional crawling of "
        "non-public material was performed."
    ),
    "dataCollectionType": "Web scraping of official, public-domain judgment portals.",
    "dataCollectionRawData": (
        "Raw HTML and PDF judgment exports from the source portals, retained as "
        "intermediate artifacts and released as the upstream JuDDGES/pl-court-raw "
        "and JuDDGES/pl-nsa datasets."
    ),
    "dataPreprocessingProtocol": (
        "Documents were converted from heterogeneous publication formats (HTML, "
        "PDF, RTF) to a uniform UTF-8 plain-text representation, with "
        "document-structure parsing (sentencja, uzasadnienie, signatures), "
        "metadata normalization (court hierarchy, chamber, dates, judgment "
        "type), and statute-citation extraction. Records with empty or "
        "unparsable content were removed."
    ),
    "dataAnnotationProtocol": (
        "Beyond raw text and source metadata, the corpora include "
        "LLM-extracted analytical fields generated with Google Gemini 2.5 Pro: "
        "factual_state (a structured summary of the case facts), legal_state "
        "(the legal questions and statutory framework), extracted_summary, "
        "extracted_thesis, extracted_keywords, and extracted_legal_bases. "
        "Extraction prompts were applied uniformly across the corpus; "
        "no human re-annotation pass was performed at scale. These fields "
        "should be treated as model-generated and used accordingly."
    ),
    "dataAnnotationPlatform": "Custom batch-inference pipeline against the Google Gemini API.",
    "dataAnnotationAnalysis": (
        "LLM-extracted fields have not undergone systematic human validation "
        "at corpus scale. Users intending to evaluate downstream models on "
        "these fields should treat them as silver labels, not gold. A separate "
        "human-validated subset is available via the upstream JuDDGES family "
        "(e.g., JuDDGES/pl-swiss-franc-loans) for tasks requiring gold labels."
    ),
    "annotationsPerItem": (
        "Each judgment carries the full set of extracted analytical fields; "
        "no inter-annotator agreement applies as labels are model-generated."
    ),
    "annotatorDemographics": (
        "Not applicable: analytical fields are produced by Google Gemini 2.5 "
        "Pro rather than human annotators."
    ),
    "machineAnnotationTools": (
        "Google Gemini 2.5 Pro accessed via the Google AI API. Prompts and "
        "extraction schemas are released alongside the paper."
    ),
    "dataUseCases": (
        "Intended uses: (1) structured information extraction from legal text; "
        "(2) statute-grounded reasoning evaluation; (3) cross-branch "
        "generalization studies between common and administrative judiciary; "
        "(4) longitudinal analysis of judicial language; (5) civil-law-aware "
        "training and fine-tuning of legal language models. The corpora are "
        "designed to support — not replace — task-specific evaluation, which "
        "requires either gold-labeled subsets or human validation."
    ),
    "dataBiases": (
        "Several biases are known and documented: (1) jurisdictional bias — "
        "data covers only Poland and reflects Polish civil-law procedure; "
        "(2) coverage bias — only judgments published by the source portals "
        "are included; not all Polish judgments are publicly published, and "
        "publication is uneven across courts and case types; (3) temporal "
        "bias — coverage is denser in recent years as digital publication "
        "expanded; (4) language bias — analytical extractions may reflect the "
        "training distribution of Google Gemini 2.5 Pro; (5) anonymization "
        "bias — court-applied pseudonymization removes named entities, "
        "which can affect identity- and entity-centered evaluations."
    ),
    "personalSensitiveInformation": (
        "Source judgments are pseudonymized at the court level before "
        "publication: personal names of natural persons are replaced with "
        "initials or letter codes; sensitive personal details (addresses, "
        "PESEL, dates of birth) are redacted. Names of public officials, "
        "judges, courts, and corporate parties are typically retained. We "
        "performed no additional de-anonymization. Users must comply with "
        "GDPR (RODO) when re-releasing derivatives, and must not attempt "
        "re-identification of pseudonymized individuals."
    ),
    "dataLimitations": (
        "(1) Only Poland is represented; cross-jurisdictional claims require "
        "additional resources. (2) LLM-extracted analytical fields are not "
        "human-validated at corpus scale and may contain extraction errors. "
        "(3) Court-applied pseudonymization is heterogeneous across courts "
        "and time, and may not satisfy modern privacy guarantees uniformly. "
        "(4) The corpora reflect published judgments only; case selection by "
        "publishers is non-random. (5) No machine translation is included; "
        "the resource is monolingual Polish."
    ),
    "dataReleaseMaintenancePlan": (
        "The corpora are released openly on Hugging Face under the JuDDGES "
        "organization (https://huggingface.co/JuDDGES) and will be maintained "
        "with versioned snapshots. Errata, schema updates, and new releases "
        "will be published as new dataset versions; older versions remain "
        "accessible via the Hugging Face commit history. Issues can be filed "
        "on the dataset Community tab. The maintaining organization is "
        "Wrocław University of Science and Technology (WUST)."
    ),
    "dataSocialImpact": (
        "Positive: improves access to Polish legal information for citizens, "
        "non-profits, and researchers; reduces the common-law/Anglophone bias "
        "of legal NLP resources; enables civic-tech applications in a "
        "jurisdiction underrepresented in ML benchmarks. Risks: model-"
        "generated analytical fields may be miscited as authoritative legal "
        "summaries; users (especially non-experts) must not treat extracted "
        "fields as legal advice. The corpora are not intended for and must "
        "not be used as a basis for individual legal decision-making."
    ),
}


def fetch_hf_croissant(hf_id: str) -> dict:
    url = f"https://huggingface.co/api/datasets/{hf_id}/croissant"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def add_rai(croissant: dict, ds: dict) -> dict:
    """Add NeurIPS-required RAI fields and missing core fields."""
    croissant["name"] = ds["title"]
    croissant["license"] = LICENSE_URL
    croissant["citeAs"] = CITE_AS
    croissant["version"] = "1.0.0"
    croissant["datePublished"] = "2026-05-07"
    croissant["dateModified"] = "2026-05-07"
    croissant.setdefault("inLanguage", "pl")
    croissant.setdefault("isAccessibleForFree", True)
    croissant.setdefault(
        "publisher",
        {
            "@type": "Organization",
            "name": "Wrocław University of Science and Technology",
            "url": "https://pwr.edu.pl",
        },
    )
    # Merge RAI fields under top-level (Croissant RAI extension lives at root).
    for key, value in RAI_COMMON.items():
        croissant[key] = value
    # Sanity: keep existing keywords; ensure 'civil-law' and 'NeurIPS-2026' tags.
    kws = list(croissant.get("keywords", []))
    for tag in ("civil-law", "NeurIPS-2026", "JuDDGES-pl", "evaluation-resource"):
        if tag not in kws:
            kws.append(tag)
    croissant["keywords"] = kws
    return croissant


def main() -> None:
    skipped = []
    written = []
    for ds in DATASETS:
        print(f"Fetching {ds['hf_id']} ...")
        try:
            cr = fetch_hf_croissant(ds["hf_id"])
        except Exception as exc:
            if ds.get("optional"):
                print(f"  -> SKIP (not yet published on HF): {exc}")
                skipped.append(ds["hf_id"])
                continue
            raise
        cr = add_rai(cr, ds)
        out_path = HERE / ds["out"]
        out_path.write_text(json.dumps(cr, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> wrote {out_path} ({out_path.stat().st_size:,} bytes)")
        written.append(out_path.name)
    print()
    print("Done.")
    if skipped:
        print(f"Skipped (not yet on HF): {', '.join(skipped)}")
        print("Re-run after publishing JuDDGES/juddges-pl on Hugging Face.")
    print("Validate output:")
    print("  pip install mlcroissant  # if not present")
    for name in written:
        print(f"  python -c \"import mlcroissant as m; m.Dataset(jsonld='{name}')\"")


if __name__ == "__main__":
    main()
