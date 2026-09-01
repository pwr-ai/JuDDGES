#!/usr/bin/env python
"""Experiment 4: Cross-Model Agreement (Task 2).

Compares extraction consistency between GPT-4.1 and Gemini 2.5 Pro
on the same documents using Schema B (16 general-purpose fields).

Workflow:
  1. Sample N documents from the enriched dataset (Gemini extractions exist)
  2. Run GPT-4.1 extraction on same documents
  3. Compute per-field agreement (exact match, ROUGE, Cohen's kappa)
  4. Identify fields with high vs low agreement

Usage:
    python scripts/neurips_experiments/exp4_cross_model_agreement.py \
        --enriched-dataset JuDDGES/pl-court-raw-enriched \
        --sample-size 1000 \
        --output results/neurips/exp4_cross_model_agreement.json
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import typer
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger

from juddges.evals.metrics import (
    evaluate_date,
    evaluate_enum,
    evaluate_list_greedy,
    evaluate_string_rouge,
)
from juddges.utils.misc import save_json

load_dotenv()

# Schema B fields (from juddges/extraction/schema.py)
SCHEMA_B_FIELDS = {
    "document_number": {"type": "string"},
    "document_type": {"type": "enum", "choices": ["judgment", "tax_interpretation", "legal_act"]},
    "title": {"type": "string"},
    "date_issued": {"type": "date"},
    "summary": {"type": "string"},
    "thesis": {"type": "string"},
    "keywords": {"type": "list"},
    "factual_state": {"type": "string"},
    "legal_state": {"type": "string"},
    "outcome": {"type": "string"},  # JSON string, compare as text
    "legal_references": {"type": "string"},  # JSON string
    "legal_concepts": {"type": "string"},  # JSON string
    "parties": {"type": "string"},  # JSON string
    "legal_analysis": {"type": "string"},  # JSON string
    "judgment_specific": {"type": "string"},  # JSON string
    "tax_interpretation_specific": {"type": "string"},  # JSON string
}

# Map enriched dataset column names to Schema B field names
ENRICHED_TO_SCHEMA_B = {
    "extracted_title": "title",
    "extracted_date_issued": "date_issued",
    "extracted_summary": "summary",
    "extracted_thesis": "thesis",
    "extracted_keywords": "keywords",
    "factual_state": "factual_state",
    "legal_state": "legal_state",
    "extracted_outcome": "outcome",
    "extracted_legal_references": "legal_references",
    "extracted_legal_concepts": "legal_concepts",
    "extracted_parties": "parties",
    "extracted_legal_analysis": "legal_analysis",
    "extracted_judgment_specific": "judgment_specific",
    "extracted_tax_interpretation_specific": "tax_interpretation_specific",
}


def extract_with_gpt41(
    text: str,
    api_key: str,
    model: str = "gpt-4.1",
) -> dict[str, Any]:
    """Extract Schema B fields from text using GPT-4.1."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    system_prompt = """You are a legal document information extraction system.
Extract the following fields from the Polish legal document.
Return ONLY valid JSON with these exact keys:
document_number, document_type, title, date_issued, summary, thesis,
keywords, factual_state, legal_state, outcome, legal_references,
legal_concepts, parties, legal_analysis, judgment_specific,
tax_interpretation_specific.

For document_type use exactly one of: judgment, tax_interpretation, legal_act.
For date_issued use YYYY-MM-DD format.
For keywords return a JSON array of strings.
For complex fields (outcome, legal_references, etc.) return JSON objects/arrays."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract structured information from this legal document:\n\n{text[:15000]}"},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


def compute_field_agreement(
    gemini_value: Any,
    gpt_value: Any,
    field_type: str,
) -> dict[str, float]:
    """Compute agreement between two model outputs for a single field."""
    if gemini_value is None and gpt_value is None:
        return {"agreement": 1.0, "both_null": True}
    if gemini_value is None or gpt_value is None:
        return {"agreement": 0.0, "one_null": True}

    # Convert to strings for comparison
    gem_str = str(gemini_value) if not isinstance(gemini_value, str) else gemini_value
    gpt_str = str(gpt_value) if not isinstance(gpt_value, str) else gpt_value

    if field_type == "date":
        result = evaluate_date(gpt_str, gem_str)
        return {"agreement": float(result["match"]), "metric": "exact_date"}

    elif field_type == "enum":
        exact = 1.0 if gem_str.strip().lower() == gpt_str.strip().lower() else 0.0
        return {"agreement": exact, "metric": "exact_match"}

    elif field_type == "list":
        # Parse JSON lists
        try:
            gem_list = json.loads(gem_str) if isinstance(gem_str, str) else gem_str
            gpt_list = json.loads(gpt_str) if isinstance(gpt_str, str) else gpt_str
            if not isinstance(gem_list, list):
                gem_list = [gem_list]
            if not isinstance(gpt_list, list):
                gpt_list = [gpt_list]
            result = evaluate_list_greedy(gpt_list, gem_list)
            return {"agreement": result["f1"], "metric": "list_f1"}
        except (json.JSONDecodeError, TypeError):
            return {"agreement": 0.0, "metric": "parse_error"}

    else:  # string
        if len(gem_str) < 3 and len(gpt_str) < 3:
            exact = 1.0 if gem_str.strip() == gpt_str.strip() else 0.0
            return {"agreement": exact, "metric": "exact_short"}

        result = evaluate_string_rouge(gpt_str, gem_str)
        return {"agreement": result["rougeL"], "metric": "rougeL"}


def main(
    enriched_dataset: str = typer.Option(
        "JuDDGES/pl-court-raw-enriched",
        help="HuggingFace enriched dataset name",
    ),
    sample_size: int = typer.Option(
        1000, help="Number of documents to sample"
    ),
    gpt_model: str = typer.Option(
        "gpt-4.1", help="GPT model name"
    ),
    skip_extraction: bool = typer.Option(
        False, help="Skip GPT extraction, load from cache"
    ),
    cache_dir: Path = typer.Option(
        "results/neurips/exp4_cache/",
        help="Directory to cache GPT extractions",
    ),
    output: Path = typer.Option(
        "results/neurips/exp4_cross_model_agreement.json",
        help="Output path",
    ),
):
    """Compare GPT-4.1 vs Gemini 2.5 Pro extraction agreement on Schema B."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not skip_extraction:
        logger.error("OPENAI_API_KEY not set. Use --skip-extraction to use cached results.")
        raise typer.Exit(1)

    # Load enriched dataset (contains Gemini extractions)
    logger.info(f"Loading dataset {enriched_dataset}...")
    ds = load_dataset(enriched_dataset, split="train")

    # Filter to documents with Gemini extractions
    has_extraction = [
        i for i in range(len(ds))
        if ds[i].get("factual_state") is not None
        or ds[i].get("extracted_summary") is not None
    ]
    logger.info(f"Documents with Gemini extractions: {len(has_extraction)}")

    # Sample
    import random
    random.seed(42)
    sample_indices = random.sample(has_extraction, min(sample_size, len(has_extraction)))
    logger.info(f"Sampled {len(sample_indices)} documents")

    # Extract Gemini values
    gemini_extractions = []
    texts = []
    for idx in sample_indices:
        row = ds[idx]
        gemini_ext = {}
        for enriched_col, schema_field in ENRICHED_TO_SCHEMA_B.items():
            gemini_ext[schema_field] = row.get(enriched_col)
        # Add fields from original columns
        gemini_ext["document_number"] = row.get("signature") or row.get("_id")
        gemini_ext["document_type"] = row.get("document_type", "judgment")
        gemini_extractions.append(gemini_ext)
        texts.append(row.get("text", row.get("content", "")))

    # GPT-4.1 extraction (or load from cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "gpt_extractions.json"

    if skip_extraction and cache_file.exists():
        logger.info(f"Loading cached GPT extractions from {cache_file}")
        with open(cache_file) as f:
            gpt_extractions = json.load(f)
    else:
        logger.info(f"Running GPT-4.1 extraction on {len(texts)} documents...")
        gpt_extractions = []
        for i, text in enumerate(texts):
            if i % 50 == 0:
                logger.info(f"  Extracting {i}/{len(texts)}...")
            try:
                ext = extract_with_gpt41(text, api_key, gpt_model)
                gpt_extractions.append(ext)
            except Exception as e:
                logger.error(f"  Error on doc {i}: {e}")
                gpt_extractions.append({})

        # Cache results
        with open(cache_file, "w") as f:
            json.dump(gpt_extractions, f, ensure_ascii=False, indent=2)
        logger.info(f"Cached GPT extractions to {cache_file}")

    # Compute agreement
    per_field_agreements = defaultdict(list)
    per_doc_agreements = []

    for i in range(min(len(gemini_extractions), len(gpt_extractions))):
        gemini = gemini_extractions[i]
        gpt = gpt_extractions[i]
        doc_scores = {}

        for field, field_props in SCHEMA_B_FIELDS.items():
            gem_val = gemini.get(field)
            gpt_val = gpt.get(field)
            agreement = compute_field_agreement(gem_val, gpt_val, field_props["type"])
            per_field_agreements[field].append(agreement["agreement"])
            doc_scores[field] = agreement["agreement"]

        per_doc_agreements.append(mean(doc_scores.values()) if doc_scores else 0)

    # Aggregate
    field_summary = {}
    for field, agreements in per_field_agreements.items():
        field_summary[field] = {
            "mean_agreement": round(mean(agreements), 4),
            "high_agreement_pct": round(sum(1 for a in agreements if a > 0.8) / len(agreements) * 100, 1),
            "exact_match_pct": round(sum(1 for a in agreements if a == 1.0) / len(agreements) * 100, 1),
            "n_compared": len(agreements),
            "field_type": SCHEMA_B_FIELDS[field]["type"],
        }

    # Rank by agreement
    ranked = sorted(field_summary.items(), key=lambda x: x[1]["mean_agreement"], reverse=True)

    results = {
        "experiment": "cross_model_agreement",
        "models": ["gemini-2.5-pro", gpt_model],
        "dataset": enriched_dataset,
        "sample_size": len(sample_indices),
        "overall_agreement": round(mean(per_doc_agreements), 4) if per_doc_agreements else 0,
        "per_field_summary": field_summary,
        "field_ranking": [{"field": f, **data} for f, data in ranked],
        "high_agreement_fields": [f for f, data in ranked if data["mean_agreement"] > 0.8],
        "low_agreement_fields": [f for f, data in ranked if data["mean_agreement"] < 0.5],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, output, ensure_ascii=False)
    logger.info(f"Results saved to {output}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"CROSS-MODEL AGREEMENT: Gemini 2.5 Pro vs {gpt_model}")
    print(f"{'='*70}")
    print(f"Documents compared: {len(per_doc_agreements)}")
    print(f"Overall agreement: {results['overall_agreement']:.4f}")
    print(f"\n{'Field':<35} {'Agreement':>10} {'Exact%':>8} {'Type':>10}")
    print(f"{'-'*65}")
    for field, data in ranked:
        print(f"{field:<35} {data['mean_agreement']:>10.4f} {data['exact_match_pct']:>7.1f}% {data['field_type']:>10}")


if __name__ == "__main__":
    typer.run(main)
