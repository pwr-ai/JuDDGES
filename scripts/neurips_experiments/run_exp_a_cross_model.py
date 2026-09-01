#!/usr/bin/env python
"""EXP A: Cross-Model Agreement (Task 2).

Runs GPT-4.1 extraction with Schema B on 1K documents from
pl-court-raw-enriched (which already have Gemini 2.5 Pro extractions).
Compares GPT-4.1 vs Gemini per-field agreement.

Cost estimate: ~$28 (5.77M input + 2M output tokens)

Usage:
    PYTHONPATH=. python scripts/neurips_experiments/run_exp_a_cross_model.py
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from datasets import load_dataset
from loguru import logger
from openai import OpenAI

import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 1000
MODEL = "gpt-4.1"
OUTPUT_DIR = Path("data/experiments/neurips_results/exp_a_cross_model")
MAX_RETRIES = 3
BATCH_LOG_EVERY = 50


def build_prompt(text: str, schema_fields: dict) -> str:
    """Build extraction prompt with Schema B."""
    fields_desc = "\n".join(
        f"- {name}: {desc[:200]}" for name, desc in schema_fields.items()
    )
    return f"""Extract the following fields from this Polish legal document.
Return ONLY a valid JSON object with these fields:

{fields_desc}

RULES:
- Extract information ONLY from the document text
- Use Polish terminology
- Dates in YYYY-MM-DD format
- Empty strings for missing string fields
- Empty arrays for missing list fields
- null for missing object fields

DOCUMENT:
{text[:12000]}

Return ONLY valid JSON:"""


def extract_with_gpt(client: OpenAI, text: str, schema_fields: dict) -> dict | None:
    """Extract fields using GPT-4.1."""
    prompt = build_prompt(text, schema_fields)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a legal document extraction system. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except json.JSONDecodeError:
            logger.warning(f"JSON parse error on attempt {attempt + 1}")
        except Exception as e:
            logger.warning(f"API error on attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)

    return None


def compare_fields(gpt_extraction: dict, gemini_extraction: dict, field_name: str) -> dict:
    """Compare a single field between GPT and Gemini extractions."""
    gpt_val = gpt_extraction.get(field_name)
    gem_val = gemini_extraction.get(field_name)

    # Both None/empty
    if not gpt_val and not gem_val:
        return {"agreement": "both_empty", "score": 1.0}

    # One empty
    if not gpt_val or not gem_val:
        return {"agreement": "one_empty", "score": 0.0}

    # Both present — compare
    if isinstance(gpt_val, str) and isinstance(gem_val, str):
        # Exact match
        if gpt_val.strip().lower() == gem_val.strip().lower():
            return {"agreement": "exact", "score": 1.0}
        # Partial overlap (Jaccard on words)
        gpt_words = set(gpt_val.lower().split())
        gem_words = set(gem_val.lower().split())
        if gpt_words and gem_words:
            jaccard = len(gpt_words & gem_words) / len(gpt_words | gem_words)
            return {"agreement": "partial", "score": jaccard}
        return {"agreement": "mismatch", "score": 0.0}

    elif isinstance(gpt_val, list) and isinstance(gem_val, list):
        gpt_set = set(str(v).lower() for v in gpt_val)
        gem_set = set(str(v).lower() for v in gem_val)
        if not gpt_set and not gem_set:
            return {"agreement": "both_empty_list", "score": 1.0}
        if not gpt_set or not gem_set:
            return {"agreement": "one_empty_list", "score": 0.0}
        jaccard = len(gpt_set & gem_set) / len(gpt_set | gem_set)
        return {"agreement": "list_overlap", "score": jaccard}

    elif isinstance(gpt_val, dict) and isinstance(gem_val, dict):
        # Compare dict keys that match
        common_keys = set(gpt_val.keys()) & set(gem_val.keys())
        if not common_keys:
            return {"agreement": "no_common_keys", "score": 0.0}
        matches = sum(1 for k in common_keys if str(gpt_val[k]).lower() == str(gem_val[k]).lower())
        return {"agreement": "dict_overlap", "score": matches / len(common_keys)}

    else:
        # Type mismatch or other
        if str(gpt_val).strip().lower() == str(gem_val).strip().lower():
            return {"agreement": "string_match", "score": 1.0}
        return {"agreement": "type_mismatch", "score": 0.0}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    # Load Schema B field definitions from extraction schema module description
    # These match the extracted_ columns in pl-court-raw-enriched
    schema_b_fields = [
        "title", "date_issued", "summary", "thesis", "keywords",
        "outcome", "legal_references", "legal_concepts", "parties",
        "legal_analysis",
    ]
    schema_fields = {
        "document_number": "string, official case reference number (sygnatura sprawy)",
        "document_type": "string enum: judgment, tax_interpretation, legal_act",
        "title": "string, document title (max 200 chars)",
        "date_issued": "date ISO 8601 (YYYY-MM-DD)",
        "summary": "string, 3-5 sentence summary covering document type, legal issue, key facts, decision, legal basis",
        "thesis": "string, main legal principle or conclusion (1-3 sentences)",
        "keywords": "List[string], 5-15 relevant Polish legal keywords",
        "factual_state": "string, factual circumstances (stan faktyczny)",
        "legal_state": "string, legal framework and provisions (stan prawny)",
        "outcome": "JSON: {decision_type: enum, decision_summary: string, awarded_amounts: array, legal_effect: string}",
        "legal_references": "JSON array of legal citations (5-15)",
        "legal_concepts": "JSON array of legal concepts (3-10)",
        "parties": "JSON array of parties with roles",
        "legal_analysis": "JSON object with structured legal reasoning",
    }

    logger.info(f"Loading {SAMPLE_SIZE} docs from pl-court-raw-enriched...")
    ds = load_dataset("JuDDGES/pl-court-raw-enriched", split="train", streaming=True)

    docs = []
    for i, row in enumerate(ds):
        docs.append(row)
        if i + 1 >= SAMPLE_SIZE:
            break
    logger.info(f"Loaded {len(docs)} documents")

    # Run GPT-4.1 extraction
    results = []
    field_agreements = defaultdict(list)
    parse_errors = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for idx, doc in enumerate(docs):
        text = doc.get("content") or doc.get("excerpt") or ""
        if not text:
            continue

        gpt_result = extract_with_gpt(client, text, schema_fields)

        if gpt_result is None:
            parse_errors += 1
            continue

        # Build Gemini extraction from extracted_ columns
        gemini_result = {}
        for field in schema_b_fields:
            gemini_result[field] = doc.get(f"extracted_{field}")

        # Compare per field
        doc_agreement = {}
        for field in schema_b_fields:
            comparison = compare_fields(gpt_result, gemini_result, field)
            field_agreements[field].append(comparison["score"])
            doc_agreement[field] = comparison

        results.append({
            "doc_idx": idx,
            "gpt_extraction": gpt_result,
            "gemini_extraction": gemini_result,
            "field_agreement": doc_agreement,
        })

        if (idx + 1) % BATCH_LOG_EVERY == 0:
            avg_so_far = mean(
                score for scores in field_agreements.values() for score in scores
            )
            logger.info(f"  [{idx + 1}/{len(docs)}] avg agreement: {avg_so_far:.3f}, errors: {parse_errors}")

    # Aggregate results
    logger.info(f"\n{'='*70}")
    logger.info(f"Cross-Model Agreement: GPT-4.1 vs Gemini 2.5 Pro (n={len(results)})")
    logger.info(f"Parse errors: {parse_errors}")
    logger.info(f"{'='*70}")

    summary = {}
    print(f"\n{'Field':<25} {'Agreement':>10} {'Std':>8} {'N':>6}")
    print("-" * 55)
    for field in schema_b_fields:
        scores = field_agreements[field]
        if scores:
            m = mean(scores)
            s = stdev(scores) if len(scores) > 1 else 0.0
            summary[field] = {"mean": m, "std": s, "n": len(scores)}
            print(f"{field:<25} {m:>10.3f} {s:>8.3f} {len(scores):>6}")

    overall = mean(score for scores in field_agreements.values() for score in scores)
    print("-" * 55)
    print(f"{'OVERALL':<25} {overall:>10.3f}")

    # Save
    output = {
        "model": MODEL,
        "n_docs": len(results),
        "n_parse_errors": parse_errors,
        "per_field_agreement": summary,
        "overall_agreement": overall,
    }
    with open(OUTPUT_DIR / "agreement_summary.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Save detailed results (first 100 for inspection)
    with open(OUTPUT_DIR / "detailed_results_sample.json", "w") as f:
        json.dump(results[:100], f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
