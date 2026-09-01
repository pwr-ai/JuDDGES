#!/usr/bin/env python
"""EXP A v2: Cross-Model Agreement (Task 2).

Runs GPT-4.1 extraction with Schema B on documents from pl-nsa-enriched
that already have Gemini 2.5 Pro extractions (extracted_* fields non-null).
Compares GPT-4.1 vs Gemini per-field agreement.

Usage:
    PYTHONPATH=. python scripts/neurips_experiments/run_exp_a_cross_model_v2.py
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from datasets import load_dataset
from loguru import logger
from openai import OpenAI

SAMPLE_SIZE = 500  # docs with Gemini extractions
MODEL = "gpt-4.1"
OUTPUT_DIR = Path("data/experiments/neurips_results/exp_a_cross_model")
MAX_RETRIES = 3
BATCH_LOG_EVERY = 25

SCHEMA_B_FIELDS = {
    "document_number": "string, official case reference number (sygnatura sprawy)",
    "document_type": "string enum: judgment, tax_interpretation, legal_act",
    "title": "string, document title (max 200 chars)",
    "date_issued": "date ISO 8601 (YYYY-MM-DD)",
    "summary": "string, 3-5 sentence summary",
    "thesis": "string, main legal principle (1-3 sentences)",
    "keywords": "List[string], 5-15 Polish legal keywords",
    "factual_state": "string, factual circumstances (stan faktyczny)",
    "legal_state": "string, legal framework and provisions (stan prawny)",
    "outcome": "JSON: {decision_type: enum, decision_summary: string}",
    "legal_references": "JSON array of legal citations (5-15)",
    "legal_concepts": "JSON array of legal concepts (3-10)",
    "parties": "JSON array of parties with roles",
    "legal_analysis": "JSON object with structured legal reasoning",
}

COMPARISON_FIELDS = [
    "title", "date_issued", "summary", "thesis", "keywords",
    "outcome", "legal_references", "legal_concepts", "parties",
    "legal_analysis",
]


def build_prompt(text: str) -> str:
    fields_desc = "\n".join(f"- {k}: {v}" for k, v in SCHEMA_B_FIELDS.items())
    return (
        "Ekstraktuj poniższe pola z polskiego dokumentu prawnego.\n"
        "Zwróć TYLKO poprawny JSON:\n\n"
        f"{fields_desc}\n\n"
        "ZASADY: Ekstraktuj WYŁĄCZNIE z tekstu. Daty: YYYY-MM-DD. "
        "Puste pola: pusty string. Puste listy: []. Null dla brakujących obiektów.\n\n"
        f"DOKUMENT:\n{text[:12000]}\n\nJSON:"
    )


def extract_with_gpt(client: OpenAI, text: str) -> dict | None:
    prompt = build_prompt(text)
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
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            logger.warning(f"JSON parse error attempt {attempt + 1}")
        except Exception as e:
            logger.warning(f"API error attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    return None


def compute_agreement(gpt_val, gem_val) -> float:
    """Compute agreement score between two field values."""
    if gpt_val is None and gem_val is None:
        return 1.0
    if gpt_val is None or gem_val is None:
        return 0.0

    # String comparison
    if isinstance(gpt_val, str) and isinstance(gem_val, str):
        if not gpt_val.strip() and not gem_val.strip():
            return 1.0
        if gpt_val.strip().lower() == gem_val.strip().lower():
            return 1.0
        # Word-level Jaccard
        gw = set(gpt_val.lower().split())
        gg = set(gem_val.lower().split())
        if gw and gg:
            return len(gw & gg) / len(gw | gg)
        return 0.0

    # List comparison
    if isinstance(gpt_val, list) and isinstance(gem_val, list):
        gs = set(str(v).lower().strip() for v in gpt_val if v)
        gg = set(str(v).lower().strip() for v in gem_val if v)
        if not gs and not gg:
            return 1.0
        if not gs or not gg:
            return 0.0
        return len(gs & gg) / len(gs | gg)

    # Dict comparison
    if isinstance(gpt_val, dict) and isinstance(gem_val, dict):
        common = set(gpt_val.keys()) & set(gem_val.keys())
        if not common:
            return 0.0
        matches = sum(1 for k in common if str(gpt_val[k]).lower() == str(gem_val[k]).lower())
        return matches / len(common)

    # Fallback string comparison
    return 1.0 if str(gpt_val).strip().lower() == str(gem_val).strip().lower() else 0.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    # Load enriched docs WITH Gemini extractions
    logger.info("Loading pl-nsa-enriched docs with Gemini extractions...")
    ds = load_dataset("JuDDGES/pl-nsa-enriched", split="train", streaming=True)

    docs = []
    scanned = 0
    for row in ds:
        scanned += 1
        if row.get("extracted_summary") is not None:
            docs.append(row)
        if len(docs) >= SAMPLE_SIZE:
            break
        if scanned % 2000 == 0:
            logger.info(f"  Scanned {scanned}, found {len(docs)} with extractions...")

    logger.info(f"Found {len(docs)} docs with Gemini extractions (scanned {scanned})")

    # Run GPT-4.1 extraction and compare
    field_agreements = defaultdict(list)
    parse_errors = 0

    for idx, doc in enumerate(docs):
        text = doc.get("full_text") or doc.get("content") or ""
        if not text:
            continue

        gpt_result = extract_with_gpt(client, text)
        if gpt_result is None:
            parse_errors += 1
            continue

        for field in COMPARISON_FIELDS:
            gpt_val = gpt_result.get(field)
            gem_val = doc.get(f"extracted_{field}")

            # Parse Gemini JSON strings if needed
            if isinstance(gem_val, str):
                try:
                    gem_val = json.loads(gem_val)
                except (json.JSONDecodeError, TypeError):
                    pass

            score = compute_agreement(gpt_val, gem_val)
            field_agreements[field].append(score)

        if (idx + 1) % BATCH_LOG_EVERY == 0:
            avg = mean(s for scores in field_agreements.values() for s in scores)
            logger.info(f"  [{idx + 1}/{len(docs)}] avg agreement: {avg:.3f}, errors: {parse_errors}")

    # Results
    logger.info(f"\n{'='*70}")
    logger.info(f"Cross-Model Agreement: GPT-4.1 vs Gemini 2.5 Pro (n={len(docs) - parse_errors})")
    logger.info(f"{'='*70}")

    summary = {}
    print(f"\n{'Field':<25} {'Agreement':>10} {'Std':>8} {'N':>6}")
    print("-" * 55)
    for field in COMPARISON_FIELDS:
        scores = field_agreements[field]
        if scores:
            m = mean(scores)
            s = stdev(scores) if len(scores) > 1 else 0.0
            summary[field] = {"mean": round(m, 4), "std": round(s, 4), "n": len(scores)}
            print(f"{field:<25} {m:>10.3f} {s:>8.3f} {len(scores):>6}")

    all_scores = [s for scores in field_agreements.values() for s in scores]
    overall = mean(all_scores) if all_scores else 0.0
    print("-" * 55)
    print(f"{'OVERALL':<25} {overall:>10.3f}")

    output = {
        "model": MODEL,
        "comparison_model": "gemini-2.5-pro",
        "dataset": "JuDDGES/pl-nsa-enriched",
        "n_docs": len(docs) - parse_errors,
        "n_parse_errors": parse_errors,
        "n_scanned": scanned,
        "per_field_agreement": summary,
        "overall_agreement": round(overall, 4),
    }
    with open(OUTPUT_DIR / "agreement_summary.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
