#!/usr/bin/env python
"""EXP B: Annotation Validation — Gemini 2.5 Pro on 100 pl-swiss-franc-loans docs.

Runs Gemini extraction with Schema A (42 fields) on 100 documents from
the test split. Compares Gemini vs GPT-4.1 (test split gold) vs
annotated split (human-reviewed gold).

This gives us a 3-way comparison: GPT-4.1 vs Gemini vs Human-reviewed.

Cost estimate: ~$4

Usage:
    PYTHONPATH=. python scripts/neurips_experiments/run_exp_b_gemini_validation.py
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import google.generativeai as genai
import yaml
from datasets import load_dataset
from loguru import logger

# ---------------------------------------------------------------------------
SAMPLE_SIZE = 100
MODEL = "gemini-2.5-pro-preview-05-06"
OUTPUT_DIR = Path("data/experiments/neurips_results/exp_b_gemini_validation")
MAX_RETRIES = 3


def build_prompt(text: str, schema: dict) -> str:
    """Build extraction prompt with Schema A fields."""
    fields_desc = []
    for name, props in schema.items():
        ftype = props.get("type", "string")
        choices = props.get("choices", [])
        desc = props.get("description", "")
        if choices:
            fields_desc.append(f"- {name} ({ftype}, choices: {choices}): {desc}")
        else:
            fields_desc.append(f"- {name} ({ftype}): {desc}")

    fields_str = "\n".join(fields_desc)

    return f"""Ekstraktuj poniższe pola z polskiego orzeczenia sądowego.
Zwróć TYLKO poprawny obiekt JSON z tymi polami:

{fields_str}

ZASADY:
- Ekstraktuj informacje WYŁĄCZNIE z tekstu dokumentu
- Dla pól enum: używaj TYLKO wartości z listy choices
- Daty: format YYYY-MM-DD
- Puste pola: pusty string "" dla string, null dla brakujących
- Odpowiedź TYLKO w JSON, bez dodatkowego tekstu

DOKUMENT:
{text[:15000]}

JSON:"""


def extract_with_gemini(text: str, schema: dict) -> dict | None:
    """Extract fields using Gemini 2.5 Pro."""
    prompt = build_prompt(text, schema)

    for attempt in range(MAX_RETRIES):
        try:
            model = genai.GenerativeModel(MODEL)
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4000,
                    response_mime_type="application/json",
                ),
            )
            text_response = response.text
            # Clean potential markdown fences
            if text_response.startswith("```"):
                text_response = text_response.split("\n", 1)[1]
                if text_response.endswith("```"):
                    text_response = text_response[:-3]
            result = json.loads(text_response)
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
        except Exception as e:
            logger.warning(f"API error on attempt {attempt + 1}: {e}")
            time.sleep(2 ** (attempt + 1))

    return None


def score_field(pred_val, gold_val) -> float:
    """Simple accuracy score for a single field."""
    # Normalize None/"None"
    if pred_val == "None":
        pred_val = None
    if gold_val == "None":
        gold_val = None

    if pred_val is None and gold_val is None:
        return 1.0
    if pred_val is None or gold_val is None:
        return 0.0
    if str(pred_val).strip().lower() == str(gold_val).strip().lower():
        return 1.0
    return 0.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Configure Gemini
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    # Load schema
    schema_path = Path("configs/ie_schema/swiss_franc_loans.yaml")
    with open(schema_path) as f:
        schema = yaml.safe_load(f)
    logger.info(f"Schema: {len(schema)} fields")

    # Load datasets
    logger.info("Loading pl-swiss-franc-loans splits...")
    ds_test = load_dataset("JuDDGES/pl-swiss-franc-loans", split="test")
    ds_annotated = load_dataset("JuDDGES/pl-swiss-franc-loans", split="annotated")
    logger.info(f"Test: {len(ds_test)}, Annotated: {len(ds_annotated)}")

    # Sample first 100 docs
    indices = list(range(min(SAMPLE_SIZE, len(ds_test))))

    results = []
    field_scores_gemini_vs_gpt = defaultdict(list)
    field_scores_gemini_vs_human = defaultdict(list)
    field_scores_gpt_vs_human = defaultdict(list)
    parse_errors = 0

    for i, idx in enumerate(indices):
        text = ds_test[idx]["context"]
        gpt_gold = json.loads(ds_test[idx]["output"])
        human_gold = json.loads(ds_annotated[idx]["output"])

        # Run Gemini
        gemini_result = extract_with_gemini(text, schema)

        if gemini_result is None:
            parse_errors += 1
            continue

        # Score per field: 3-way comparison
        doc_scores = {}
        for field_name in schema:
            gem_val = gemini_result.get(field_name)
            gpt_val = gpt_gold.get(field_name)
            human_val = human_gold.get(field_name)

            s_gem_gpt = score_field(gem_val, gpt_val)
            s_gem_human = score_field(gem_val, human_val)
            s_gpt_human = score_field(gpt_val, human_val)

            field_scores_gemini_vs_gpt[field_name].append(s_gem_gpt)
            field_scores_gemini_vs_human[field_name].append(s_gem_human)
            field_scores_gpt_vs_human[field_name].append(s_gpt_human)

            doc_scores[field_name] = {
                "gemini_vs_gpt": s_gem_gpt,
                "gemini_vs_human": s_gem_human,
                "gpt_vs_human": s_gpt_human,
            }

        results.append({
            "idx": idx,
            "gemini_extraction": gemini_result,
            "scores": doc_scores,
        })

        if (i + 1) % 10 == 0:
            avg_gg = mean(s for scores in field_scores_gemini_vs_gpt.values() for s in scores)
            avg_gh = mean(s for scores in field_scores_gemini_vs_human.values() for s in scores)
            logger.info(f"  [{i + 1}/{len(indices)}] Gem vs GPT: {avg_gg:.3f}, Gem vs Human: {avg_gh:.3f}, errors: {parse_errors}")

    # Aggregate
    logger.info(f"\n{'='*80}")
    logger.info(f"3-Way Annotation Validation (n={len(results)}, parse_errors={parse_errors})")
    logger.info(f"{'='*80}")

    summary = {}
    print(f"\n{'Field':<35} {'Gem↔GPT':>8} {'Gem↔Hum':>8} {'GPT↔Hum':>8}")
    print("-" * 65)

    for field_name in schema:
        gg = field_scores_gemini_vs_gpt[field_name]
        gh = field_scores_gemini_vs_human[field_name]
        gp = field_scores_gpt_vs_human[field_name]

        if gg:
            summary[field_name] = {
                "gemini_vs_gpt": {"mean": mean(gg), "std": stdev(gg) if len(gg) > 1 else 0},
                "gemini_vs_human": {"mean": mean(gh), "std": stdev(gh) if len(gh) > 1 else 0},
                "gpt_vs_human": {"mean": mean(gp), "std": stdev(gp) if len(gp) > 1 else 0},
            }
            print(f"{field_name:<35} {mean(gg):>8.3f} {mean(gh):>8.3f} {mean(gp):>8.3f}")

    # Overall
    overall_gg = mean(s for scores in field_scores_gemini_vs_gpt.values() for s in scores)
    overall_gh = mean(s for scores in field_scores_gemini_vs_human.values() for s in scores)
    overall_gp = mean(s for scores in field_scores_gpt_vs_human.values() for s in scores)
    print("-" * 65)
    print(f"{'OVERALL':<35} {overall_gg:>8.3f} {overall_gh:>8.3f} {overall_gp:>8.3f}")

    output = {
        "n_docs": len(results),
        "n_parse_errors": parse_errors,
        "overall": {
            "gemini_vs_gpt": overall_gg,
            "gemini_vs_human": overall_gh,
            "gpt_vs_human": overall_gp,
        },
        "per_field": summary,
    }
    with open(OUTPUT_DIR / "validation_summary.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "detailed_results.json", "w") as f:
        json.dump(results[:50], f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
