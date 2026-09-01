#!/usr/bin/env python
"""EXP D: Error Taxonomy — classify extraction errors.

Compares GPT-4.1 test-split extractions vs human-reviewed annotated-split
gold labels. Identifies extraction errors, samples 300, and classifies them
into 5 categories using GPT-4.1-mini.

Categories:
1. Hallucination — model generates content not in document
2. Partial match — correct info but incomplete
3. Wrong field — info from wrong part of document
4. Format error — correct content, wrong format
5. Missing — field not extracted despite info being present

Cost estimate: ~$0.22

Usage:
    PYTHONPATH=. python scripts/neurips_experiments/run_exp_d_error_taxonomy.py
"""

import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from datasets import load_dataset
from loguru import logger
from openai import OpenAI

random.seed(42)

SAMPLE_SIZE = 300
JUDGE_MODEL = "gpt-4.1-mini"
OUTPUT_DIR = Path("data/experiments/neurips_results/exp_d_error_taxonomy")

ERROR_CATEGORIES = [
    "hallucination",
    "partial_match",
    "wrong_field",
    "format_error",
    "missing",
    "other",
]

CLASSIFY_PROMPT = """You are an extraction error classifier for legal documents.

Given a document excerpt, the expected gold value, and the model's predicted value for a specific field, classify the error into exactly ONE category:

1. **hallucination** — The predicted value contains information NOT present in the document.
2. **partial_match** — The prediction contains some correct information but is incomplete or has extra content.
3. **wrong_field** — The prediction contains information from the document, but from the wrong section/context (intended for a different field).
4. **format_error** — The content is essentially correct but in the wrong format (e.g., different date format, different enum spelling, string vs list).
5. **missing** — The prediction is empty/null but the gold value shows information IS present in the document.
6. **other** — None of the above.

FIELD: {field_name}
FIELD DESCRIPTION: {field_desc}

GOLD VALUE: {gold_value}
PREDICTED VALUE: {pred_value}

DOCUMENT EXCERPT (first 2000 chars):
{doc_excerpt}

Respond with ONLY a JSON object:
{{"category": "<one of: hallucination, partial_match, wrong_field, format_error, missing, other>", "explanation": "<1 sentence>"}}"""


def find_errors(ds_test, ds_annotated, schema_fields: list[str]) -> list[dict]:
    """Find all field-level extraction errors between test (GPT-4.1) and annotated (human)."""
    errors = []

    for idx in range(len(ds_test)):
        text = ds_test[idx]["context"]
        pred = json.loads(ds_test[idx]["output"])  # GPT-4.1 extraction
        gold = json.loads(ds_annotated[idx]["output"])  # Human-reviewed

        for field_name in schema_fields:
            pred_val = pred.get(field_name)
            gold_val = gold.get(field_name)

            # Normalize
            if pred_val == "None":
                pred_val = None
            if gold_val == "None":
                gold_val = None

            # Check if they disagree
            if str(pred_val).strip().lower() != str(gold_val).strip().lower():
                errors.append({
                    "doc_idx": idx,
                    "field_name": field_name,
                    "pred_value": pred_val,
                    "gold_value": gold_val,
                    "doc_excerpt": text[:2000],
                })

    return errors


def classify_error(client: OpenAI, error: dict, field_desc: str) -> dict:
    """Classify a single error using GPT-4.1-mini."""
    prompt = CLASSIFY_PROMPT.format(
        field_name=error["field_name"],
        field_desc=field_desc,
        gold_value=str(error["gold_value"])[:500],
        pred_value=str(error["pred_value"])[:500],
        doc_excerpt=error["doc_excerpt"],
    )

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        logger.warning(f"Classification error: {e}")
        return {"category": "other", "explanation": f"API error: {e}"}


def main():
    import yaml

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    # Load schema for field descriptions
    with open("configs/ie_schema/swiss_franc_loans.yaml") as f:
        schema = yaml.safe_load(f)
    field_descs = {name: props.get("description", "") for name, props in schema.items()}

    # Load datasets
    logger.info("Loading pl-swiss-franc-loans test + annotated splits...")
    ds_test = load_dataset("JuDDGES/pl-swiss-franc-loans", split="test")
    ds_annotated = load_dataset("JuDDGES/pl-swiss-franc-loans", split="annotated")

    # Find all errors
    logger.info("Finding extraction errors (GPT-4.1 test vs human annotated)...")
    all_errors = find_errors(ds_test, ds_annotated, list(schema.keys()))
    logger.info(f"Found {len(all_errors)} field-level disagreements across {len(ds_test)} docs × {len(schema)} fields")

    # Error rate
    total_fields = len(ds_test) * len(schema)
    error_rate = len(all_errors) / total_fields
    logger.info(f"Error rate: {error_rate:.4f} ({len(all_errors)}/{total_fields})")

    # Distribution by field
    field_error_counts = Counter(e["field_name"] for e in all_errors)
    logger.info(f"\nTop 10 error fields:")
    for field, count in field_error_counts.most_common(10):
        logger.info(f"  {field}: {count} errors ({count/len(ds_test)*100:.1f}% of docs)")

    # Sample errors for classification
    sample_size = min(SAMPLE_SIZE, len(all_errors))
    sampled_errors = random.sample(all_errors, sample_size)
    logger.info(f"\nClassifying {sample_size} sampled errors with {JUDGE_MODEL}...")

    # Classify
    classified = []
    category_counts = Counter()
    field_category_counts = defaultdict(Counter)

    for i, error in enumerate(sampled_errors):
        field_desc = field_descs.get(error["field_name"], "")
        result = classify_error(client, error, field_desc)

        category = result.get("category", "other")
        if category not in ERROR_CATEGORIES:
            category = "other"

        category_counts[category] += 1
        field_category_counts[error["field_name"]][category] += 1

        classified.append({
            "field_name": error["field_name"],
            "pred_value": str(error["pred_value"])[:200],
            "gold_value": str(error["gold_value"])[:200],
            "category": category,
            "explanation": result.get("explanation", ""),
        })

        if (i + 1) % 50 == 0:
            logger.info(f"  [{i + 1}/{sample_size}] classified")

    # Results
    logger.info(f"\n{'='*60}")
    logger.info(f"Error Taxonomy (n={sample_size})")
    logger.info(f"{'='*60}")

    print(f"\n{'Category':<20} {'Count':>6} {'%':>8}")
    print("-" * 36)
    for cat in ERROR_CATEGORIES:
        count = category_counts[cat]
        pct = count / sample_size * 100 if sample_size > 0 else 0
        print(f"{cat:<20} {count:>6} {pct:>7.1f}%")
    print("-" * 36)
    print(f"{'Total':<20} {sample_size:>6}")

    # Per-field top categories
    print(f"\n{'Field':<35} {'Top Category':<20} {'Count':>6}")
    print("-" * 65)
    for field, counts in sorted(field_category_counts.items(), key=lambda x: sum(x[1].values()), reverse=True)[:15]:
        top_cat = counts.most_common(1)[0] if counts else ("none", 0)
        print(f"{field:<35} {top_cat[0]:<20} {top_cat[1]:>6}")

    # Save
    output = {
        "n_total_errors": len(all_errors),
        "n_total_fields": total_fields,
        "error_rate": error_rate,
        "n_sampled": sample_size,
        "category_distribution": dict(category_counts),
        "field_error_counts": dict(field_error_counts.most_common()),
        "field_category_distribution": {
            field: dict(counts) for field, counts in field_category_counts.items()
        },
    }
    with open(OUTPUT_DIR / "error_taxonomy.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "classified_errors.json", "w") as f:
        json.dump(classified, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
