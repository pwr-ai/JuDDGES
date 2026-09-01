#!/usr/bin/env python
"""Rule-Based Baseline for EN Appeal Court dataset.

All fields are list-typed. Rule-based approach uses regex and keyword matching.

Usage:
    python scripts/neurips_experiments/run_rule_based_en.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import yaml
from datasets import load_dataset
from loguru import logger


# Court name patterns
COURT_PATTERN = re.compile(
    r"(?:Crown Court at |Crown Court, |Southwark Crown Court|"
    r"(?:\w+ )+Crown Court|Central Criminal Court)",
    re.IGNORECASE,
)

# Date pattern: "22 January 2003", "2003-01-22", "22/01/2003"
EN_MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}
DATE_PATTERN = re.compile(
    r"(\d{1,2})\s+(" + "|".join(EN_MONTHS.keys()) + r")\s+(\d{4})",
    re.IGNORECASE,
)
DATE_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

SENTENCE_PATTERN = re.compile(
    r"(\d+)\s*(?:years?|months?)\s*(?:'?\s*)?(?:imprisonment|detention|"
    r"community order|suspended|immediate custody)",
    re.IGNORECASE,
)

APPEAL_OUTCOME_KEYWORDS = {
    "dismissed": ["dismissed", "refuse", "failed"],
    "allowed": ["allowed", "quashed"],
}


def extract_courts(text: str) -> list[str]:
    return list(set(COURT_PATTERN.findall(text)))


def extract_dates(text: str) -> list[str]:
    dates = []
    for m in DATE_PATTERN.finditer(text):
        day, month_name, year = m.groups()
        month = EN_MONTHS.get(month_name.lower(), "01")
        dates.append(f"{year}-{month}-{int(day):02d}")
    for m in DATE_ISO.finditer(text):
        dates.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    return list(set(dates))


def extract_sentences(text: str) -> list[str]:
    return list(set(SENTENCE_PATTERN.findall(text)))


def extract_enum_list(text: str, choices: list[str]) -> list[str]:
    text_lower = text.lower()
    found = []
    for choice in choices:
        if choice and choice.lower() in text_lower:
            found.append(choice)
    return found


def extract_offences(text: str) -> list[str]:
    """Extract offence descriptions using keyword patterns."""
    patterns = [
        re.compile(r"(?:convicted of|guilty of|charged with)\s+(.{10,80}?)(?:\.|,|;|\band\b)", re.IGNORECASE),
        re.compile(r"count\s*\d+[:\s]+(.{10,80}?)(?:\.|,|;)", re.IGNORECASE),
    ]
    offences = set()
    for p in patterns:
        for m in p.finditer(text):
            offences.add(m.group(1).strip())
    return list(offences)


def extract_ages(text: str) -> list[str]:
    pattern = re.compile(r"aged?\s*(\d{1,2})", re.IGNORECASE)
    return list(set(pattern.findall(text)))


def extract_fields_en(text: str, schema: dict) -> dict:
    """Extract all EN schema fields using rule-based methods."""
    result = {}
    text_lower = text.lower()

    for field_name, field_props in schema.items():
        items = field_props.get("items", {})
        item_type = items.get("type", "string")
        choices = items.get("choices", [])

        if "CourtName" in field_name:
            result[field_name] = extract_courts(text)
        elif "Date" in field_name:
            result[field_name] = extract_dates(text)
        elif "Offence" in field_name and item_type == "string":
            result[field_name] = extract_offences(text)
        elif "Age" in field_name:
            result[field_name] = extract_ages(text)
        elif "Sentence" == field_name:
            result[field_name] = extract_sentences(text)
        elif item_type == "enum":
            result[field_name] = extract_enum_list(text, choices)
        elif field_name == "AppealOutcome":
            outcomes = []
            if any(kw in text_lower for kw in APPEAL_OUTCOME_KEYWORDS["dismissed"]):
                outcomes.append("Dismissed")
            if any(kw in text_lower for kw in APPEAL_OUTCOME_KEYWORDS["allowed"]):
                outcomes.append("Allowed")
            result[field_name] = outcomes
        elif item_type == "int":
            nums = re.findall(r"\b(\d{1,3})\b", text[:2000])
            result[field_name] = nums[:3] if nums else []
        else:
            result[field_name] = []

    return result


def evaluate_list_field(pred: list, gold: list) -> dict:
    """Evaluate list prediction vs gold using set-based F1."""
    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred or not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Normalize for comparison
    pred_set = set(str(v).lower().strip() for v in pred if v)
    gold_set = set(str(v).lower().strip() for v in gold if v)

    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    schema_path = Path("configs/ie_schema/en_appealcourt.yaml")
    with open(schema_path) as f:
        schema = yaml.safe_load(f)
    logger.info(f"EN Schema: {len(schema)} fields")

    logger.info("Loading en-appealcourt test set from HuggingFace...")
    # Try different dataset names
    try:
        ds = load_dataset("JuDDGES/en-appealcourt", split="test")
    except Exception:
        ds = load_dataset("JuDDGES/en-appealcourt", split="annotated")
    logger.info(f"Loaded {len(ds)} test samples")

    field_scores = defaultdict(list)
    for sample in ds:
        text = sample["context"]
        gold = json.loads(sample["output"])
        pred = extract_fields_en(text, schema)

        for field_name in schema:
            pred_val = pred.get(field_name, [])
            gold_val = gold.get(field_name, [])

            if isinstance(gold_val, str):
                try:
                    gold_val = json.loads(gold_val)
                except (json.JSONDecodeError, TypeError):
                    gold_val = [gold_val] if gold_val else []

            if not isinstance(pred_val, list):
                pred_val = [pred_val] if pred_val else []
            if not isinstance(gold_val, list):
                gold_val = [gold_val] if gold_val else []

            scores = evaluate_list_field(pred_val, gold_val)
            field_scores[field_name].append(scores["f1"])

    # Aggregate
    results = {}
    for field_name, scores in field_scores.items():
        results[field_name] = {
            "f1": sum(scores) / len(scores) if scores else 0.0,
            "n": len(scores),
        }

    overall = sum(r["f1"] for r in results.values()) / len(results) if results else 0.0

    logger.info(f"\n{'='*60}")
    logger.info(f"Rule-Based Baseline on en-appealcourt (n={len(ds)})")
    logger.info(f"Overall F1: {overall:.4f}")
    logger.info(f"{'='*60}")

    fields_sorted = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    print(f"\n{'Field':<40} {'F1':>8} {'N':>6}")
    print("-" * 56)
    for field_name, metrics in fields_sorted:
        print(f"{field_name:<40} {metrics['f1']:>8.3f} {metrics['n']:>6}")
    print("-" * 56)
    print(f"{'OVERALL':<40} {overall:>8.4f}")

    output_dir = Path("data/experiments/neurips_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "rule_based_en_appealcourt.json", "w") as f:
        json.dump({"overall_f1": overall, "fields": results}, f, indent=2)
    logger.info(f"Saved to {output_dir / 'rule_based_en_appealcourt.json'}")


if __name__ == "__main__":
    main()
