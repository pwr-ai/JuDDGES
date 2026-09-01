#!/usr/bin/env python
"""Rule-Based Baseline on HuggingFace data.

Loads pl-swiss-franc-loans test set from HF, runs rule-based extraction,
evaluates against gold annotations, and outputs per-field results.

Usage:
    python scripts/neurips_experiments/run_rule_based_hf.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import yaml
from datasets import load_dataset
from loguru import logger

# ---------------------------------------------------------------------------
# Polish date/keyword resources
# ---------------------------------------------------------------------------
PL_MONTHS = {
    "stycznia": "01", "lutego": "02", "marca": "03", "kwietnia": "04",
    "maja": "05", "czerwca": "06", "lipca": "07", "sierpnia": "08",
    "września": "09", "października": "10", "listopada": "11", "grudnia": "12",
}

DATE_PATTERNS = [
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),
    re.compile(r"(\d{1,2})\.(\d{2})\.(\d{4})"),
    re.compile(
        r"(\d{1,2})\s+(" + "|".join(PL_MONTHS.keys()) + r")\s+(\d{4})\s*r?\.",
        re.IGNORECASE,
    ),
]

# Court name dictionary
COURT_NAMES = {
    "sąd rejonowy": "Sąd Rejonowy",
    "sąd okręgowy": "Sąd Okręgowy",
    "sąd frankowy": "Sąd Frankowy",
    "sąd ochrony konkurencji": "Sąd Ochrony Konkurencji i Konsumentów",
}

# Apelacja name dictionary
APELACJA_NAMES = [
    "białostocka", "gdańska", "katowicka", "krakowska", "lubelska",
    "łódzka", "poznańska", "rzeszowska", "szczecińska", "warszawska",
    "wrocławska",
]


def extract_date(text: str) -> str | None:
    """Extract the first date from text in YYYY-MM-DD format."""
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                if groups[1] in PL_MONTHS:
                    return f"{groups[2]}-{PL_MONTHS[groups[1]]}-{int(groups[0]):02d}"
                elif len(groups[0]) == 4:
                    return f"{groups[0]}-{groups[1]}-{groups[2]}"
                elif len(groups[2]) == 4:
                    return f"{groups[2]}-{groups[1]}-{int(groups[0]):02d}"
    return None


def extract_court_type(text: str) -> str | None:
    """Extract court type from text."""
    text_lower = text.lower()
    for key, value in COURT_NAMES.items():
        if key in text_lower:
            return value
    return None


def extract_apelacja(text: str) -> str | None:
    """Extract apelacja name from text."""
    text_lower = text.lower()
    for name in APELACJA_NAMES:
        if name in text_lower:
            return name.capitalize()
    # Try regex pattern
    m = re.search(r"apelacj[ai]\s+(\w+)", text_lower)
    if m:
        return m.group(1).capitalize()
    return None


def extract_boolean(text: str, positive_keywords: list[str], negative_keywords: list[str]) -> str | None:
    """Simple keyword-based boolean extraction."""
    text_lower = text.lower()
    pos = sum(1 for kw in positive_keywords if kw in text_lower)
    neg = sum(1 for kw in negative_keywords if kw in text_lower)
    if pos > neg:
        return "Tak"
    elif neg > pos:
        return "Nie"
    return None


def extract_enum(text: str, choices: list[str]) -> str | None:
    """Extract enum value by finding the first matching choice in text."""
    text_lower = text.lower()
    for choice in choices:
        if choice and choice.lower() in text_lower:
            return choice
    return None


def extract_legal_references(text: str) -> str:
    """Extract legal provision references."""
    patterns = [
        re.compile(r"[Aa]rt\.?\s*\d+[a-z]?(?:\s*§\s*\d+)?(?:\s*(?:ust|pkt|lit)\.\s*\d+)*"),
    ]
    refs = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            refs.add(match.group().strip())
    return "; ".join(sorted(refs)) if refs else None


def extract_section(text: str, keywords: list[str], max_chars: int = 300) -> str | None:
    """Extract text following section header keywords."""
    text_lower = text.lower()
    for kw in keywords:
        idx = text_lower.find(kw.lower())
        if idx != -1:
            start = idx + len(kw)
            while start < len(text) and text[start] in " :\n\t":
                start += 1
            return text[start:start + max_chars].strip()
    return None


def extract_fields(text: str, schema: dict) -> dict:
    """Extract all schema fields using rule-based heuristics."""
    result = {}

    for field_name, field_props in schema.items():
        field_type = field_props.get("type", "string")
        choices = field_props.get("choices", [])

        # Date fields
        if "data" in field_name.lower() or field_type == "date":
            result[field_name] = extract_date(text)

        # Court type
        elif field_name == "typ_sadu":
            result[field_name] = extract_court_type(text)

        # Apelacja
        elif field_name == "apelacja":
            result[field_name] = extract_apelacja(text)

        # Instancja
        elif field_name == "instancja_sadu":
            if "odwoławczy" in text.lower() or "ii instancj" in text.lower() or "apelacyjn" in text.lower():
                result[field_name] = "Sąd odwoławczy"
            elif "i instancj" in text.lower() or "rejonow" in text.lower() or "okręgow" in text.lower():
                result[field_name] = "Sąd I instancji"
            else:
                result[field_name] = None

        # Swiss franc case detection
        elif field_name == "sprawa_frankowiczow":
            text_lower = text.lower()
            franc_kw = ["frank", "chf", "indeksow", "denomin", "walut"]
            if any(kw in text_lower for kw in franc_kw):
                result[field_name] = "Tak"
            else:
                result[field_name] = "Nie"

        # Binary Tak/Nie fields
        elif field_type == "enum" and set(choices) <= {"Tak", "Nie", None}:
            # Field-specific keywords
            field_kw_map = {
                "podstawa_prawna_podana": (["podstaw", "art."], ["brak podstaw"]),
                "modyfikacje_powodztwa": (["modyfikacj", "zmian", "rozszerzeni"], ["bez modyfikacji"]),
                "wspoluczestnictwo_powodowe": (["współuczestni", "powodów", "powódki"], []),
                "wspoluczestnictwo_pozwanego": (["współpozwan"], []),
                "wczesniejsze_skargi_do_rzecznika": (["rzecznik", "skargi"], []),
                "klauzula_niedozwolona": (["klauzul", "niedozwolon", "abuzyw"], []),
                "wpisana_do_rejestru_uokik": (["uokik", "rejestr"], []),
                "aneks_do_umowy": (["aneks"], []),
                "sesja_sadowa": (["rozprawa", "sesja", "posiedzeni"], []),
                "oswiadczenie_niewaznosci": (["nieważnoś", "unieważnieni"], []),
                "odwolanie_do_sn": (["sąd najwyższy", "kasacj"], []),
                "odwolanie_do_tsue": (["tsue", "trybunał sprawiedliwości", "c-260"], []),
                "zarzut_zatrzymania": (["zatrzym", "prawo zatrzym"], []),
                "zarzut_potracenia": (["potrąceni"], []),
                "odsetki_ustawowe": (["odsetk", "ustawow"], []),
                "zabezpieczenie_udzielone": (["zabezpiecz"], []),
            }
            pos_kw, neg_kw = field_kw_map.get(field_name, ([], []))
            result[field_name] = extract_boolean(text, pos_kw, neg_kw)

        # Non-binary enum fields
        elif field_type == "enum":
            result[field_name] = extract_enum(text, choices)

        # String fields with section-based extraction
        elif field_type == "string":
            section_map = {
                "podstawa_prawna": ["podstawa prawna", "podstawę prawną", "na podstawie art."],
                "rozstrzygniecie_sadu": ["rozstrzygnięcie", "orzeka", "postanawia"],
                "teoria_prawna": ["teoria", "koncepcj"],
                "szczegoly_wyniku_sprawy": ["wynik", "rozstrzygnięcie"],
                "strony_umowy": ["strony umowy", "kredytobiorc", "kredytodawc"],
                "umowa_kredytowa": ["umowa kredyt", "umow"],
                "przedmiot_aneksu": ["aneks", "przedmiot aneksu"],
                "rodzaj_zabezpieczenia": ["zabezpiecz"],
            }
            section_kws = section_map.get(field_name, [])
            if section_kws:
                result[field_name] = extract_section(text, section_kws)
            else:
                result[field_name] = None

        else:
            result[field_name] = None

    return result


def evaluate(predictions: list[dict], golds: list[dict], schema: dict) -> dict:
    """Evaluate predictions against gold, returning per-field accuracy."""
    field_scores = defaultdict(list)

    for pred, gold in zip(predictions, golds):
        for field_name in schema:
            pred_val = pred.get(field_name)
            gold_val = gold.get(field_name)

            # Normalize None/"None" comparisons
            if pred_val == "None":
                pred_val = None
            if gold_val == "None":
                gold_val = None

            # Score
            if pred_val is None and gold_val is None:
                score = 1.0
            elif pred_val is None or gold_val is None:
                score = 0.0
            elif str(pred_val).strip().lower() == str(gold_val).strip().lower():
                score = 1.0
            else:
                score = 0.0

            field_scores[field_name].append(score)

    # Aggregate
    results = {}
    for field_name, scores in field_scores.items():
        results[field_name] = {
            "accuracy": sum(scores) / len(scores) if scores else 0.0,
            "n": len(scores),
            "correct": sum(scores),
        }

    overall = sum(r["accuracy"] for r in results.values()) / len(results) if results else 0.0
    return {"overall": overall, "fields": results}


def main():
    # Load schema
    schema_path = Path("configs/ie_schema/swiss_franc_loans.yaml")
    with open(schema_path) as f:
        schema = yaml.safe_load(f)
    logger.info(f"Schema: {len(schema)} fields")

    # Load dataset from HF
    logger.info("Loading pl-swiss-franc-loans test set from HuggingFace...")
    ds = load_dataset("JuDDGES/pl-swiss-franc-loans", split="test")
    logger.info(f"Loaded {len(ds)} test samples")

    # Run extraction
    predictions = []
    golds = []
    for sample in ds:
        text = sample["context"]
        gold = json.loads(sample["output"])
        pred = extract_fields(text, schema)
        predictions.append(pred)
        golds.append(gold)

    # Evaluate
    results = evaluate(predictions, golds, schema)

    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"Rule-Based Baseline on pl-swiss-franc-loans (test, n={len(ds)})")
    logger.info(f"Overall accuracy: {results['overall']:.4f}")
    logger.info(f"{'='*60}")

    # Print per-field sorted by accuracy
    fields_sorted = sorted(results["fields"].items(), key=lambda x: x[1]["accuracy"], reverse=True)
    print(f"\n{'Field':<45} {'Accuracy':>8} {'Correct':>8} {'Total':>6}")
    print("-" * 70)
    for field_name, metrics in fields_sorted:
        field_type = schema[field_name].get("type", "string")
        print(f"{field_name:<45} {metrics['accuracy']:>8.3f} {int(metrics['correct']):>8} {metrics['n']:>6}  [{field_type}]")
    print("-" * 70)
    print(f"{'OVERALL':<45} {results['overall']:>8.4f}")

    # Save results
    output_dir = Path("data/experiments/neurips_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "rule_based_swiss_franc.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_file}")

    # Also run on en-appealcourt if available
    try:
        ds_en = load_dataset("JuDDGES/en-appealcourt", split="test")
        logger.info(f"\nEN dataset available: {len(ds_en)} samples — skipping (needs EN schema)")
    except Exception:
        pass


if __name__ == "__main__":
    main()
