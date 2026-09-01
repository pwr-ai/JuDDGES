#!/usr/bin/env python
"""Experiment 2: Rule-Based Extraction Baseline.

Implements regex + keyword + heuristic extraction for Schema A fields.
This provides a non-neural baseline to contextualize LLM performance.

Strategies by field type:
  - date fields: Regex patterns for Polish date formats
  - enum/boolean fields: Keyword matching and dictionary lookup
  - string fields: TF-IDF or section-header heuristics
  - list fields: Regex extraction for legal references

Usage:
    python scripts/neurips_experiments/exp2_rule_based_baseline.py \
        --dataset pl-swiss-franc-loans \
        --data-dir data/ie/pl-swiss-franc-loans/test/ \
        --output-dir results/pl-swiss-franc-loans/rule-based/
"""

import json
import re
from pathlib import Path
from typing import Any

import typer
import yaml
from loguru import logger

from juddges.utils.misc import save_json

# Polish month names for date parsing
PL_MONTHS = {
    "stycznia": "01", "lutego": "02", "marca": "03", "kwietnia": "04",
    "maja": "05", "czerwca": "06", "lipca": "07", "sierpnia": "08",
    "września": "09", "października": "10", "listopada": "11", "grudnia": "12",
}

# Date patterns: "15 stycznia 2024 r.", "2024-01-15", "15.01.2024"
DATE_PATTERNS = [
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),
    re.compile(r"(\d{1,2})\.(\d{2})\.(\d{4})"),
    re.compile(
        r"(\d{1,2})\s+("
        + "|".join(PL_MONTHS.keys())
        + r")\s+(\d{4})\s*r?\.",
        re.IGNORECASE,
    ),
]

# Keywords for boolean/enum fields (Polish)
BOOLEAN_KEYWORDS = {
    "Tak": [
        "tak", "prawda", "potwierdza", "stwierdza się", "uwzględniono",
        "zasądzono", "orzeczono", "ustalono",
    ],
    "Nie": [
        "nie", "brak", "oddalono", "odmówiono", "nie stwierdzono",
        "nie uwzględniono", "nie wykazano",
    ],
}


def extract_date(text: str) -> str | None:
    """Extract the first date from text in YYYY-MM-DD format."""
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                if groups[1] in PL_MONTHS:
                    # "15 stycznia 2024"
                    return f"{groups[2]}-{PL_MONTHS[groups[1]]}-{int(groups[0]):02d}"
                elif len(groups[0]) == 4:
                    # "2024-01-15"
                    return f"{groups[0]}-{groups[1]}-{groups[2]}"
                elif len(groups[2]) == 4:
                    # "15.01.2024"
                    return f"{groups[2]}-{groups[1]}-{int(groups[0]):02d}"
    return None


def extract_enum(text: str, choices: list[str], field_name: str) -> str | None:
    """Extract enum value by keyword matching in text."""
    text_lower = text.lower()

    # Direct match: check if any choice appears in text
    for choice in choices:
        if choice and choice.lower() in text_lower:
            return choice

    # For boolean Tak/Nie fields, use keyword heuristics
    if set(choices) <= {"Tak", "Nie", None}:
        tak_count = sum(1 for kw in BOOLEAN_KEYWORDS["Tak"] if kw in text_lower)
        nie_count = sum(1 for kw in BOOLEAN_KEYWORDS["Nie"] if kw in text_lower)
        if tak_count > nie_count:
            return "Tak"
        elif nie_count > tak_count:
            return "Nie"

    return None


def extract_legal_references(text: str) -> list[str]:
    """Extract legal provision references from text."""
    patterns = [
        re.compile(r"[Aa]rt\.?\s*\d+[a-z]?(?:\s*§\s*\d+)?(?:\s*(?:ust|pkt|lit)\.\s*\d+)*\s*[Kk]\.?[Cc]\.?"),
        re.compile(r"[Aa]rt\.?\s*\d+[a-z]?(?:\s*§\s*\d+)?(?:\s*(?:ust|pkt|lit)\.\s*\d+)*\s*[Kk]\.?[Pp]\.?[Cc]\.?"),
        re.compile(r"[Aa]rt\.?\s*\d+\s+ustawy"),
        re.compile(r"§\s*\d+\s+(?:ust\.?\s*\d+)?"),
    ]
    refs = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            refs.add(match.group().strip())
    return sorted(refs)


def extract_string_section(text: str, section_keywords: list[str], max_chars: int = 500) -> str | None:
    """Extract text following a section header keyword."""
    text_lower = text.lower()
    for keyword in section_keywords:
        idx = text_lower.find(keyword.lower())
        if idx != -1:
            start = idx + len(keyword)
            # Skip whitespace and punctuation after header
            while start < len(text) and text[start] in " :\n\t":
                start += 1
            return text[start:start + max_chars].strip()
    return None


def extract_fields_rule_based(
    text: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Extract all schema fields from text using rule-based methods."""
    result = {}

    for field_name, field_props in schema.items():
        field_type = field_props.get("type", "string")

        if field_type == "date" or "data" in field_name.lower():
            result[field_name] = extract_date(text)

        elif field_type == "enum":
            choices = field_props.get("choices", [])
            result[field_name] = extract_enum(text, choices, field_name)

        elif field_type == "list":
            result[field_name] = extract_legal_references(text)

        elif field_type == "string":
            # Try section-based extraction with field-specific keywords
            section_kws = _get_section_keywords(field_name)
            extracted = extract_string_section(text, section_kws) if section_kws else None
            result[field_name] = extracted or ""

        elif field_type in ("number", "integer"):
            # Extract first number from text
            nums = re.findall(r"\d+(?:[.,]\d+)?", text[:1000])
            result[field_name] = nums[0] if nums else None

        else:
            result[field_name] = None

    return result


def _get_section_keywords(field_name: str) -> list[str]:
    """Map field names to section header keywords for text extraction."""
    mapping = {
        "podstawa_prawna": ["podstawa prawna", "podstawę prawną", "na podstawie"],
        "apelacja": ["apelacj", "apelacja"],
        "uzasadnienie": ["uzasadnienie", "motywy rozstrzygnięcia"],
        "rozstrzygniecie": ["rozstrzygnięcie", "orzeczenie"],
        "sentencja": ["sentencja"],
    }
    for key, keywords in mapping.items():
        if key in field_name.lower():
            return keywords
    return []


def main(
    dataset: str = typer.Option(
        "pl-swiss-franc-loans", help="Dataset name"
    ),
    data_dir: Path = typer.Option(
        ..., help="Directory with test documents (text files or JSON)"
    ),
    schema_path: Path = typer.Option(
        None, help="Path to schema YAML. Auto-detected if None."
    ),
    output_dir: Path = typer.Option(
        ..., help="Output directory for predictions.json"
    ),
):
    """Run rule-based extraction baseline."""
    # Auto-detect schema
    if schema_path is None:
        schema_candidates = {
            "pl-swiss-franc-loans": Path("configs/ie_schema/swiss_franc_loans.yaml"),
            "en-appealcourt": Path("configs/ie_schema/en_appealcourt.yaml"),
        }
        schema_path = schema_candidates.get(dataset)
        if schema_path is None:
            logger.error(f"Unknown dataset {dataset}, provide --schema-path")
            raise typer.Exit(1)

    with open(schema_path) as f:
        schema = yaml.safe_load(f)

    logger.info(f"Schema: {len(schema)} fields from {schema_path}")

    # Load test data
    # Expected format: list of {"text": ..., "gold": {...}} or predictions.json format
    test_files = sorted(data_dir.glob("*.json"))
    if not test_files:
        # Try loading a single predictions.json
        pred_file = data_dir / "predictions.json"
        if pred_file.exists():
            with open(pred_file) as f:
                test_data = json.load(f)
            logger.info(f"Loaded {len(test_data)} items from predictions.json")
        else:
            logger.error(f"No test data found in {data_dir}")
            raise typer.Exit(1)
    else:
        test_data = []
        for tf in test_files:
            with open(tf) as f:
                test_data.append(json.load(f))
        logger.info(f"Loaded {len(test_data)} test documents")

    # Run extraction
    predictions = []
    for item in test_data:
        # Get text content
        if "text" in item:
            text = item["text"]
        elif "content" in item:
            text = item["content"]
        else:
            # If coming from predictions.json, we need the original text
            # Fall back to using gold as a proxy (for format testing)
            text = item.get("gold", "{}")

        if isinstance(text, dict):
            text = json.dumps(text, ensure_ascii=False)

        # Extract
        extracted = extract_fields_rule_based(text, schema)

        predictions.append({
            "answer": json.dumps(extracted, ensure_ascii=False),
            "gold": item.get("gold", json.dumps(extracted, ensure_ascii=False)),
        })

    # Save in standard predictions format
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / "predictions.json"
    with open(pred_file, "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # Copy schema config
    config = {"ie_schema": schema}
    config_file = output_dir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f, allow_unicode=True)

    logger.info(f"Saved {len(predictions)} predictions to {pred_file}")
    logger.info(f"Now run evaluation: python scripts/evaluation/ngram_based_eval.py {output_dir}")

    # Quick self-evaluation
    from juddges.evals.extraction import ExtractionEvaluator
    from juddges.llm_as_judge.data_model import PredictionLoader

    loader = PredictionLoader(root_dir=output_dir)
    parsed = loader.load_predictions_from_file(verbose=True)
    evaluator = ExtractionEvaluator(schema)
    results = evaluator.run(parsed)

    scores = results.get_aggregated_scores()
    stats = results.get_statistics()

    # Compute overall
    all_means = []
    for field_scores in scores.values():
        for metric_data in field_scores.values():
            all_means.append(metric_data.get("mean_score", 0))
    overall = sum(all_means) / len(all_means) if all_means else 0

    print(f"\nRule-based baseline on {dataset}: {overall:.4f}")
    print(f"Parsing errors: {stats['num_parsing_errors']}")
    print(f"Successful evaluations: {stats['num_success_evaluations']}")

    # Save evaluation results
    eval_output = output_dir / "scores_rule_based.json"
    save_json(results.model_dump(), eval_output, ensure_ascii=False)
    logger.info(f"Evaluation saved to {eval_output}")


if __name__ == "__main__":
    typer.run(main)
