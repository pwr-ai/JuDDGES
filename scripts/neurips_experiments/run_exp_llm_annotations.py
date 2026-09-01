#!/usr/bin/env python
"""LLM-as-Human Annotations: GPT-5.4-pro + Claude Sonnet 4.6.

Runs two independent frontier models on 100 docs from pl-swiss-franc-loans
to create synthetic "human-level" annotations without GPT-4 circularity.

Produces:
  - Per-model extractions (JSON)
  - Inter-annotator agreement (GPT-5.4 vs Sonnet)
  - 4-way comparison: GPT-5.4 vs Sonnet vs GPT-4.1-gold vs Human-reviewed

Cost estimate: ~$19 (Option B)

Usage:
    PYTHONPATH=. python scripts/neurips_experiments/run_exp_llm_annotations.py
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import anthropic
import yaml
from datasets import load_dataset
from loguru import logger
from openai import OpenAI

N_DOCS = 100
OUTPUT_DIR = Path("data/experiments/neurips_results/exp_llm_annotations")
MAX_RETRIES = 3


def build_extraction_prompt(text: str, schema: dict) -> str:
    """Build a detailed extraction prompt with Schema A."""
    fields_desc = []
    for name, props in schema.items():
        ftype = props.get("type", "string")
        choices = props.get("choices", [])
        desc = props.get("description", "")
        if choices:
            fields_desc.append(f"- {name} (typ: {ftype}, dozwolone wartości: {choices}): {desc}")
        else:
            fields_desc.append(f"- {name} (typ: {ftype}): {desc}")

    fields_str = "\n".join(fields_desc)

    return f"""Jesteś ekspertem prawnym analizującym polskie orzeczenia sądowe dotyczące kredytów frankowych.

Twoim zadaniem jest precyzyjna ekstrakcja strukturalnych informacji z poniższego dokumentu prawnego.

POLA DO EKSTRAKCJI:
{fields_str}

SZCZEGÓŁOWE INSTRUKCJE:
1. Ekstraktuj informacje WYŁĄCZNIE z tekstu dokumentu — nie domyślaj się
2. Dla pól typu enum: używaj TYLKO wartości z podanej listy dozwolonych wartości
3. Daty: zawsze w formacie YYYY-MM-DD
4. Jeśli informacja nie jest dostępna w dokumencie, użyj:
   - Dla pól string: pusty string ""
   - Dla pól enum: null
   - Dla pól list: []
5. Bądź precyzyjny — rozróżniaj między tym co jest explicite stwierdzone a tym co jest domniemane
6. Pole "oswiadczenie_niewaznosci" oznacza czy SĄD stwierdził nieważność, nie czy strona o to wnosiła
7. Pole "zarzut_zatrzymania" — czy podniesiono zarzut prawa zatrzymania (explicite w tekście)

DOKUMENT:
{text}

Zwróć TYLKO poprawny obiekt JSON z powyższymi polami. Bez dodatkowego tekstu."""


def extract_gpt54(client: OpenAI, text: str, schema: dict) -> dict | None:
    """Extract with GPT-5.4-pro."""
    prompt = build_extraction_prompt(text, schema)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": "You are a Polish legal expert performing precise information extraction. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_completion_tokens=4000,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            logger.warning(f"GPT-5.4 JSON parse error attempt {attempt + 1}")
        except Exception as e:
            logger.warning(f"GPT-5.4 API error attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    return None


def extract_sonnet(client: anthropic.Anthropic, text: str, schema: dict) -> dict | None:
    """Extract with Claude Sonnet 4.6."""
    prompt = build_extraction_prompt(text, schema)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt + "\n\nJSON:"},
                ],
            )
            text_response = response.content[0].text.strip()
            # Clean markdown fences
            if text_response.startswith("```"):
                lines = text_response.split("\n")
                text_response = "\n".join(lines[1:])
                if text_response.endswith("```"):
                    text_response = text_response[:-3]
            return json.loads(text_response)
        except json.JSONDecodeError:
            logger.warning(f"Sonnet JSON parse error attempt {attempt + 1}")
        except Exception as e:
            logger.warning(f"Sonnet API error attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    return None


def score_field(val_a, val_b) -> float:
    """Simple exact-match accuracy for a field pair."""
    if val_a == "None": val_a = None
    if val_b == "None": val_b = None
    if val_a is None and val_b is None: return 1.0
    if val_a is None or val_b is None: return 0.0
    if str(val_a).strip().lower() == str(val_b).strip().lower(): return 1.0
    return 0.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    openai_client = OpenAI()
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    with open("configs/ie_schema/swiss_franc_loans.yaml") as f:
        schema = yaml.safe_load(f)
    logger.info(f"Schema: {len(schema)} fields")

    logger.info("Loading pl-swiss-franc-loans...")
    ds_test = load_dataset("JuDDGES/pl-swiss-franc-loans", split="test")
    ds_annotated = load_dataset("JuDDGES/pl-swiss-franc-loans", split="annotated")

    indices = list(range(min(N_DOCS, len(ds_test))))

    # Storage
    all_gpt54 = []
    all_sonnet = []
    scores = {
        "gpt54_vs_sonnet": defaultdict(list),
        "gpt54_vs_gpt4gold": defaultdict(list),
        "gpt54_vs_human": defaultdict(list),
        "sonnet_vs_gpt4gold": defaultdict(list),
        "sonnet_vs_human": defaultdict(list),
        "gpt4gold_vs_human": defaultdict(list),
    }
    errors_gpt54 = 0
    errors_sonnet = 0

    for i, idx in enumerate(indices):
        text = ds_test[idx]["context"]
        gpt4_gold = json.loads(ds_test[idx]["output"])
        human_gold = json.loads(ds_annotated[idx]["output"])

        # Extract with both models
        gpt54_result = extract_gpt54(openai_client, text, schema)
        sonnet_result = extract_sonnet(anthropic_client, text, schema)

        if gpt54_result is None:
            errors_gpt54 += 1
        if sonnet_result is None:
            errors_sonnet += 1

        if gpt54_result is None or sonnet_result is None:
            continue

        all_gpt54.append({"idx": idx, "extraction": gpt54_result})
        all_sonnet.append({"idx": idx, "extraction": sonnet_result})

        # 4-way scoring
        for field in schema:
            g54 = gpt54_result.get(field)
            son = sonnet_result.get(field)
            g4g = gpt4_gold.get(field)
            hum = human_gold.get(field)

            scores["gpt54_vs_sonnet"][field].append(score_field(g54, son))
            scores["gpt54_vs_gpt4gold"][field].append(score_field(g54, g4g))
            scores["gpt54_vs_human"][field].append(score_field(g54, hum))
            scores["sonnet_vs_gpt4gold"][field].append(score_field(son, g4g))
            scores["sonnet_vs_human"][field].append(score_field(son, hum))
            scores["gpt4gold_vs_human"][field].append(score_field(g4g, hum))

        if (i + 1) % 10 == 0:
            iaa = mean(s for sc in scores["gpt54_vs_sonnet"].values() for s in sc)
            g54h = mean(s for sc in scores["gpt54_vs_human"].values() for s in sc)
            sonh = mean(s for sc in scores["sonnet_vs_human"].values() for s in sc)
            logger.info(
                f"  [{i+1}/{len(indices)}] "
                f"GPT5.4↔Sonnet: {iaa:.3f}  "
                f"GPT5.4↔Human: {g54h:.3f}  "
                f"Sonnet↔Human: {sonh:.3f}  "
                f"errors: gpt54={errors_gpt54}, sonnet={errors_sonnet}"
            )

    # Aggregate
    n_success = len(all_gpt54)
    logger.info(f"\n{'='*80}")
    logger.info(f"LLM-as-Human Annotations (n={n_success})")
    logger.info(f"Parse errors: GPT-5.4={errors_gpt54}, Sonnet={errors_sonnet}")
    logger.info(f"{'='*80}")

    # Overall scores
    overall = {}
    for comparison_name, field_scores in scores.items():
        all_s = [s for sc in field_scores.values() for s in sc]
        overall[comparison_name] = mean(all_s) if all_s else 0.0

    print(f"\n{'Comparison':<30} {'Agreement':>10}")
    print("-" * 42)
    display_names = {
        "gpt54_vs_sonnet": "GPT-5.4 ↔ Sonnet (IAA)",
        "gpt54_vs_gpt4gold": "GPT-5.4 ↔ GPT-4.1 gold",
        "gpt54_vs_human": "GPT-5.4 ↔ Human-reviewed",
        "sonnet_vs_gpt4gold": "Sonnet ↔ GPT-4.1 gold",
        "sonnet_vs_human": "Sonnet ↔ Human-reviewed",
        "gpt4gold_vs_human": "GPT-4.1 ↔ Human (baseline)",
    }
    for key, name in display_names.items():
        print(f"{name:<30} {overall[key]:>10.3f}")

    # Per-field detail for IAA
    print(f"\n{'Field':<35} {'GPT5.4↔Son':>10} {'GPT5.4↔Hum':>10} {'Son↔Hum':>10}")
    print("-" * 70)
    for field in schema:
        iaa = scores["gpt54_vs_sonnet"][field]
        g54h = scores["gpt54_vs_human"][field]
        sonh = scores["sonnet_vs_human"][field]
        if iaa:
            print(f"{field:<35} {mean(iaa):>10.3f} {mean(g54h):>10.3f} {mean(sonh):>10.3f}")

    # Save
    summary = {
        "n_docs": n_success,
        "errors_gpt54": errors_gpt54,
        "errors_sonnet": errors_sonnet,
        "overall_agreement": overall,
        "per_field": {
            field: {
                comp: {"mean": mean(scores[comp][field]), "n": len(scores[comp][field])}
                for comp in scores
            }
            for field in schema
            if scores["gpt54_vs_sonnet"][field]
        },
    }
    with open(OUTPUT_DIR / "annotation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "gpt54_extractions.json", "w") as f:
        json.dump(all_gpt54, f, indent=2, ensure_ascii=False, default=str)

    with open(OUTPUT_DIR / "sonnet_extractions.json", "w") as f:
        json.dump(all_sonnet, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
