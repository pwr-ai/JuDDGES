#!/usr/bin/env python
"""EXP B: Annotation Validation — GPT-4.1 re-extraction on 100 docs.

Re-runs GPT-4.1 extraction with Schema A on 100 docs from pl-swiss-franc-loans.
Compares:
  (a) Fresh GPT-4.1 extraction (this run)
  (b) GPT-4.1 test-split gold (original extraction)
  (c) Human-reviewed annotated split

This measures:
- GPT-4.1 self-consistency (a vs b)
- Pre-annotation bias: how close is original GPT-4.1 to human review (b vs c)
- Fresh extraction vs human gold (a vs c)

Cost estimate: ~$3 (100 docs × ~10K input + 2K output tokens)

Usage:
    PYTHONPATH=. python scripts/neurips_experiments/run_exp_b_gpt_validation.py
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import yaml
from datasets import load_dataset
from loguru import logger
from openai import OpenAI

SAMPLE_SIZE = 100
MODEL = "gpt-4.1"
OUTPUT_DIR = Path("data/experiments/neurips_results/exp_b_validation")
MAX_RETRIES = 3


def build_prompt(text: str, schema: dict) -> str:
    fields_desc = []
    for name, props in schema.items():
        ftype = props.get("type", "string")
        choices = props.get("choices", [])
        desc = props.get("description", "")
        if choices:
            fields_desc.append(f"- {name} ({ftype}, choices: {choices}): {desc}")
        else:
            fields_desc.append(f"- {name} ({ftype}): {desc}")

    return (
        "Ekstraktuj poniższe pola z polskiego orzeczenia sądowego.\n"
        "Zwróć TYLKO poprawny obiekt JSON z tymi polami:\n\n"
        + "\n".join(fields_desc)
        + "\n\nZASADY:\n"
        "- Ekstraktuj informacje WYŁĄCZNIE z tekstu dokumentu\n"
        "- Dla pól enum: używaj TYLKO wartości z listy choices\n"
        "- Daty: format YYYY-MM-DD\n"
        "- Puste pola: pusty string '' dla string, null dla brakujących\n\n"
        f"DOKUMENT:\n{text[:15000]}\n\nJSON:"
    )


def extract_with_gpt(client: OpenAI, text: str, schema: dict) -> dict | None:
    prompt = build_prompt(text, schema)
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


def score_field(pred_val, gold_val) -> float:
    if pred_val == "None": pred_val = None
    if gold_val == "None": gold_val = None
    if pred_val is None and gold_val is None: return 1.0
    if pred_val is None or gold_val is None: return 0.0
    if str(pred_val).strip().lower() == str(gold_val).strip().lower(): return 1.0
    return 0.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    with open("configs/ie_schema/swiss_franc_loans.yaml") as f:
        schema = yaml.safe_load(f)
    logger.info(f"Schema: {len(schema)} fields")

    logger.info("Loading pl-swiss-franc-loans test + annotated splits...")
    ds_test = load_dataset("JuDDGES/pl-swiss-franc-loans", split="test")
    ds_annotated = load_dataset("JuDDGES/pl-swiss-franc-loans", split="annotated")

    indices = list(range(min(SAMPLE_SIZE, len(ds_test))))

    scores_fresh_vs_orig = defaultdict(list)   # (a) vs (b)
    scores_fresh_vs_human = defaultdict(list)  # (a) vs (c)
    scores_orig_vs_human = defaultdict(list)   # (b) vs (c)
    parse_errors = 0
    results = []

    for i, idx in enumerate(indices):
        text = ds_test[idx]["context"]
        orig_gold = json.loads(ds_test[idx]["output"])       # (b) original GPT-4.1
        human_gold = json.loads(ds_annotated[idx]["output"]) # (c) human-reviewed

        # (a) Fresh extraction
        fresh = extract_with_gpt(client, text, schema)
        if fresh is None:
            parse_errors += 1
            continue

        doc_scores = {}
        for field in schema:
            f_val = fresh.get(field)
            o_val = orig_gold.get(field)
            h_val = human_gold.get(field)

            s_fo = score_field(f_val, o_val)
            s_fh = score_field(f_val, h_val)
            s_oh = score_field(o_val, h_val)

            scores_fresh_vs_orig[field].append(s_fo)
            scores_fresh_vs_human[field].append(s_fh)
            scores_orig_vs_human[field].append(s_oh)

            doc_scores[field] = {"fresh_vs_orig": s_fo, "fresh_vs_human": s_fh, "orig_vs_human": s_oh}

        results.append({"idx": idx, "scores": doc_scores})

        if (i + 1) % 10 == 0:
            avg_fo = mean(s for sc in scores_fresh_vs_orig.values() for s in sc)
            avg_fh = mean(s for sc in scores_fresh_vs_human.values() for s in sc)
            avg_oh = mean(s for sc in scores_orig_vs_human.values() for s in sc)
            logger.info(f"  [{i+1}/{len(indices)}] Fresh↔Orig: {avg_fo:.3f}  Fresh↔Human: {avg_fh:.3f}  Orig↔Human: {avg_oh:.3f}  errors: {parse_errors}")

    # Aggregate
    logger.info(f"\n{'='*80}")
    logger.info(f"3-Way Validation: GPT-4.1 Fresh vs Original vs Human (n={len(results)})")
    logger.info(f"{'='*80}")

    summary = {}
    print(f"\n{'Field':<35} {'Fresh↔Orig':>10} {'Fresh↔Hum':>10} {'Orig↔Hum':>10}")
    print("-" * 70)
    for field in schema:
        fo = scores_fresh_vs_orig[field]
        fh = scores_fresh_vs_human[field]
        oh = scores_orig_vs_human[field]
        if fo:
            summary[field] = {
                "fresh_vs_orig": {"mean": mean(fo), "std": stdev(fo) if len(fo) > 1 else 0},
                "fresh_vs_human": {"mean": mean(fh), "std": stdev(fh) if len(fh) > 1 else 0},
                "orig_vs_human": {"mean": mean(oh), "std": stdev(oh) if len(oh) > 1 else 0},
            }
            print(f"{field:<35} {mean(fo):>10.3f} {mean(fh):>10.3f} {mean(oh):>10.3f}")

    overall_fo = mean(s for sc in scores_fresh_vs_orig.values() for s in sc)
    overall_fh = mean(s for sc in scores_fresh_vs_human.values() for s in sc)
    overall_oh = mean(s for sc in scores_orig_vs_human.values() for s in sc)
    print("-" * 70)
    print(f"{'OVERALL':<35} {overall_fo:>10.3f} {overall_fh:>10.3f} {overall_oh:>10.3f}")

    output = {
        "model": MODEL,
        "n_docs": len(results),
        "n_parse_errors": parse_errors,
        "overall": {
            "fresh_vs_orig": overall_fo,
            "fresh_vs_human": overall_fh,
            "orig_vs_human": overall_oh,
        },
        "per_field": summary,
    }
    with open(OUTPUT_DIR / "validation_summary.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
