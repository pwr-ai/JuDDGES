"""Prompt-parity variant of run_exp_b_gpt_validation.py.

The original fresh re-extraction differs from the production annotation run in two
ways beyond model non-determinism: it uses a short generic prompt instead of the
production template, and it truncates the document to 15,000 characters. The
reported fresh-vs-original agreement of 0.607 therefore conflates non-determinism,
a prompt downgrade and context truncation.

This variant holds everything else constant (model, temperature, sample, scoring)
and changes only those two factors:
  - prompts with configs/prompt/info_extraction_annotated_json.yaml, as production does
  - passes the full document, untruncated

Comparing this run against the original isolates the contribution of prompt
provenance to the agreement gap.

Cost estimate: ~$4 (100 docs, untruncated input).
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import typer
import yaml
from datasets import load_dataset
from loguru import logger

MODEL = "gpt-4.1"
PROMPT_PATH = Path("configs/prompt/info_extraction_annotated_json.yaml")
SCHEMA_PATH = Path("configs/ie_schema/swiss_franc_loans.yaml")
OUTPUT_DIR = Path("data/experiments/neurips_results/exp_b_validation_prompt_parity")
MAX_RETRIES = 3


def build_prompt(template: str, text: str, schema: dict) -> str:
    """Render the production template. Schema is passed as YAML, as in production."""
    return template.format(schema=yaml.dump(schema, allow_unicode=True, sort_keys=False), context=text)


def extract_with_gpt(client, prompt: str, temperature: float) -> dict | None:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=4000,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            logger.warning(f"JSON parse error attempt {attempt + 1}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"API error attempt {attempt + 1}: {e}")
            time.sleep(2**attempt)
    return None


def score_field(pred_val, gold_val) -> float:
    """Identical to the original script, so the two runs are directly comparable."""
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


def main(
    sample_size: int = typer.Option(100, help="Number of documents to score."),
    temperature: float = typer.Option(0.1, help="Held equal to the original re-run."),
    dry_run: bool = typer.Option(False, help="Render one prompt and exit without calling the API."),
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    template = yaml.safe_load(PROMPT_PATH.read_text())["content"]
    schema = yaml.safe_load(SCHEMA_PATH.read_text())
    logger.info(f"Schema: {len(schema)} fields | prompt: {PROMPT_PATH}")

    logger.info("Loading pl-swiss-franc-loans test + annotated splits...")
    ds_test = load_dataset("JuDDGES/pl-swiss-franc-loans", split="test")
    ds_annotated = load_dataset("JuDDGES/pl-swiss-franc-loans", split="annotated")

    indices = list(range(min(sample_size, len(ds_test))))

    if dry_run:
        prompt = build_prompt(template, ds_test[0]["context"], schema)
        logger.info(f"Rendered prompt: {len(prompt)} chars (document {len(ds_test[0]['context'])} chars)")
        print(prompt[:1500])
        print("\n...[truncated for display only]...\n")
        return

    from openai import OpenAI  # imported lazily so --dry-run needs no API client

    client = OpenAI()
    scores_fresh_vs_orig = defaultdict(list)
    scores_fresh_vs_human = defaultdict(list)
    scores_orig_vs_human = defaultdict(list)
    parse_errors = 0
    results = []
    doc_chars = []

    for i, idx in enumerate(indices):
        text = ds_test[idx]["context"]
        doc_chars.append(len(text))
        orig_gold = json.loads(ds_test[idx]["output"])
        human_gold = json.loads(ds_annotated[idx]["output"])

        fresh = extract_with_gpt(client, build_prompt(template, text, schema), temperature)
        if fresh is None:
            parse_errors += 1
            continue

        doc_scores = {}
        for field in schema:
            s_fo = score_field(fresh.get(field), orig_gold.get(field))
            s_fh = score_field(fresh.get(field), human_gold.get(field))
            s_oh = score_field(orig_gold.get(field), human_gold.get(field))
            scores_fresh_vs_orig[field].append(s_fo)
            scores_fresh_vs_human[field].append(s_fh)
            scores_orig_vs_human[field].append(s_oh)
            doc_scores[field] = {"fresh_vs_orig": s_fo, "fresh_vs_human": s_fh, "orig_vs_human": s_oh}

        results.append({"idx": idx, "scores": doc_scores})

        if (i + 1) % 10 == 0:
            avg_fo = mean(s for sc in scores_fresh_vs_orig.values() for s in sc)
            avg_fh = mean(s for sc in scores_fresh_vs_human.values() for s in sc)
            logger.info(f"  [{i + 1}/{len(indices)}] Fresh-Orig: {avg_fo:.3f}  Fresh-Human: {avg_fh:.3f}  errors: {parse_errors}")

    summary = {}
    print(f"\n{'Field':<35} {'Fresh-Orig':>11} {'Fresh-Hum':>11} {'Orig-Hum':>11}")
    print("-" * 72)
    for field in schema:
        fo, fh, oh = scores_fresh_vs_orig[field], scores_fresh_vs_human[field], scores_orig_vs_human[field]
        if fo:
            summary[field] = {
                "fresh_vs_orig": {"mean": mean(fo), "std": stdev(fo) if len(fo) > 1 else 0},
                "fresh_vs_human": {"mean": mean(fh), "std": stdev(fh) if len(fh) > 1 else 0},
                "orig_vs_human": {"mean": mean(oh), "std": stdev(oh) if len(oh) > 1 else 0},
            }
            print(f"{field:<35} {mean(fo):>11.3f} {mean(fh):>11.3f} {mean(oh):>11.3f}")

    overall = {
        "fresh_vs_orig": mean(s for sc in scores_fresh_vs_orig.values() for s in sc),
        "fresh_vs_human": mean(s for sc in scores_fresh_vs_human.values() for s in sc),
        "orig_vs_human": mean(s for sc in scores_orig_vs_human.values() for s in sc),
    }
    print("-" * 72)
    print(f"{'OVERALL':<35} {overall['fresh_vs_orig']:>11.3f} {overall['fresh_vs_human']:>11.3f} {overall['orig_vs_human']:>11.3f}")

    output = {
        "model": MODEL,
        "variant": "prompt_parity",
        "prompt_source": str(PROMPT_PATH),
        "truncation": None,
        "temperature": temperature,
        "n_docs": len(results),
        "n_parse_errors": parse_errors,
        "doc_chars": {"mean": mean(doc_chars), "max": max(doc_chars)} if doc_chars else None,
        "overall": overall,
        "per_field": summary,
    }
    (OUTPUT_DIR / "validation_summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
    (OUTPUT_DIR / "per_doc_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {OUTPUT_DIR}/validation_summary.json")


if __name__ == "__main__":
    typer.run(main)
