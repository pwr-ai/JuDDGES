from typing import Any
from torchmetrics.functional.text import chrf_score
import datetime
from collections import defaultdict
from juddges.utils.misc import parse_yaml

EMPTY_ANSWER = ""


def evaluate_extraction(results: list[dict[str, str]]) -> dict[str, float]:
    res_gold, res_pred = parse_results(results)
    per_field_chrf_score = extraction_chrf_score(preds=res_pred, gold=res_gold)
    return per_field_chrf_score


def extraction_chrf_score(
    preds: dict[str, list[str]],
    gold: dict[str, list[str]],
) -> dict[str, Any]:
    per_field_chrf_score = {
        key: chrf_score(preds=preds[key], target=gold[key], n_word_order=0).item()
        for key in gold.keys()
    }
    return {"metric": "chrf", "field_scores": per_field_chrf_score}


def parse_results(
    results: list[dict[str, str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    res_pred: dict[str, list[str]] = defaultdict(list)
    res_gold: dict[str, list[str]] = defaultdict(list)

    for item in results:
        gold = _parse_item(item["gold"])
        ans = _parse_item(item["answer"])
        if ans is None:
            ans = dict.fromkeys(gold.keys(), EMPTY_ANSWER)
        assert gold is not None

        for k in gold.keys():
            res_pred[k].append(ans[k])
            res_gold[k].append(gold[k])

    return res_gold, res_pred


def _parse_item(item: str) -> dict[str, str] | None:
    try:
        data = parse_yaml(item)
    except Exception:
        return None

    for k, v in data.items():
        if isinstance(v, list):
            data[k] = ", ".join(sorted(v))
        elif isinstance(v, datetime.date):
            data[k] = v.strftime("%Y-%m-%d")
        elif data[k] is None:
            data[k] = EMPTY_ANSWER

        assert isinstance(data[k], str)

    return data


if __name__ == "__main__":
    import json

    f_name = "data/results/pl/zero_shot/results_llama_3_8B.json"
    with open(f_name) as file:
        res = json.load(file)
