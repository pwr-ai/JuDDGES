from torchmetrics.functional.text import chrf_score
import datetime
from collections import defaultdict
from juddges.utils.misc import parse_yaml

EMPTY_ANSWER = ""


def evaluate_extraction(results: list[dict[str, str]]) -> dict[str, float]:
    """Evaluates information extraction by computing metrics per each field."""
    res_gold, res_pred = parse_results(results)
    full_text_chrf = full_text_chrf_score(results)
    per_field_chrf_score = extraction_chrf_score(preds=res_pred, gold=res_gold)
    return {"full_text_chrf": full_text_chrf, "field_chrf": per_field_chrf_score}


def full_text_chrf_score(results: list[dict[str, str]]) -> float:
    preds, golds = [r["answer"] for r in results], [r["gold"] for r in results]
    return chrf_score(preds=preds, target=golds, n_word_order=0).item()


def extraction_chrf_score(
    preds: dict[str, list[str]],
    gold: dict[str, list[str]],
) -> dict[str, float]:
    per_field_chrf_score = {
        key: chrf_score(preds=preds[key], target=gold[key], n_word_order=0).item()
        for key in gold.keys()
    }
    return per_field_chrf_score


def parse_results(
    results: list[dict[str, str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    res_pred: dict[str, list[str]] = defaultdict(list)
    res_gold: dict[str, list[str]] = defaultdict(list)

    for item in results:
        gold = _parse_item(item["gold"])
        assert gold is not None

        ans = _parse_item(item["answer"])
        if ans is None:
            ans = dict.fromkeys(gold.keys(), EMPTY_ANSWER)

        for k in gold.keys():
            res_pred[k].append(ans.get(k, EMPTY_ANSWER))
            res_gold[k].append(gold[k])

    return res_gold, res_pred


def _parse_item(item: str) -> dict[str, str] | None:
    """Parses yaml model output to a dictionary.
    The following format is applied:
        - dates -> strings
        - lists -> comma-separated sorted strings
        - None  -> EMPTY_ANSWER
        - If the input cannot be parsed, returns None.
    """
    try:
        data = parse_yaml(item)
    except Exception:
        return None

    if (data is None) or (not isinstance(data, dict)):
        return None

    for k, v in data.items():
        if isinstance(v, list):
            # list values might be None (need to cast to string)
            data[k] = ", ".join(sorted(str(v_i) for v_i in v))
        elif isinstance(v, datetime.date):
            data[k] = v.strftime("%Y-%m-%d")
        elif data[k] is None:
            data[k] = EMPTY_ANSWER
        else:
            data[k] = str(v)

    return data
