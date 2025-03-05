import datetime
from collections import defaultdict
from dataclasses import dataclass

from juddges.utils.misc import parse_yaml

EMPTY_ANSWER = ""
DATE_FMT = "%Y-%m-%d"


@dataclass
class ParseResults:
    preds: dict[str, list[str]]
    golds: dict[str, list[str]]
    failed_preds_parse_mask: list[bool]
    num_preds_parse_errors: int


def parse_results(
    results: list[dict[str, str]],
) -> ParseResults:
    """Parses the results of the model into gold and predicted dictionaries.

    Args:
        results (list[dict[str, str]]): list of model and gold answers in format [{"answer": str, "gold": str}]

    Returns:
        ParseResults: parsed gold and predicted fields
    """
    res_pred: dict[str, list[str]] = defaultdict(list)
    res_gold: dict[str, list[str]] = defaultdict(list)
    failed_preds_parse_mask: list[bool] = []

    num_preds_parse_errors = 0
    for item in results:
        gold = _parse_item(item["gold"])
        assert gold is not None

        preds = _parse_item(item["answer"])
        if preds is None:
            num_preds_parse_errors += 1
            failed_preds_parse_mask.append(True)
            preds = dict.fromkeys(gold.keys(), EMPTY_ANSWER)
        else:
            failed_preds_parse_mask.append(False)

        # NOTE: it doesn't account for fields that were extra added by LLM
        for k in gold.keys():
            res_pred[k].append(preds.get(k, EMPTY_ANSWER))
            res_gold[k].append(gold[k])

    res_pred = dict(res_pred)
    res_gold = dict(res_gold)

    return ParseResults(
        preds=res_pred,
        golds=res_gold,
        failed_preds_parse_mask=failed_preds_parse_mask,
        num_preds_parse_errors=num_preds_parse_errors,
    )


def _parse_item(item: str) -> dict[str, str] | None:
    """Parses yaml model output to a dictionary.
    The following format is applied:
        - dates -> string formatted according to DATE_FMT
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
            data[k] = v.strftime(DATE_FMT)
        elif data[k] is None:
            data[k] = EMPTY_ANSWER
        else:
            data[k] = str(v)

    return data
