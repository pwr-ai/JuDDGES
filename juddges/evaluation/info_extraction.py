from torchmetrics.functional.text import chrf_score

from juddges.evaluation.parse import parse_results


def evaluate_extraction(results: list[dict[str, str]]) -> dict[str, float | dict[str, float]]:
    """Evaluates information extraction by computing metrics per each field."""
    res_gold, res_pred = parse_results(results)
    full_text_chrf = full_text_chrf_score(results)
    per_field_chrf_score = extraction_chrf_score(preds=res_pred, gold=res_gold)
    return {"full_text_chrf": full_text_chrf, "field_chrf": per_field_chrf_score}


def full_text_chrf_score(results: list[dict[str, str]]) -> float:
    preds, golds = [r["answer"] for r in results], [r["gold"] for r in results]
    return chrf_score(preds=preds, target=golds, n_word_order=0).item()  # type: ignore


def extraction_chrf_score(
    preds: dict[str, list[str]],
    gold: dict[str, list[str]],
) -> dict[str, float]:
    per_field_chrf_score = {
        key: chrf_score(preds=preds[key], target=gold[key], n_word_order=0).item()  # type: ignore
        for key in gold.keys()
    }
    return per_field_chrf_score
