from typing import Any

import pandas as pd
from flask import json

from juddges.llm_as_judge.data_model import PredictionLoader


def llm_as_judge_avg_scores(
    pred_loader: PredictionLoader,
) -> pd.DataFrame:
    judge_res = json.loads(pred_loader.llm_judge_scores_file.read_text())

    avg_scores = []
    for field_name, field_res in judge_res["aggregated_scores"].items():
        avg_scores.append(
            {
                "field": field_name,
                "mean_judge_score": field_res["score"]["mean_score"],
                "se_judge_score": field_res["score"]["standard_error"],
            }
        )
    return pd.DataFrame(avg_scores).set_index("field")


def llm_as_judge_detailed_results(
    pred_loader: PredictionLoader,
) -> pd.DataFrame:
    judge_res = json.loads(pred_loader.llm_judge_scores_file.read_text())

    detailed_judge_res = []
    for result in judge_res["all_results"]:
        item_res = {
            "status": result["status"],
            "error": result["error"],
            "missing_keys": result["missing_keys"],
            "extra_keys": result["extra_keys"],
        }
        for field, field_res in result["result"].items():
            item_res[field] = field_res["score"]
        detailed_judge_res.append(item_res)
    detailed_judge_res = pd.DataFrame(detailed_judge_res)


def ngram_avg_scores(
    pred_loader: PredictionLoader,
) -> pd.DataFrame:
    ngram_res = json.loads(pred_loader.ngram_scores_file.read_text())

    df_data = []
    for field, field_res in ngram_res["aggregated_scores"].items():
        df_data.append(
            {
                "field": field,
                **get_metric_values(field_res),
            }
        )

    return pd.DataFrame(df_data).set_index(["field", "ngram_metric"])


def get_metric_values(field_res: dict[str, Any]) -> dict[str, Any]:
    match field_res:
        case {"f1": dict(f1_score)}:
            return {
                "ngram_metric": "f1",
                "ngram_metric_mean": f1_score["mean_score"],
                "ngram_metric_se": f1_score["standard_error"],
            }
        case {"rougeL": dict(rouge_score)}:
            return {
                "ngram_metric": "rougeL",
                "ngram_metric_mean": rouge_score["mean_score"],
                "ngram_metric_se": rouge_score["standard_error"],
            }
        case {"match": dict(match_score)}:
            return {
                "ngram_metric": "exact_match",
                "ngram_metric_mean": match_score["mean_score"],
                "ngram_metric_se": match_score["standard_error"],
            }
        case {}:
            return {"ngram_metric": "<unknown>", "ngram_metric_mean": 0, "ngram_metric_se": 0}
        case _:
            raise ValueError(f"Unknown field type: {field_res}")
