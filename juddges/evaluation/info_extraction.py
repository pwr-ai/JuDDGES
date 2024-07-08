import tqdm

from juddges.evaluation.eval_full_text import FullTextCHRFScorer
from juddges.evaluation.eval_structured import StructuredCHRFEvaluator
from juddges.evaluation.parse import parse_results


class InfoExtractionEvaluator:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

        self.structured_evaluators = [
            StructuredCHRFEvaluator(),
        ]
        self.full_text_evaluators = [
            FullTextCHRFScorer(),
        ]

    def evaluate(self, results: list[dict[str, str]]) -> dict[str, dict[str, float]]:
        results = {}
        preds, golds = parse_results(results)

        for eval in (
            pbar := tqdm(self.structured_evaluators, desc="Structured", disable=not self.verbose)
        ):
            pbar.set_postfix({"eval": eval.name})
            results[eval.name] = eval.evaluate(preds=preds, golds=golds)

        text_preds = [res["answer"] for res in results]
        text_golds = [res["gold"] for res in results]
        for eval in (
            pbar := tqdm(self.full_text_evaluators, desc="Full text", disable=not self.verbose)
        ):
            pbar.set_postfix({"eval": eval.name})
            results[eval.name] = eval.evaluate(preds=text_preds, golds=text_golds)

        return results
