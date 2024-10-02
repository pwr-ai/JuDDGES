from tqdm import tqdm

from juddges.evaluation.eval_full_text import FullTextCHRFScorer
from juddges.evaluation.eval_structured import StructuredChrfEvaluator
from juddges.evaluation.parse import parse_results


class InfoExtractionEvaluator:
    def __init__(self, num_proc: int, verbose: bool = True):
        self.verbose = verbose

        self.structured_evaluators = [
            StructuredChrfEvaluator(num_proc=num_proc),
        ]
        self.full_text_evaluators = [
            FullTextCHRFScorer(),
        ]

    def evaluate(self, results: list[dict[str, str]]) -> dict[str, dict[str, float]]:
        preds, golds = parse_results(results)

        metrics = {}
        with tqdm(self.structured_evaluators, desc="Structured", disable=not self.verbose) as pbar:
            for eval in pbar:
                pbar.set_postfix({"eval": eval.name})
                metrics[eval.name] = eval.evaluate(preds=preds, golds=golds)

        text_preds = [res["answer"] for res in results]
        text_golds = [res["gold"] for res in results]
        with tqdm(self.full_text_evaluators, desc="Full text", disable=not self.verbose) as pbar:
            for eval in pbar:
                pbar.set_postfix({"eval": eval.name})
                metrics[eval.name] = eval.evaluate(preds=text_preds, golds=text_golds)

        return metrics
