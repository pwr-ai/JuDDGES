from tqdm import tqdm

from juddges.evaluation.eval_full_text import FullTextCHRFScorer
from juddges.evaluation.eval_structured import StructuredChrfEvaluator
from juddges.evaluation.parse import T_format, parse_results


class InfoExtractionEvaluator:
    def __init__(self, num_proc: int, format: T_format, verbose: bool = True):
        self.verbose = verbose
        self.format = format
        self.structured_evaluators = [
            StructuredChrfEvaluator(num_proc=num_proc),
        ]
        self.full_text_evaluators = [
            FullTextCHRFScorer(),
        ]

    def evaluate(self, results: list[dict[str, str]]) -> dict[str, dict[str, float]]:
        parsed = parse_results(results, self.format)

        metrics = {
            "stats": {
                "total_docs": len(results),
                "num_preds_parse_errors": parsed.num_preds_parse_errors,
            }
        }
        with tqdm(self.structured_evaluators, desc="Structured", disable=not self.verbose) as pbar:
            for eval in pbar:
                pbar.set_postfix({"eval": eval.name})
                assert eval.name != "stats"
                metrics[eval.name] = eval.evaluate(
                    preds=parsed.preds,
                    golds=parsed.golds,
                )

                correct_structure_preds = {}
                correct_structure_golds = {}
                for k in parsed.preds.keys():
                    correct_structure_preds[k] = [
                        p_i
                        for i, p_i in enumerate(parsed.preds[k])
                        if not parsed.failed_preds_parse_mask[i]
                    ]
                    correct_structure_golds[k] = [
                        g_i
                        for i, g_i in enumerate(parsed.golds[k])
                        if not parsed.failed_preds_parse_mask[i]
                    ]
                metrics[f"{eval.name}__correctly_parsed"] = eval.evaluate(
                    preds=correct_structure_preds,
                    golds=correct_structure_golds,
                )

        text_preds = [res["answer"] for res in results]
        text_golds = [res["gold"] for res in results]
        with tqdm(self.full_text_evaluators, desc="Full text", disable=not self.verbose) as pbar:
            for eval in pbar:
                pbar.set_postfix({"eval": eval.name})
                assert eval.name != "stats"
                metrics[eval.name] = eval.evaluate(
                    preds=text_preds,
                    golds=text_golds,
                )

        return metrics
