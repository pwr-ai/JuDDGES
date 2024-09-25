from unittest import TestCase

from juddges.evaluation.eval_structured import StructuredChrfEvaluator


class TestEvalStructured(TestCase):
    def test_eval_structured_chrf_score(self):
        preds = {
            "field_1": ["abc", "", "ddd", ""],
            "field_2": [""],
        }
        golds = {
            "field_1": ["abc", "abc", "abc", "abc"],
            "field_2": ["abc"],
        }

        target_output = {
            "field_1": 0.25,
            "field_2": 0.0,
        }
        evaluator = StructuredChrfEvaluator()
        eval_output = evaluator.evaluate(preds, golds)

        self.assertDictEqual(eval_output, target_output)
