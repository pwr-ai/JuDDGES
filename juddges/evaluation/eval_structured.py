from abc import ABC, abstractmethod
from statistics import mean

import multiprocess
from torchmetrics.functional.text import chrf_score


class StructuredEvaluatorBase(ABC):
    def __init__(self, name: str, num_proc: int) -> None:
        self.name = f"field_{name}"
        self.num_proc = num_proc

    @abstractmethod
    def evaluate(
        self,
        preds: dict[str, list[str]],
        golds: dict[str, list[str]],
    ) -> dict[str, dict[str, float]]:
        """Evaluates information extraction by computing metrics per each field.
        The inputs should be formatted according to output of parse_results function.
        """


class StructuredMetricEvaluator(StructuredEvaluatorBase, ABC):
    def evaluate(
        self,
        preds: dict[str, list[str]],
        golds: dict[str, list[str]],
    ) -> dict[str, dict[str, float]]:
        if self.num_proc == -1:
            num_proc = len(golds)
        else:
            num_proc = min(self.num_proc, len(golds))
        if num_proc > 1:

            def _compute_task(key: str) -> float:
                return self._compute(preds=preds[key], gold=golds[key])

            with multiprocess.Pool(num_proc) as pool:
                results = pool.map(
                    _compute_task,
                    golds.keys(),
                )
                return dict(zip(golds.keys(), results))
        else:
            return {key: self._compute(preds=preds[key], gold=golds[key]) for key in golds.keys()}

    @abstractmethod
    def _compute(self, preds: list[str], gold: list[str]) -> float:
        pass


class StructuredChrfEvaluator(StructuredMetricEvaluator):
    def __init__(self, num_proc: int = 1) -> None:
        super().__init__(name="chrf", num_proc=num_proc)

    def _compute(self, preds: list[str], gold: list[str]) -> float:
        scores = []
        for p, g in zip(preds, gold):
            scores.append(chrf_score(preds=[p], target=[g], n_word_order=1, n_char_order=1).item())
        return mean(scores)  # type: ignore
