from abc import ABC, abstractmethod

from torchmetrics.functional.text import chrf_score


class StructuredEvaluatorBase(ABC):
    def __init__(self, name: str) -> None:
        self.name = f"field_{name}"

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
        return {key: self._compute(preds=preds[key], gold=golds[key]) for key in golds.keys()}

    @abstractmethod
    def _compute(self, preds: list[str], gold: list[str]) -> float:
        pass


class StructuredCHRFEvaluator(StructuredMetricEvaluator):
    def __init__(self) -> None:
        super().__init__(name="chrf")

    def _compute(self, preds: list[str], gold: list[str]) -> float:
        return chrf_score(preds=preds, target=gold, n_word_order=0).item()  # type: ignore
