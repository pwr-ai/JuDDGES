from abc import ABC, abstractmethod

from torchmetrics.functional.text import chrf_score


class FullTextEvaluatorBase(ABC):
    def __init__(self, name: str) -> None:
        self.name = f"full_text_{name}"

    @abstractmethod
    def evaluate(self, preds: list[str], golds: list[str]) -> float:
        pass


class FullTextCHRFScorer(FullTextEvaluatorBase):
    def __init__(self) -> None:
        super().__init__(name="chrf")

    def evaluate(self, preds: list[str], golds: list[str]) -> float:
        return chrf_score(preds=preds, target=golds, n_word_order=0).item()  # type: ignore
