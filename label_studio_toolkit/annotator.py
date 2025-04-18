from abc import ABC, abstractmethod
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Annotator(ABC):
    @abstractmethod
    def annotate(self, input_data: dict[str, Any]) -> T:
        pass


class LLMAnnotator(Annotator):
    def __init__(self, llm: BaseChatModel, schema: T) -> None:
        self.llm = llm
        self.schema = schema
        self.llm = self.llm.with_structured_output(self.schema, method="function_calling")

    def annotate(self, input_data: dict[str, Any]) -> T:
        return self.llm.invoke(input_data)
