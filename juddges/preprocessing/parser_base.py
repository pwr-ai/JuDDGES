from abc import ABC, abstractmethod
from typing import Any


class DocParserBase(ABC):
    """Base class for parser retrieving data from a document."""

    def __call__(self, document: str) -> dict[str, Any]:
        return self.parse(document)

    @property
    @abstractmethod
    def schema(self) -> list[str]:
        pass

    @abstractmethod
    def parse(self, document: str) -> dict[str, Any]:
        pass
