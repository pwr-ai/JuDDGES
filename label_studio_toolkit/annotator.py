from abc import ABC, abstractmethod

from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class Annotator(ABC):
    def __init__(self, schema: type[BaseModel]) -> None:
        self.schema = schema

    @abstractmethod
    def annotate(self, input_data: str) -> BaseModel:
        pass


class LangChainAnnotator(Annotator):
    def __init__(
        self, llm: BaseChatModel, prompt: str, schema: type[BaseModel], method: str
    ) -> None:
        super().__init__(schema)
        assert "{text}" in prompt, "Prompt must contain {text} placeholder"
        self.llm = llm
        self.prompt_template = PromptTemplate.from_template(prompt)
        structured_llm = self.llm.with_structured_output(self.schema, method=method)
        self.chain = self.prompt_template | structured_llm

    def annotate(self, input_data: str) -> BaseModel:
        result = self.chain.invoke({"text": input_data})
        assert isinstance(result, self.schema)
        return result

    async def async_annotate(self, input_data: str, language: str) -> BaseModel:
        result = await self.chain.ainvoke({"text": input_data, "language": language})
        assert isinstance(result, self.schema)
        return result


class LangChainOpenAIAnnotator(LangChainAnnotator):
    def __init__(self, model: str, prompt: str, schema: type[BaseModel]) -> None:
        llm = ChatOpenAI(model=model, temperature=0, max_retries=5)
        super().__init__(llm, prompt, schema, "json_schema")
