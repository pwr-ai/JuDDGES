import os
from enum import Enum
from typing import List, Optional

from langchain_core.pydantic_v1 import BaseModel, Field
from langdetect import detect as lang_detect
from pymongo import MongoClient
from pymongo.collection import Collection

from juddges.exception import (
    GeneratedQAPairsEmptyError,
    GeneratedQAPairsLenghtMismatchError,
    GeneratedQAPairsNotUniqueError,
    LanguageMismatchError,
)


class LangCode(Enum):
    """Language code (ISO 639-1)"""

    POLISH: str = "pl"
    ENGLISH: str = "en"


class SyntheticQAPairs(BaseModel):
    questions: List[str] = Field(description="List of generated questions")
    answers: List[str] = Field(description="List of generated answers")

    def __len__(self) -> int:
        return len(self.questions)

    def test_empty(self) -> None:
        if not any(self.questions):
            raise GeneratedQAPairsEmptyError("At least one question should be generated")
        else:
            if not any(self.answers):
                raise GeneratedQAPairsEmptyError("At least one answer should be generated")

    def test_equal_length(self) -> None:
        assertion_msg = "Number of questions and answers should be equal"
        if len(self.questions) != len(self.answers):
            raise GeneratedQAPairsLenghtMismatchError(assertion_msg)

    def test_unique_questions(self) -> None:
        assertion_msg = "Questions should be unique"
        if len(set(self.questions)) != len(self.questions):
            raise GeneratedQAPairsNotUniqueError(assertion_msg)

    def test_language(self, lang: LangCode) -> None:
        msg = "{smth} should match the context language" + f" ({lang.name}/{lang.value})"
        if lang_detect("\n".join(self.questions)) != lang.value:
            raise LanguageMismatchError(msg.format(smth="Questions"))
        if lang_detect("\n".join(self.answers)) != lang.value:
            raise LanguageMismatchError(msg.format(smth="Answers"))

    def test(self, language: LangCode) -> None:
        self.test_empty()
        self.test_equal_length()
        self.test_unique_questions()
        self.test_language(language)


class QAGenerationJudgementMetadata(BaseModel):
    type_: Optional[str] = Field(default=None, description="Type of the judgement")
    excerpt: Optional[str] = Field(default=None, description="Excerpt of the judgement")
    chairman: Optional[str] = Field(default=None, description="Chairman title and full name")
    decision: Optional[str] = Field(default=None, description="Decision of the judgement")
    judges: List[str] = Field(default=None, description="List of judges")
    legal_bases: List[str] = Field(default=None, description="List of legal bases")
    publisher: Optional[str] = Field(default=None, description="Publisher of the judgement")
    recorder: Optional[str] = Field(default=None, description="Recorder of the judgement")
    references: List[str] = Field(default=None, description="List of reference materials")
    reviser: Optional[str] = Field(default=None, description="Reviser of the judgement")
    theme_phrases: List[str] = Field(default=None, description="List of theme phrases")


if os.environ.get("MONGO_URI", None) is None:
    raise Exception("Missing `MONGO_URI` environment variable.")


if os.environ.get("MONGO_DB_NAME", None) is None:
    raise Exception("Missing `MONGO_DB_NAME` environment variable.")


def get_mongo_collection(collection_name: str = "judgements") -> Collection:
    client = MongoClient(os.environ["MONGO_URI"])
    db = client[os.environ["MONGO_DB_NAME"]]
    return db[collection_name]
