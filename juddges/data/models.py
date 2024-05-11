import os
from enum import Enum
from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field
from langdetect import detect as lang_detect
from pymongo import MongoClient
from pymongo.collection import Collection

from juddges.exception import LanguageMismatchError


class LangCode(Enum):
    """Language code (ISO 639-1)"""

    POLISH: str = "pl"
    ENGLISH: str = "en"


class SyntheticLegisQAPairs(BaseModel):
    questions: List[str] = Field(description="List of generated questions")
    answers: List[str] = Field(description="List of generated answers")

    def test_empty(self) -> None:
        assert len(self.questions) > 0, "At least one question should be generated"

    def test_equal_length(self) -> None:
        assertion_msg = "Number of questions and answers should be equal"
        assert len(self.questions) == len(self.answers), assertion_msg

    def test_unique_questions(self) -> None:
        assertion_msg = "Questions should be unique"
        assert len(set(self.questions)) == len(self.questions), assertion_msg

    def test_language(self, lang: LangCode) -> None:
        msg = "{smth} should match context language" + f" ({lang.name}/{lang.value})"
        assert lang_detect("\n".join(self.questions)) == lang.value, LanguageMismatchError(
            msg.format(smth="Questions")
        )
        assert lang_detect("\n".join(self.answers)) == lang.value, LanguageMismatchError(
            msg.format(smth="Answers")
        )

    def test(self, language: LangCode) -> None:
        self.test_empty()
        self.test_equal_length()
        self.test_unique_questions()
        self.test_language(language)


if os.environ.get("MONGO_URI", None) is None:
    raise Exception("Missing `MONGO_URI` environment variable.")


if os.environ.get("MONGO_DB_NAME", None) is None:
    raise Exception("Missing `MONGO_DB_NAME` environment variable.")


def get_mongo_collection(collection_name: str = "judgements") -> Collection:
    client = MongoClient(os.environ["MONGO_URI"])
    db = client[os.environ["MONGO_DB_NAME"]]
    return db[collection_name]
