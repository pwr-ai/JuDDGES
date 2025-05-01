from pathlib import Path
from textwrap import dedent

import yaml
from langchain.output_parsers.json import parse_json_markdown
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableSequence,
)
from langchain_openai import ChatOpenAI

from juddges.settings import PROMPTS_PATH


def load_prompt_from_yaml(path: Path) -> str:
    return dedent(yaml.safe_load(path.read_text())["content"]).strip()


SCHEMA_PROMPT_TEMPLATE = load_prompt_from_yaml(PROMPTS_PATH / "schema.yaml")
EXTRACTION_PROMPT_TEMPLATE = load_prompt_from_yaml(PROMPTS_PATH / "extraction.yaml")


def prepare_information_extraction_chain_from_user_prompt() -> RunnableSequence:
    schema_chain = prepare_schema_chain()
    inputs = {
        "SCHEMA": schema_chain,
        "TEXT": RunnablePassthrough(),
        "LANGUAGE": RunnablePassthrough(),
    }
    return inputs | RunnableLambda(route)


def prepare_information_extraction_chain(
    model_name: str = "gpt-4-0125-preview",
    log_to_mlflow: bool = False,
) -> RunnableSequence:
    model = ChatOpenAI(model=model_name, temperature=0)
    human_message_template = HumanMessagePromptTemplate.from_template(EXTRACTION_PROMPT_TEMPLATE)
    _prompt = ChatPromptTemplate(
        messages=[human_message_template],
        input_variables=["TEXT", "LANGUAGE", "SCHEMA"],
    )

    if log_to_mlflow:
        import mlflow

        mlflow.log_dict(_prompt.save_to_json(), "prompt.json")

    return _prompt | model | (lambda x: parse_json_markdown(x.content))


def prepare_schema_chain(model_name: str = "gpt-3.5-turbo") -> RunnableSequence:
    model = ChatOpenAI(model=model_name, temperature=0)
    human_message_template = HumanMessagePromptTemplate.from_template(SCHEMA_PROMPT_TEMPLATE)
    _prompt = ChatPromptTemplate(
        messages=[human_message_template],
        input_variables=["TEXT", "LANGUAGE", "SCHEMA"],
    )

    return _prompt | model | parse_schema


def parse_schema(ai_message: AIMessage) -> str:
    response_schema = parse_json_markdown(ai_message.content)
    return "\n".join(f"{key}: {val}" for key, val in response_schema.items())


def route(response_schema: str) -> dict[str, str]:
    if response_schema["SCHEMA"]:
        return prepare_information_extraction_chain()

    raise ValueError(
        "Cannot determine schema for the given input prompt. Please try different query."
    )
