from langchain.output_parsers.json import parse_json_markdown
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence
from langchain_openai import ChatOpenAI

SCHEMA_PROMPT_TEMPLATE = """
Act as a assistant that prepares schema for information extraction

Based on the user input prepare schema containing variables with their short description and type. 
Be precise about variable names, format names using snake_case. 
If user asks irrelevant question always return empty JSON.
As example:
User: I want extract age, gender, and plea from the judgement
Agent: 
    age: integer
    gender: male or female
    plea: string

====
{SCHEMA_TEXT}
====

Format response as JSON:
"""

EXTRACTION_PROMPT_TEMPLATE = """Act as a legal document tool that extracts information and answer questions based on judgements. 

Instruction for extracting information from judgements:
- Judgements are in {LANGUAGE} language, please extract information in {LANGUAGE}.
- Do not provide information that are not explicitly mentioned in judgements. If you can't extract information from the text field, leave the field with empty string "".

Follow the following YAML structure to extract information and answer questions based on judgements: 
{SCHEMA}

====
{TEXT}
====

Format response as JSON:
"""

EXAMPLE_SCHEMA = """
defendant_gender: string
defendant_age: integer
defendant_relationship_status: string
defendant_has_children: boolean
defendant_homeless: boolean
appellant: string
appeal_against: string
defendant_plead_or_convicted: string "guilty plea" or "convicted at trial"
jury_unanimous: string "unanimous" or "other"
first_trial: boolean
drug_offence: boolean
original_sentence: string
tried_court_type: string "Crown" or "magistrates'"
single_transaction_multiple_offence: boolean
multiple_transactions_period: string "within 1 year" or "more than 1 year"
concurrent_or_consecutive_sentence: string "concurrently" or "consecutively"
sentence_on_top_existing: boolean
sentence_adding_up: boolean
sentence_leniency: string "unduly lenient" or "too excessive"
guilty_plea_reduction_reason: string
sentence_discount_mention: boolean
totality_issues_similar_offences: boolean
sentence_proportionality_mention: boolean
sentence_type_issues_totality: boolean
sentence_adjustment_mention: boolean
offender_culpability_determination: boolean
harm_caused_determination: boolean
offence_seriousness: boolean
aggravating_factors: string list of factors excluding previous convictions
previous_convictions_similarity: string "similar" or "dissimilar"
mitigating_factors: string list of factors
immediate_sentence_concurrency: boolean
totality_conflicting_guidelines: boolean
totality_principle_misapplication: boolean
"""


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
