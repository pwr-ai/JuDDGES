import streamlit as st

from juddges.data.pl_court_api import PolishCourtAPI
from juddges.prompts.information_extraction import (
    EXAMPLE_SCHEMA,
    prepare_information_extraction_chain,
    prepare_schema_chain,
)
from juddges.settings import prepare_langchain_cache, prepare_mlflow

prepare_langchain_cache()
prepare_mlflow()

TITLE = "⚖️ JuDDGES Information Extraction from Court Decisions ⚖️"

st.set_page_config(page_title=TITLE, page_icon="⚖️", layout="wide")

st.title(TITLE)

st.info(
    "You can provide an URL to court decision or plain text of it, describe in written form schema of the information that will be extracted, choose model and language and start extraction."
)

st.header("Data source")
source_option = st.selectbox("Choose the source of the judgement text:", ["API", "Plain text"])

if source_option == "API":
    api = PolishCourtAPI()
    judgement_url = st.text_input(
        "Enter the judgement URL:",
        "https://orzeczenia.wroclaw.sa.gov.pl/details/$N/155000000001006_II_AKa_000334_2019_Uz_2020-02-06_001",
    )
    judgement_id = judgement_url.strip().split("/")[-1]
    judgement_text = api.get_content(id=judgement_id)
else:
    judgement_text = st.text_area("Enter the judgement text here:", height=500)

st.header("Schema extraction/definition")
schema_query = st.text_input(
    "Ask for schema in natural language:",
    "Extract the date, verdict, and court from  the judgement.",
)
llm_schema = st.selectbox(
    "Select the LLM model (schema)",
    ["gpt-3.5-turbo-1106", "gpt-4-0125-preview", "gpt-4-1106-preview"],
)

if st.button("Generate schema to extract information"):
    chain = prepare_schema_chain(model_name=llm_schema)
    schema = chain.invoke({"SCHEMA_TEXT": schema_query})
    if not schema:
        st.error("Could not extract schema from the given query. Try with a different one.")
    else:
        st.session_state.schema = schema

schema_text = st.text_area(
    "Enter the schema text here:", st.session_state.get("schema") or EXAMPLE_SCHEMA, height=500
)

st.header("Information extraction")
llm_extraction = st.selectbox(
    "Select the LLM model", ["gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-3.5-turbo-1106"]
)
language = st.selectbox("Enter the language of the judgement text:", ["Polish", "English"])


if st.button("Extract information"):
    with st.spinner("Extracting information from the judgement text..."):
        chain = prepare_information_extraction_chain(model_name=llm_extraction)
        retrieved_informations = chain.invoke(
            {"LANGUAGE": language, "TEXT": judgement_text, "SCHEMA": schema_text}
        )
        col_left, col_right = st.columns(2)

        col_left.write(judgement_text)
        col_right.write(retrieved_informations)
