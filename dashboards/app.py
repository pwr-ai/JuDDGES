import streamlit as st

from juddges.data.pl_court_api import PolishCourtAPI
from juddges.prompts.information_extraction import SCHEMA, prepare_information_extraction_chain
from juddges.settings import prepare_langchain_cache, prepare_mlflow

prepare_langchain_cache()
prepare_mlflow()

st.title("JuDDGES Dashboard")

api = PolishCourtAPI()

judgement_url = st.text_input(
    "Enter the judgement URL:",
    "https://orzeczenia.wroclaw.sa.gov.pl/details/$N/155000000001006_II_AKa_000334_2019_Uz_2020-02-06_001",
)

judgement_id = judgement_url.strip().split("/")[-1]
judgement_text = api.get_content(id=judgement_id)

schema_text = st.text_area("Enter the schema text here:", SCHEMA)
LLM_model = st.selectbox(
    "Select the LLM model", ["gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-3.5-turbo-1106"]
)
language = st.selectbox("Enter the language of the judgement text:", ["Polish", "English"])


if st.button("Extract information"):
    with st.spinner("Extracting information from the judgement text..."):
        chain = prepare_information_extraction_chain(model_name=LLM_model)
        retrieved_informations = chain.invoke(
            {"LANGUAGE": language, "TEXT": judgement_text, "SCHEMA": schema_text}
        )
        st.write(retrieved_informations)
