import pandas as pd
import streamlit as st

from juddges.prompts.information_extraction import EXAMPLE_SCHEMA
from juddges.settings import SAMPLE_DATA_PATH

TITLE = "Analyse Judgements"

st.set_page_config(page_title=TITLE, page_icon="⚖️", layout="wide")

st.title(TITLE)


@st.cache_resource
def load_data():
    return pd.read_csv(SAMPLE_DATA_PATH / "judgements-100-sample-with-retrieved-informations.csv")


df = load_data()

st.info(
    "We sampled 100 random judgements from the dataset and extracted information from them. Below is the extracted information and the schema (questions) used to extract it."
)

st.header("Schema:")
st.write(EXAMPLE_SCHEMA)

st.header("Extracted Information - tabular format")
st.write(df)

st.header("Analyse Extracted Information")

st.subheader("How many judgements we analyzed?")

st.write(f"Number of judgements: {len(df)}")

st.subheader("What courts judgement do we analyse")

st.write(df.groupby("court")["_id"].count())

st.subheader("How many judgements are drug offences?")

drug_offences = df["drug_offence"].sum()

st.info(f"Number of drug offences: {drug_offences}")

st.subheader("How many judgements are child offences?")

child_offences = df["child_offence"].sum()

st.info(f"Number of child offences: {child_offences}")

st.subheader("Show examples of judgements that are child offences")

drug_offences_df = df[df["child_offence"]]

st.write("We can check the sentences of them")

for _, row in drug_offences_df.iterrows():
    st.subheader(row["signature"])
    st.markdown(row["text"])
    st.markdown("---")  # Add a horizontal line
