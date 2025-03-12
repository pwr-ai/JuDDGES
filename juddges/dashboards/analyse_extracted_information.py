import io

import pandas as pd
import streamlit as st

from juddges.prompts.information_extraction import EXAMPLE_SCHEMA
from juddges.settings import SAMPLE_DATA_PATH

TITLE = "Analyse judgments"

st.title(TITLE)


@st.cache_resource
def load_data():
    return pd.read_csv(
        SAMPLE_DATA_PATH / "judgements-100-sample-with-retrieved-informations.csv"
    )


df = load_data()
extracted_keys = [
    line.split(":")[0] for line in EXAMPLE_SCHEMA.split("\n") if len(line) > 3
] + [
    "signature",
    "excerpt",
    "text",
    "judges",
    "references",
]

st.info(
    "We sampled 100 random judgments from the dataset and extracted information from them. Below is the extracted information and the schema (questions) used to extract it."
)

st.text_area(
    "Example schema for extracted informations: ",
    value=EXAMPLE_SCHEMA,
    height=300,
    disabled=True,
)

st.header("Extracted Information - tabular format")
st.write(df[extracted_keys])


output = io.BytesIO()
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="Sheet1", index=False)
output.seek(0)
st.download_button(
    label="Download data as Excel",
    data=output,
    file_name="judgments.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.header("Analyse Extracted Information")

st.subheader("How many judgments we analyzed?")

st.write(f"Number of judgments: {len(df)}")

st.subheader("What courts judgment do we analyse")

st.write(df.groupby("court")["_id"].count())

st.subheader("How many judgments are drug offences?")

drug_offences = df["drug_offence"].sum()

st.info(f"Number of drug offences: {drug_offences}")

st.subheader("How many judgments are child offences?")

child_offences = df["child_offence"].sum()

st.info(f"Number of child offences: {child_offences}")

st.subheader("Show examples of judgments that are child offences")

drug_offences_df = df[df["child_offence"]]

st.write("We can check the sentences of them")

for row_id, row in drug_offences_df.iterrows():
    st.subheader(row["signature"])
    st.info(row["verdict_summary"])
    if st.toggle(key=row, label="Show judgment's text"):
        st.markdown(row["text"])
        st.markdown("---")
