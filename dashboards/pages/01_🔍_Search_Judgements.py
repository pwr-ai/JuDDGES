from typing import Any

import streamlit as st
from pymongo.collection import Collection
from sentence_transformers import SentenceTransformer

from juddges.data.database import get_mongo_collection
from juddges.retrieval.mongo_hybrid_search import run_hybrid_search
from juddges.settings import TEXT_EMBEDDING_MODEL

TITLE = "Search for Judgements"

st.set_page_config(page_title=TITLE, page_icon="⚖️", layout="wide")

st.title(TITLE)
st.header(
    "Search is based on hybrid search using text and vector search with the same priority for both."
)

judgement_country = st.sidebar.selectbox("Select judgement country", ["pl", "uk"])
judgement_collection_name = f"{judgement_country}-court"
st.sidebar.info(f"Selected country: {judgement_collection_name}")


@st.cache_resource
def get_judgements_collection(collection_name: str = "pl-court") -> Collection:
    return get_mongo_collection(collection_name=collection_name)


@st.cache_resource
def get_embedding_model() -> Any:
    return SentenceTransformer(TEXT_EMBEDDING_MODEL)


judgements_collection = get_judgements_collection(judgement_collection_name)

model = get_embedding_model()

with st.form(key="search_form"):
    query = st.text_area("What you are looking for in the judgements?")
    max_judgements = st.slider(
        "Max judgements to show", min_value=1, max_value=20, value=5
    )
    submit_button = st.form_submit_button(label="Search")

if submit_button:
    with st.spinner("Searching..."):
        items = run_hybrid_search(
            collection=judgements_collection,
            collection_name=judgement_collection_name,
            embedding=model.encode(query).tolist(),
            query=query,
            limit=max_judgements,
        )

        st.header("Judgements - Results")
        for item in items:
            st.header(item["signature"])
            st.info(f"Department: {item['department_name']}")
            st.info(f"Score: {item['score']}")
            st.subheader(item["excerpt"])
            st.text_area(label="Judgement text", value=item["text"], height=200)
