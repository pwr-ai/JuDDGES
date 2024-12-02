from typing import Any

import streamlit as st
from pymongo.collection import Collection
from sentence_transformers import SentenceTransformer

from juddges.data.database import get_mongo_collection
from juddges.data_models import Judgment
from juddges.retrieval.mongo_hybrid_search import run_hybrid_search
from juddges.retrieval.mongo_term_based_search import search_judgements
from juddges.settings import TEXT_EMBEDDING_MODEL

TITLE = "Search for Judgements"

st.title(TITLE)

judgement_country = st.sidebar.selectbox("Select judgement country", ["pl", "uk"])
judgement_collection_name = f"{judgement_country}-court"
st.sidebar.info(f"Selected country: {judgement_collection_name}")

use_hybrid_search = st.sidebar.toggle("Use hybrid search", value=False)

if use_hybrid_search:
    st.header(
        "Search is based on hybrid search using text and vector search with the same priority for both."
    )
else:
    st.header("Search is based on term-based search with highlighting.")


@st.cache_resource
def get_judgements_collection(collection_name: str = "pl-court") -> Collection:
    return get_mongo_collection(collection_name=collection_name)


@st.cache_resource
def get_embedding_model() -> Any:
    return SentenceTransformer(TEXT_EMBEDDING_MODEL)


judgements_collection = get_judgements_collection(judgement_collection_name)

if use_hybrid_search:
    model = get_embedding_model()

with st.form(key="search_form"):
    query = st.text_area("What you are looking for in the judgements?")
    max_judgements = st.slider(
        "Max judgements to show", min_value=1, max_value=20, value=5
    )
    submit_button = st.form_submit_button(label="Search")

if submit_button:
    with st.spinner("Searching..."):
        if use_hybrid_search:
            items = run_hybrid_search(
                collection=judgements_collection,
                collection_name=judgement_collection_name,
                embedding=model.encode(query).tolist(),
                query=query,
                limit=max_judgements,
            )

            st.header("Judgements - Results")
            for item in items:
                st.header(item[Judgment.SIGNATURE.value])
                st.info(f"Court: {item[Judgment.COURT_NAME.value]}")
                st.info(f"Department: {item[Judgment.DEPARTMENT_NAME.value]}")
                st.info(f"Date: {item[Judgment.DATE.value]}")
                st.info(f"Score: {item['score']}")
                st.subheader(item[Judgment.EXCERPT.value])
                st.text_area(
                    label="Judgement text", value=item[Judgment.TEXT.value], height=200
                )
        else:
            items = search_judgements(query=query, max_docs=max_judgements)

            st.header("Judgements - Results")
            for item in items:
                st.header(item[Judgment.SIGNATURE.value])
                st.info(f"Court: {item[Judgment.COURT_NAME.value]}")
                st.info(f"Date: {item[Judgment.DATE.value]}")
                st.info(f"Score: {item['score']}")

                # Process and display highlights
                if "highlights" in item:
                    st.subheader("Highlighted Excerpts")
                    for highlight in item["highlights"]:
                        text = ""
                        for segment in highlight["texts"]:
                            if segment["type"] == "hit":
                                # Use yellow background with black text for better visibility in dark mode
                                text += f"<span style='background-color: #FFD700; color: black; padding: 0 2px;'>{segment['value']}</span>"
                            else:
                                text += segment["value"]
                        st.markdown(
                            f"""
---
{text}
""",
                            unsafe_allow_html=True,
                        )

                # Add toggle for full text
                with st.expander("Show Full Judgment Text"):
                    st.markdown(item[Judgment.TEXT.value])
