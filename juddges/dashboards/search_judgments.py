import streamlit as st

from juddges.dashboards.components.search_results import display_search_results
from juddges.dashboards.utils.search_utils import (
    get_embedding_model,
    get_judgments_collection,
)
from juddges.retrieval.mongo_hybrid_search import run_hybrid_search
from juddges.retrieval.mongo_term_based_search import search_judgments
from juddges.settings import ROOT_PATH

# Load CSS
with open(ROOT_PATH / "juddges/dashboards/styles/search_judgements.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

TITLE = "🔍 Search for Judgements"

st.title(TITLE)

# Sidebar configuration
judgement_country = st.sidebar.selectbox("Select judgement country", ["pl", "uk"])
judgement_collection_name = f"{judgement_country}-court"
st.sidebar.info(f"Selected country: {judgement_collection_name}")

use_hybrid_search = st.sidebar.toggle("Use hybrid search", value=False)

# Header based on search type
if use_hybrid_search:
    st.header(
        "Search is based on hybrid search using text and vector search with the same priority for both"
    )
else:
    st.header("Search is based on term-based search with highlighting")

# Initialize collections and models
judgements_collection = get_judgments_collection(judgement_collection_name)
if use_hybrid_search:
    model = get_embedding_model()

# Search form
with st.form(key="search_form"):
    query = st.text_area(
        "What you are looking for in the judgements?",
        value="kredyty hipoteczne franki szwajcarskie",
    )
    max_judgements = st.slider(
        "Max judgements to show", min_value=1, max_value=20, value=5
    )
    submit_button = st.form_submit_button(label="Search")

# Process search
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
        else:
            items = search_judgments(query=query, max_docs=max_judgements)

        display_search_results(items)
