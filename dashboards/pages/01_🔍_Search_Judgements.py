import streamlit as st

from juddges.data.models import get_mongo_collection

TITLE = "Search for Judgements"

st.set_page_config(page_title=TITLE, page_icon="⚖️", layout="wide")

st.title(TITLE)


@st.cache_resource
def get_judgements_collection():
    return get_mongo_collection("judgements")


judgements_collection = get_judgements_collection()


def search_data(query: str, max_judgements: int = 5):
    items = list(judgements_collection.find({"$text": {"$search": query}}).limit(max_judgements))
    return items


with st.form(key="search_form"):
    text = st.text_area("What you are looking for in the judgements?")
    max_judgements = st.slider("Max judgements to show", min_value=1, max_value=20, value=5)
    submit_button = st.form_submit_button(label="Search")

if submit_button:
    with st.spinner("Searching..."):
        items = search_data(text, max_judgements)

        st.header("Judgements - Results")
        for item in items:
            st.header(item["signature"])
            st.subheader(item["publicationDate"])
            st.write(item["text"])
