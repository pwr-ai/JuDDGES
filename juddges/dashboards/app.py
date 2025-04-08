import streamlit as st

st.set_page_config(page_title="JuDDGES", page_icon="⚖️", layout="wide")

project_info_page = st.Page("project_info.py", title="Project Info", icon="💡")
search_judgments_page = st.Page("search_judgments.py", title="Search Judgments", icon="🔍")
extract_information_from_judgments_page = st.Page(
    "extract_information_from_judgments.py", title="Extract Information", icon="📄"
)
analyse_extracted_information_page = st.Page(
    "analyse_extracted_information.py", title="Analyse Information", icon="📊"
)
linie_orzecznicze_page = st.Page("linie_orzecznicze.py", title="Linie Orzecznicze", icon="📊")

sections = {
    "Project Info": [project_info_page],
    "Search Judgments": [search_judgments_page],
    "Extract Information": [extract_information_from_judgments_page],
    "Analyse Information": [linie_orzecznicze_page, analyse_extracted_information_page],
}

pg = st.navigation(sections)

pg.run()
