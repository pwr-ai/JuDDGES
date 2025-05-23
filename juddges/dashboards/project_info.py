import streamlit as st

from juddges.settings import ROOT_PATH

TITLE = "⚖️ JuDDGES Information Extraction from Court Decisions ⚖️"


st.title(TITLE)

st.warning("JuDDGES stands for Judicial Decision Data Gathering, Encoding, and Sharing")

st.info(
    """The JuDDGES project aims to revolutionize the accessibility and analysis of judicial decisions across varied legal systems using advanced Natural Language Processing and Human-In-The-Loop technologies. It focuses on criminal court records from jurisdictions with diverse legal constitutions, including Poland and England & Wales. By overcoming barriers related to resources, language, data, and format inhomogeneity, the project facilitates the development and testing of theories on judicial decision-making and informs judicial policy and practice. Open software and tools produced by the project will enable extensive, flexible meta-annotation of legal texts, benefiting researchers and public legal institutions alike. This initiative not only advances empirical legal research by adopting Open Science principles but also creates the most comprehensive legal research repository in Europe, fostering cross-disciplinary and cross-jurisdictional collaboration."""
)

st.image((ROOT_PATH / "nbs/images/baner.png").as_posix())

st.info(
    "The JuDDGES project encompasses several Work Packages (WPs) designed to cover all aspects of its objectives, from project management to the open science practices and engaging early career researchers. Below is an overview of the project’s WPs based on the provided information."
)

st.header("WP1: Project Management")
st.subheader("Duration: 24 Months")

st.info(
    "Main Aim: To ensure the project’s successful completion on time and within budget. This includes administrative management, scientific and technological management, quality innovation and risk management, ethical and legal consideration, and facilitating open science."
)

st.header("WP2: Gathering and Human Encoding of Judicial Decision Data")
st.subheader("Duration: 22 Months")

st.info(
    "Main Aim: To establish the data foundation for developing and testing the project’s tools. This involves collating/gathering legal case records and judgments, developing a coding scheme, training human coders, making human-coded data available for WP3, facilitating human-in-loop coding for WP3, and enabling WP4 to make data open and reusable beyond the project team."
)

st.header("WP3: NLP and HITL Machine Learning Methodological Development")
st.subheader("Duration: 24 Months")

st.info(
    "Main Aim: To create a bridge between machine learning (led by WUST and MUHEC) and Open Science facilitation (by ELICO), focusing on the development and deployment of annotation methodologies. This includes baseline information extraction, intelligent inference methods for legal corpus data, and constructing an annotation tool through active learning and human-in-the-loop annotation methods."
)

st.header("WP4: Open Science Practices & Engaging Early Career Researchers")
st.subheader("Duration: 12 Months")

st.info(
    "Main Aim: To implement the Open Science policy of the call and engage with relevant early career researchers (ECRs). Objectives include providing open access to publication data and software, disseminating/exploiting project results, and promoting the project and its findings."
)

st.info(
    "Each WP includes specific tasks aimed at achieving its goals, involving collaboration among project partners and contributing to the overarching aim of the JuDDGES project​​."
)

st.header("Project Partners")

st.subheader("Wroclaw University of Science and Technology (WUST)")
st.subheader("Middlesex University London (UK)")
st.subheader("University of Lyon 1 (France)​​.")
