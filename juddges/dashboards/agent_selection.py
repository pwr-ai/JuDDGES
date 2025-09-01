from typing import Any, Dict

import streamlit as st
import yaml

from juddges.data.pl_court_api import PolishCourtAPI
from juddges.llms import (
    CLAUDE_3_5_SONNET,
    CLAUDE_3_HAIKU,
    CLAUDE_3_OPUS,
    GPT_4_0125_PREVIEW,
    GPT_4o,
    GPT_4o_MINI,
)
from juddges.prompts.information_extraction import (
    prepare_information_extraction_chain,
    prepare_schema_chain,
)
from juddges.settings import prepare_langchain_cache, prepare_mlflow

prepare_langchain_cache()
prepare_mlflow()

TITLE = "🤖 JuDDGES AI Agent Selection 🤖"

# Define available agents with their configurations
AGENTS = {
    "schema_generator": {
        "name": "Schema Generation Agent",
        "description": "Generates extraction schemas from natural language descriptions for legal document processing",
        "icon": "📋",
        "use_case": "Create structured schemas for extracting specific information from legal documents",
        "example": "Generate a schema to extract court dates, case numbers, and judge names from court decisions",
        "models": [GPT_4o, GPT_4o_MINI, GPT_4_0125_PREVIEW, CLAUDE_3_5_SONNET],
        "default_model": GPT_4o_MINI,
    },
    "information_extractor": {
        "name": "Information Extraction Agent",
        "description": "Extracts structured information from legal documents using predefined or custom schemas",
        "icon": "📄",
        "use_case": "Extract specific data points from legal texts in a structured format",
        "example": "Extract loan amounts, court decisions, and legal precedents from Swiss franc loan cases",
        "models": [GPT_4o, GPT_4o_MINI, GPT_4_0125_PREVIEW, CLAUDE_3_5_SONNET, CLAUDE_3_HAIKU],
        "default_model": GPT_4o,
    },
    "legal_search": {
        "name": "Legal Document Search Agent",
        "description": "Performs semantic search through legal document databases using natural language queries",
        "icon": "🔍",
        "use_case": "Find relevant legal documents, precedents, and case law using natural language",
        "example": "Search for court decisions related to employment discrimination in the last 5 years",
        "models": [GPT_4o, GPT_4o_MINI, CLAUDE_3_5_SONNET],
        "default_model": GPT_4o_MINI,
    },
    "legal_analyzer": {
        "name": "Legal Analysis Agent",
        "description": "Analyzes legal trends, patterns, and relationships in court decisions and legal documents",
        "icon": "📊",
        "use_case": "Identify patterns, trends, and insights from large collections of legal documents",
        "example": "Analyze trends in court sentencing for financial crimes over the past decade",
        "models": [GPT_4o, GPT_4_0125_PREVIEW, CLAUDE_3_5_SONNET, CLAUDE_3_OPUS],
        "default_model": GPT_4o,
    },
    "document_summarizer": {
        "name": "Court Decision Summarization Agent",
        "description": "Creates concise summaries of court decisions and legal documents",
        "icon": "📝",
        "use_case": "Generate executive summaries of lengthy court decisions and legal documents",
        "example": "Summarize a 50-page court decision into key findings, reasoning, and outcome",
        "models": [GPT_4o, GPT_4o_MINI, CLAUDE_3_5_SONNET, CLAUDE_3_HAIKU],
        "default_model": GPT_4o_MINI,
    },
}


def display_agent_card(agent_key: str, agent_config: Dict[str, Any]) -> None:
    """Display an agent configuration card"""
    with st.container():
        col1, col2 = st.columns([1, 4])

        with col1:
            st.markdown(f"## {agent_config['icon']}")

        with col2:
            st.markdown(f"### {agent_config['name']}")
            st.markdown(agent_config["description"])

            with st.expander("View Details"):
                st.markdown(f"**Use Case:** {agent_config['use_case']}")
                st.markdown(f"**Example:** _{agent_config['example']}_")
                st.markdown(f"**Available Models:** {', '.join(agent_config['models'])}")
                st.markdown(f"**Default Model:** {agent_config['default_model']}")

            if st.button(f"Select {agent_config['name']}", key=f"select_{agent_key}"):
                st.session_state.selected_agent = agent_key
                st.session_state.selected_agent_config = agent_config
                st.rerun()


def run_schema_generator():
    """Run the schema generation agent"""
    st.subheader("🧠 Schema Generation Agent")
    st.info(
        "Describe what information you want to extract, and I'll create a structured schema for you."
    )

    schema_query = st.text_area(
        "Describe the information you want to extract:",
        placeholder="I need to extract court dates, case numbers, judge names, and case outcomes from court decisions",
        height=100,
    )

    model = st.selectbox(
        "Select Model:",
        st.session_state.selected_agent_config["models"],
        index=st.session_state.selected_agent_config["models"].index(
            st.session_state.selected_agent_config["default_model"]
        ),
    )

    if st.button("Generate Schema"):
        if schema_query.strip():
            with st.spinner("Generating schema..."):
                try:
                    chain = prepare_schema_chain(model_name=model)
                    schema = chain.invoke({"SCHEMA_TEXT": schema_query})

                    if schema:
                        st.success("Schema generated successfully!")
                        st.markdown("### Generated Schema:")
                        st.code(
                            yaml.dump(schema, allow_unicode=True, sort_keys=False), language="yaml"
                        )
                        st.session_state.generated_schema = schema
                    else:
                        st.error(
                            "Could not generate schema. Please try with a different description."
                        )
                except Exception as e:
                    st.error(f"Error generating schema: {str(e)}")
        else:
            st.warning("Please provide a description of the information you want to extract.")


def run_information_extractor():
    """Run the information extraction agent"""
    st.subheader("📄 Information Extraction Agent")
    st.info("Extract structured information from legal documents using a predefined schema.")

    # Document input
    st.markdown("#### Document Input")
    source_option = st.selectbox(
        "Choose document source:", ["URL (Polish Court API)", "Plain text"]
    )

    if source_option == "URL (Polish Court API)":
        api = PolishCourtAPI()
        judgment_url = st.text_input(
            "Enter judgment URL:", placeholder="https://orzeczenia.wroclaw.sa.gov.pl/details/$N/..."
        )
        if judgment_url:
            judgment_id = judgment_url.strip().split("/")[-1]
            try:
                judgment_text = api.get_content(id=judgment_id)
                st.text_area(
                    "Document preview:", judgment_text[:500] + "...", height=100, disabled=True
                )
            except Exception as e:
                st.error(f"Error fetching document: {str(e)}")
                judgment_text = ""
        else:
            judgment_text = ""
    else:
        judgment_text = st.text_area("Enter document text:", height=200)

    # Schema input
    st.markdown("#### Extraction Schema")
    schema_text = st.text_area(
        "Enter extraction schema (YAML):",
        value=st.session_state.get("generated_schema", ""),
        height=150,
        help="Use the Schema Generation Agent to create a schema, or write your own in YAML format",
    )

    # Model selection
    model = st.selectbox(
        "Select Model:",
        st.session_state.selected_agent_config["models"],
        index=st.session_state.selected_agent_config["models"].index(
            st.session_state.selected_agent_config["default_model"]
        ),
    )

    language = st.selectbox("Document language:", ["Polish", "English"])

    if st.button("Extract Information"):
        if judgment_text.strip() and schema_text.strip():
            with st.spinner("Extracting information..."):
                try:
                    chain = prepare_information_extraction_chain(model_name=model)
                    result = chain.invoke(
                        {"LANGUAGE": language, "TEXT": judgment_text, "SCHEMA": schema_text}
                    )

                    st.success("Information extracted successfully!")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Original Document")
                        st.text_area("", judgment_text, height=400, disabled=True)

                    with col2:
                        st.markdown("### Extracted Information")
                        st.json(result)

                except Exception as e:
                    st.error(f"Error extracting information: {str(e)}")
        else:
            st.warning("Please provide both document text and extraction schema.")


def run_legal_search():
    """Run the legal search agent"""
    st.subheader("🔍 Legal Document Search Agent")
    st.info("Search through legal documents using natural language queries.")

    search_query = st.text_input(
        "Enter your search query:", placeholder="Find cases about employment discrimination in 2023"
    )

    # Search filters
    col1, col2 = st.columns(2)
    with col1:
        jurisdiction = st.selectbox("Jurisdiction:", ["All", "Polish Courts", "English Courts"])
    with col2:
        doc_type = st.selectbox(
            "Document Type:", ["All", "Court Decisions", "Appeals", "Regulations"]
        )

    if st.button("Search Documents"):
        if search_query.strip():
            with st.spinner("Searching legal documents..."):
                # Placeholder for search functionality
                st.info("🚧 Search functionality will be implemented with Weaviate integration")
                st.markdown("**Query:** " + search_query)
                st.markdown("**Filters:** " + f"Jurisdiction: {jurisdiction}, Type: {doc_type}")
        else:
            st.warning("Please enter a search query.")


def run_legal_analyzer():
    """Run the legal analysis agent"""
    st.subheader("📊 Legal Analysis Agent")
    st.info("Analyze legal trends and patterns in court decisions.")

    analysis_type = st.selectbox(
        "Analysis Type:",
        ["Trend Analysis", "Pattern Recognition", "Comparative Analysis", "Statistical Summary"],
    )

    analysis_query = st.text_area(
        "Describe what you want to analyze:",
        placeholder="Analyze sentencing trends for financial crimes in Polish courts over the last 5 years",
        height=100,
    )

    if st.button("Run Analysis"):
        if analysis_query.strip():
            with st.spinner("Running legal analysis..."):
                # Placeholder for analysis functionality
                st.info(
                    "🚧 Analysis functionality will be implemented with data processing pipeline"
                )
                st.markdown(f"**Analysis Type:** {analysis_type}")
                st.markdown(f"**Query:** {analysis_query}")
        else:
            st.warning("Please describe what you want to analyze.")


def run_document_summarizer():
    """Run the document summarization agent"""
    st.subheader("📝 Court Decision Summarization Agent")
    st.info("Generate concise summaries of court decisions and legal documents.")

    # Document input (similar to information extractor)
    source_option = st.selectbox(
        "Choose document source:", ["URL (Polish Court API)", "Plain text"]
    )

    if source_option == "URL (Polish Court API)":
        api = PolishCourtAPI()
        judgment_url = st.text_input(
            "Enter judgment URL:", placeholder="https://orzeczenia.wroclaw.sa.gov.pl/details/$N/..."
        )
        if judgment_url:
            judgment_id = judgment_url.strip().split("/")[-1]
            try:
                judgment_text = api.get_content(id=judgment_id)
            except Exception as e:
                st.error(f"Error fetching document: {str(e)}")
                judgment_text = ""
        else:
            judgment_text = ""
    else:
        judgment_text = st.text_area("Enter document text:", height=300)

    summary_type = st.selectbox(
        "Summary Type:",
        ["Executive Summary", "Key Findings", "Legal Reasoning", "Case Outcome", "Full Summary"],
    )

    model = st.selectbox(
        "Select Model:",
        st.session_state.selected_agent_config["models"],
        index=st.session_state.selected_agent_config["models"].index(
            st.session_state.selected_agent_config["default_model"]
        ),
    )

    if st.button("Generate Summary"):
        if judgment_text.strip():
            with st.spinner("Generating summary..."):
                # Placeholder for summarization functionality
                st.info("🚧 Summarization functionality will be implemented with LLM chains")
                st.markdown(f"**Summary Type:** {summary_type}")
                st.markdown(f"**Model:** {model}")
                st.markdown("**Document length:** " + str(len(judgment_text)) + " characters")
        else:
            st.warning("Please provide document text to summarize.")


def main():
    st.title(TITLE)

    st.markdown("""
    Welcome to the JuDDGES AI Agent Selection interface! Choose from specialized AI agents 
    designed for different legal document processing tasks.
    """)

    # Check if an agent is already selected
    if "selected_agent" in st.session_state and st.session_state.selected_agent:
        selected_agent = st.session_state.selected_agent
        agent_config = st.session_state.selected_agent_config

        # Show current selection and option to change
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Currently using: **{agent_config['name']}** {agent_config['icon']}")
        with col2:
            if st.button("Change Agent"):
                del st.session_state.selected_agent
                del st.session_state.selected_agent_config
                st.rerun()

        st.divider()

        # Run the selected agent
        if selected_agent == "schema_generator":
            run_schema_generator()
        elif selected_agent == "information_extractor":
            run_information_extractor()
        elif selected_agent == "legal_search":
            run_legal_search()
        elif selected_agent == "legal_analyzer":
            run_legal_analyzer()
        elif selected_agent == "document_summarizer":
            run_document_summarizer()

    else:
        # Show agent selection interface
        st.markdown("## Available AI Agents")

        for agent_key, agent_config in AGENTS.items():
            display_agent_card(agent_key, agent_config)
            st.divider()


if __name__ == "__main__":
    main()
