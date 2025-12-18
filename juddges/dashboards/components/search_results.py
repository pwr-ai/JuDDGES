import streamlit as st

from juddges.dashboards.utils.search_utils import (
    format_highlight_segments,
    process_highlights,
)
from juddges.data_models import Judgment


def display_search_results(items: list[dict]) -> None:
    """Display search results in a formatted way using cards and columns.

    Args:
        items: List of search result items to display
    """
    st.header(f"judgments - Results ({len(items)})")

    # Custom CSS for cards
    st.markdown(
        """
        <style>
        .judgment-card {
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            margin: 10px 0;
            background-color: white;
        }
        .metadata-label {
            font-weight: bold;
            color: #555;
        }
        .score-badge {
            background-color: #0066cc;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 14px;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    for item in items:
        with st.container():
            st.markdown('<div class="judgment-card">', unsafe_allow_html=True)

            # Header row with signature and score
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.markdown(f"### {item[Judgment.SIGNATURE.value]}")
            with col2:
                st.markdown(
                    f'<div class="score-badge">Score: {item["score"]:.2f}</div>',
                    unsafe_allow_html=True,
                )

            # Metadata columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f'<span class="metadata-label">Court:</span> {item[Judgment.COURT_NAME.value]}',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f'<span class="metadata-label">Department:</span> {item.get(Judgment.DEPARTMENT_NAME.value, "N/A")}',
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f'<span class="metadata-label">Date:</span> {item[Judgment.DATE.value]}',
                    unsafe_allow_html=True,
                )

            # Excerpt
            st.markdown("#### Excerpt")
            st.markdown(item.get(Judgment.EXCERPT.value, "No excerpt available"))

            # Highlighted excerpts in an expander
            if "highlights" in item:
                with st.expander("View Highlighted Excerpts"):
                    for highlight in item["highlights"]:
                        text = format_highlight_segments(highlight)
                        st.markdown(text, unsafe_allow_html=True)
                        st.markdown("---")

            # Full text in an expander
            with st.expander("View Full Text"):
                full_text = process_highlights(item)
                st.markdown(full_text, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---")
