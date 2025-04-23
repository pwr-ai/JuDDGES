"""
Functions for processing judgment data.
"""

import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

from juddges.use_cases.agitation.search import QUERY_COLUMN


def merge_judgment_results(bm25_df, vector_df):
    """
    Merge BM25 and vector search results with deduplication.

    Args:
        bm25_df: DataFrame with BM25 search results
        vector_df: DataFrame with vector search results

    Returns:
        DataFrame with combined results and relevance scores
    """
    logger.info("Merging BM25 and vector search results")
    assert bm25_df.columns.equals(vector_df.columns), "Columns must be the same"

    # Concatenate dataframes
    combined_df = pd.concat([bm25_df, vector_df])

    # Count occurrences of each judgment_id
    judgment_counts = pd.DataFrame(combined_df["uuid"].value_counts()).reset_index()
    judgment_counts.columns = ["uuid", "occurrence_count"]

    # Merge occurrence counts back into combined_df
    combined_df = combined_df.merge(judgment_counts, on="uuid", how="left")
    combined_df.sort_values(by="occurrence_count", ascending=False, inplace=True)

    # Group by judgment_id to deduplicate
    grouped = (
        combined_df.groupby("uuid")
        .agg(
            {
                "occurrence_count": "first",
                QUERY_COLUMN: lambda x: list(set(x)),
                "search_type": lambda x: list(set(x)),
                "properties": "first",
                "metadata": "first",
                "references": "first",
                "vector": "first",
                "collection": "first",
            }
        )
        .reset_index()
    )

    # Add query count as a relevance signal
    grouped["query_count"] = grouped[QUERY_COLUMN].apply(len)

    return grouped


def filter_judgments(merged_df, threshold=10):
    """
    Filter judgments based on query count.

    Args:
        merged_df: DataFrame with merged judgment results
        threshold: Minimum query count

    Returns:
        DataFrame with filtered judgments
    """
    filtered_df = merged_df[merged_df["query_count"] >= threshold]
    logger.info(f"Filtered from {len(merged_df)} to {len(filtered_df)} judgments")
    return filtered_df
