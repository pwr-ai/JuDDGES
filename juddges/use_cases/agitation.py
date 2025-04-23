"""
Script to analyze electoral judgments focusing on digital campaign materials
under Article 111 § 1 of the Electoral Code.
"""

import asyncio
import os

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from prompts.schemas.agitation import AGITATION_SCHEMA
from settings import ARTICLE_111_DATA_PATH, prepare_langchain_cache

from juddges.data.weaviate_db import WeaviateJudgmentsDatabase
from juddges.use_cases.agitation.constants import AGITATION_QUERIES
from juddges.use_cases.agitation.extraction import extract_judgment_information
from juddges.use_cases.agitation.processing import (
    filter_judgments,
    merge_judgment_results,
)
from juddges.use_cases.agitation.search import (
    bm25_judgment_search,
    semantic_judgment_search,
)
from juddges.use_cases.visualization import plot_search_score_distributions

# Configure matplotlib for non-interactive use
plt.switch_backend("agg")

MIN_THRESHOLD = 5


async def main(overwrite: bool = False):
    """
    Main execution function.

    Args:
        overwrite: Whether to overwrite existing files
    """
    prepare_langchain_cache()

    # Create output directory if it doesn't exist
    os.makedirs(ARTICLE_111_DATA_PATH, exist_ok=True)

    # Set up file paths
    bm25_output_path = ARTICLE_111_DATA_PATH / "all_judgments_bm25.pkl"
    vector_output_path = ARTICLE_111_DATA_PATH / "all_judgments_vector.pkl"
    merged_output_path = ARTICLE_111_DATA_PATH / "all_judgments_merged.pkl"
    filtered_output_path = ARTICLE_111_DATA_PATH / "filtered_judgments.pkl"
    score_dist_path = ARTICLE_111_DATA_PATH / "search_score_distributions.png"
    judgments_with_extraction_path = (
        ARTICLE_111_DATA_PATH / "judgments_with_extraction.pkl"
    )

    async with WeaviateJudgmentsDatabase() as db:
        # Get all Article 111 § 1 judgments using BM25 search
        bm25_judgments_df = await bm25_judgment_search(
            db,
            AGITATION_QUERIES["bm25_queries"],
            output_path=bm25_output_path,
            overwrite=overwrite,
        )

        # Get all Article 111 § 1 judgments using vector search
        vector_judgments_df = await semantic_judgment_search(
            db,
            AGITATION_QUERIES["vector_queries"],
            output_path=vector_output_path,
            overwrite=overwrite,
        )

        # Plot score distributions for both search types
        plot_search_score_distributions(
            bm25_judgments_df, vector_judgments_df, score_dist_path
        )

        # Merge results
        merged_df = merge_judgment_results(bm25_judgments_df, vector_judgments_df)

        # Count total judgments before deduplication
        total_before = len(bm25_judgments_df) + len(vector_judgments_df)
        total_after = len(merged_df)
        duplicate_count = total_before - total_after
        duplicate_percentage = (duplicate_count / total_before) * 100

        logger.info(f"Total judgments before deduplication: {total_before}")
        logger.info(f"Total judgments after deduplication: {total_after}")
        logger.info(
            f"Duplicates removed: {duplicate_count} ({duplicate_percentage:.2f}%)"
        )

        # Visualize score distribution to help determine thresholds
        # visualize_score_distribution(merged_df, score_viz_path)

        # Save all results before filtering
        merged_df.to_pickle(merged_output_path)

        # Filter judgments based on query count
        filtered_df = filter_judgments(merged_df, threshold=MIN_THRESHOLD)

        # Save filtered results
        filtered_df.to_pickle(filtered_output_path)

        # Extract information from filtered judgments
        judgments_with_extraction = await extract_judgment_information(
            filtered_df, AGITATION_SCHEMA
        )

        judgments_with_extraction.to_pickle(judgments_with_extraction_path)

        logger.info(
            f"Successfully extracted information from {len(judgments_with_extraction)} judgments and saved to {judgments_with_extraction_path}"
        )


if __name__ == "__main__":
    asyncio.run(main())
