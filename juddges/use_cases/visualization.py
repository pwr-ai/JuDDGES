import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


def plot_search_score_distributions(bm25_df, vector_df, output_path):
    """
    Plot score distributions for BM25 and vector search to help determine thresholds.

    Args:
        bm25_df: DataFrame with BM25 search results
        vector_df: DataFrame with vector search results
        output_path: Path to save the visualization
    """
    logger.info("Generating score distribution visualizations for search methods")

    plt.figure(figsize=(12, 10))

    # Extract scores
    bm25_scores = bm25_df["metadata"].apply(lambda x: x.score)
    vector_distances = vector_df["metadata"].apply(lambda x: x.distance)

    # BM25 scores (higher is better)
    plt.subplot(2, 1, 1)
    plt.hist(bm25_scores, bins=30, alpha=0.7, color="blue")
    plt.axvline(
        x=bm25_scores.quantile(0.5),
        color="red",
        linestyle="--",
        label=f"Median: {bm25_scores.quantile(0.5):.4f}",
    )
    plt.axvline(
        x=bm25_scores.quantile(0.75),
        color="green",
        linestyle="--",
        label=f"75th percentile: {bm25_scores.quantile(0.75):.4f}",
    )
    plt.title("BM25 Score Distribution")
    plt.xlabel("Score (higher is better)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Vector distances (lower is better)
    plt.subplot(2, 1, 2)
    plt.hist(vector_distances, bins=30, alpha=0.7, color="purple")
    plt.axvline(
        x=vector_distances.quantile(0.5),
        color="red",
        linestyle="--",
        label=f"Median: {vector_distances.quantile(0.5):.4f}",
    )
    plt.axvline(
        x=vector_distances.quantile(0.25),
        color="green",
        linestyle="--",
        label=f"25th percentile: {vector_distances.quantile(0.25):.4f}",
    )
    plt.title("Vector Distance Distribution")
    plt.xlabel("Distance (lower is better)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Search score distributions saved to {output_path}")


def visualize_score_distribution(merged_df, output_path):
    """
    Create visualization to help determine appropriate thresholds.

    Args:
        merged_df: DataFrame with merged judgment results
        output_path: Path to save the visualization
    """
    logger.info("Generating score distribution visualizations")

    plt.figure(figsize=(15, 10))

    # Score distributions
    plt.subplot(2, 2, 1)
    plt.hist(merged_df["normalized_score"], bins=20)
    plt.title("Distribution of Normalized Scores")
    plt.xlabel("Score")
    plt.ylabel("Count")

    plt.subplot(2, 2, 2)
    plt.hist(merged_df["relevance_score"], bins=20)
    plt.title("Distribution of Relevance Scores")
    plt.xlabel("Score")
    plt.ylabel("Count")

    # Cumulative distribution
    plt.subplot(2, 2, 3)
    counts, bins = np.histogram(merged_df["relevance_score"], bins=50)
    plt.stairs(np.cumsum(counts) / len(merged_df), bins)
    plt.title("Cumulative Distribution of Relevance Scores")
    plt.xlabel("Score Threshold")
    plt.ylabel("Proportion of Judgments Retained")
    plt.grid(True)

    # Query count distribution
    plt.subplot(2, 2, 4)
    plt.hist(
        merged_df["query_count"], bins=range(1, merged_df["query_count"].max() + 2)
    )
    plt.title("Distribution of Query Matches per Judgment")
    plt.xlabel("Number of Queries Matched")
    plt.ylabel("Count")
    plt.xticks(range(1, merged_df["query_count"].max() + 1))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Score visualization saved to {output_path}")
