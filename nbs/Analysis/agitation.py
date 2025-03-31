#!/usr/bin/env python
# coding: utf-8

"""
Script to analyze electoral judgments focusing on digital campaign materials
under Article 111 § 1 of the Electoral Code.
"""
import asyncio
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from settings import ARTICLE_111_DATA_PATH, prepare_langchain_cache
from tqdm import tqdm

from juddges.data.weaviate_db import WeaviateJudgmentsDatabase
from juddges.llms import GPT_4o
from juddges.prompts.information_extraction import prepare_information_extraction_chain
from juddges.prompts.schemas.agitation import AGITATION_SCHEMA

BATCH_SIZE = 100
LLM_BATCH_SIZE = 10
MAX_OBJECTS = 10_000
MAX_TEXT_LENGTH = 150000

QUERY_COLUMN = "query"


# Configure matplotlib for non-interactive use
plt.switch_backend("agg")

AGITATION_QUERIES = {
    "bm25_queries": [
        "art. 111 kodeksu wyborczego",
        "fałszywe informacje w materiałach wyborczych",
        "postępowanie wyborcze internetowe",
        "kampania wyborcza media społecznościowe",
        "pozew wyborczy fałszywe informacje",
        "proces wyborczy dezinformacja",
        "orzeczenie sądu kampania wyborcza online",
        "orzeczenie fałszywe informacje wybory",
        "orzeczenia dotyczące agitacji wyborczej",
        "orzeczenie sądu art. 111 Kodeks wyborczy",
    ],
    "vector_queries": [
        "Orzeczenia dotyczące fałszywych informacji wyborczych rozpowszechnianych online podczas kampanii",
        "Polskie orzeczenia sądowe w sporach dotyczących prawa wyborczego w mediach społecznościowych",
        "Wyroki dotyczące wprowadzających w błąd lub fałszywych treści kampanii",
        "Sprawy, w których kandydat zakwestionował treści wyborcze z powodu dezinformacji",
        "Środki prawne stosowane do przeciwdziałania dezinformacji online w wyborach",
        "Postępowania sądowe dotyczące cyfrowej propagandy wyborczej",
        "Reakcja sądownictwa na kampanie algorytmiczne lub działania VLOP",
        "Jak polskie sądy interpretują artykuł 111 w sprawach cyfrowych",
        "Spory dotyczące kampanii prowadzonych przez osoby trzecie bez zgody komitetu",
        "Praktyki sądowe dotyczące fałszywych materiałów wyborczych online",
    ],
}


async def bm25_judgment_search(
    db: WeaviateJudgmentsDatabase, queries: List[str], max_objects: int = MAX_OBJECTS
) -> pd.DataFrame:
    """
    Retrieve all judgments related to Article 111 § 1 of Electoral Code using BM25 search.

    Args:
        db: WeaviateJudgmentsDatabase instance

    Returns:
        List of judgment dictionaries
    """
    output_path = ARTICLE_111_DATA_PATH / "all_judgments_bm25.pkl"

    # Check if the file already exists
    if os.path.exists(output_path):
        logger.info(f"Loading existing BM25 judgments from {output_path}")
        return pd.read_pickle(output_path)

    logger.info("Searching for all Article 111 § 1 judgments using BM25 search")

    # Define multiple search queries instead of regex patterns
    search_queries = queries

    all_judgments = {}  # Dictionary with query as key
    query_stats = {}  # Track statistics for each query

    # Initialize progress bar with unknown total
    pbar = tqdm(desc="Retrieving judgments", unit="query")

    # Process each query
    for query in search_queries:
        logger.info(f"Executing BM25 search with query: '{query}'")
        offset = 0
        query_judgments = []
        query_scores = []

        while True:
            # Query the judgments collection using BM25
            search = db.judgments_collection.query.bm25(
                query=query,
                query_properties=["full_text", "legal_bases"],
                limit=BATCH_SIZE,
                offset=offset,
                include_vector=True,
                return_metadata=[
                    "creation_time",
                    "last_update_time",
                    "distance",
                    "certainty",
                    "score",
                    "explain_score",
                    "is_consistent",
                ],
                return_properties=[
                    "judgment_id",
                    "full_text",
                    "court_name",
                    "decision",
                    "docket_number",
                    "judgment_date",
                    "legal_bases",
                ],
            )

            # Execute the query
            result = await search
            batch_judgments = result.objects

            # If no more results, break the loop
            if not batch_judgments:
                break

            # Process judgments and collect scores
            for judgment in batch_judgments:
                query_judgments.append(judgment)
                query_scores.append(judgment.metadata.score)

            # Increment offset for the next batch
            offset += BATCH_SIZE

            # If we reached the max, break the loop
            if len(query_judgments) >= max_objects:
                break

        # Store query statistics
        if query_scores:
            query_stats[query] = {
                "count": len(query_judgments),
                "min_score": min(query_scores),
                "max_score": max(query_scores),
                "avg_score": sum(query_scores) / len(query_scores),
            }
        else:
            query_stats[query] = {
                "count": 0,
                "min_score": 0,
                "max_score": 0,
                "avg_score": 0,
            }

        # Log query statistics
        logger.info(
            f"Query '{query}': Found {query_stats[query]['count']} judgments, "
            f"Score min: {query_stats[query]['min_score']:.4f}, "
            f"max: {query_stats[query]['max_score']:.4f}, "
            f"avg: {query_stats[query]['avg_score']:.4f}"
        )

        # Add judgments to the dictionary with query as key
        all_judgments[query] = query_judgments

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix(
            {"total": sum(len(judgments) for judgments in all_judgments.values())}
        )

    # Close progress bar
    pbar.close()

    # save as df all judgments
    data = []
    for query, judgments in all_judgments.items():
        for judgment in judgments:

            judgment_dict = judgment.__dict__
            judgment_dict["uuid"] = str(judgment_dict["uuid"])

            data.append(
                {
                    **judgment_dict,
                    QUERY_COLUMN: query,
                    "search_type": "bm25",
                }
            )

    judgments_df = pd.DataFrame(data)
    judgments_df.to_pickle(output_path)

    return judgments_df


def deduplicate_judgments(judgments: List[Dict]) -> List[Dict]:
    """
    Deduplicate judgments and accumulate scores for duplicates.

    Args:
        judgments: List of judgment objects

    Returns:
        List of deduplicated judgment objects with accumulated scores
    """
    logger.info(f"Deduplicating {len(judgments)} judgments...")

    judgment_scores = {}  # Track scores for each judgment

    # Process judgments and accumulate scores for duplicates
    for judgment in judgments:
        judgment_id = judgment.properties.get("judgment_id")
        score = judgment.metadata.score

        if judgment_id in judgment_scores:
            # Sum the score for existing judgment
            judgment_scores[judgment_id]["score"] += score
        else:
            # Add new judgment with its score
            judgment_scores[judgment_id] = {
                "judgment": judgment,
                "score": score,
            }

    # Sort judgments by accumulated score and extract judgment objects
    sorted_judgments = sorted(
        judgment_scores.items(), key=lambda x: x[1]["score"], reverse=True
    )

    # Create deduplicated list with accumulated scores
    deduplicated_judgments = []
    for judgment_id, data in sorted_judgments:
        judgment = data["judgment"]
        judgment.accumulated_score = data["score"]
        deduplicated_judgments.append(judgment)

    logger.info(f"Deduplicated to {len(deduplicated_judgments)} unique judgments")

    # Log some statistics about deduplication
    duplicate_rate = (
        (len(judgments) - len(deduplicated_judgments)) / len(judgments) * 100
    )
    logger.info(f"Duplicate rate: {duplicate_rate:.2f}%")

    return deduplicated_judgments


async def semantic_judgment_search(
    db: WeaviateJudgmentsDatabase, queries: List[str], max_objects: int = MAX_OBJECTS
) -> pd.DataFrame:
    """
    Perform semantic search for digital campaign materials in judgments.

    Args:
        db: WeaviateJudgmentsDatabase instance
        queries: List of search queries
        max_objects: Maximum number of objects to retrieve

    Returns:
        DataFrame containing judgment data with query information
    """
    output_path = ARTICLE_111_DATA_PATH / "all_judgments_vector.pkl"

    # Check if the file already exists
    if os.path.exists(output_path):
        logger.info(f"Loading existing vector judgments from {output_path}")
        return pd.read_pickle(output_path)

    model = SentenceTransformer("sdadas/mmlw-roberta-large")

    logger.info("Performing semantic search for digital campaign materials")

    all_judgments = {}  # Dictionary with query as key
    query_stats = {}  # Track statistics for each query

    # Initialize progress bar with unknown total
    pbar = tqdm(desc="Retrieving judgments", unit="query")

    # Process each query
    for query in queries:
        logger.info(f"Executing semantic search with query: '{query}'")
        offset = 0
        query_judgments = []
        query_scores = []

        while True:
            # Embed the query
            vector = model.encode(query)
            # Query the judgments collection using vector search
            search = db.judgments_collection.query.near_vector(
                near_vector=vector,
                limit=BATCH_SIZE,
                distance=0.5,
                offset=offset,
                include_vector=True,
                return_metadata=[
                    "creation_time",
                    "last_update_time",
                    "distance",
                    "certainty",
                    "score",
                    "explain_score",
                    "is_consistent",
                ],
                return_properties=[
                    "judgment_id",
                    "full_text",
                    "court_name",
                    "decision",
                    "docket_number",
                    "judgment_date",
                    "legal_bases",
                ],
            )

            # Execute the query
            try:
                result = await search
                batch_judgments = result.objects
            except Exception as e:
                logger.error(f"Error in semantic search with query '{query}': {e}")
                break

            # If no more results, break the loop
            if not batch_judgments:
                break

            # Process judgments and collect scores
            for judgment in batch_judgments:
                query_judgments.append(judgment)
                # Use distance as the score for near_vector search
                query_scores.append(judgment.metadata.distance)

            # Update offset for next batch
            offset += BATCH_SIZE

            # Check if we've reached the maximum number of objects
            if len(query_judgments) >= max_objects:
                logger.info(
                    f"Reached maximum number of objects ({max_objects}) for query '{query}'"
                )
                break

        # Store judgments and statistics for this query
        all_judgments[query] = query_judgments
        query_stats[query] = {
            "count": len(query_judgments),
            "avg_score": sum(query_scores) / len(query_scores) if query_scores else 0,
            "max_score": max(query_scores) if query_scores else 0,
            "min_score": min(query_scores) if query_scores else 0,
        }

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix(count=len(query_judgments))

    # Close progress bar
    pbar.close()

    data = []
    for query, judgments in all_judgments.items():
        for judgment in judgments:
            judgment_dict = judgment.__dict__
            judgment_dict["uuid"] = str(judgment_dict["uuid"])

            data.append(
                {
                    **judgment_dict,
                    QUERY_COLUMN: query,
                    "search_type": "vector",
                }
            )

    judgments_df = pd.DataFrame(data)
    judgments_df.to_pickle(output_path)

    return judgments_df


async def extract_judgment_information(df, batch_size=LLM_BATCH_SIZE):
    """
    Extract structured information from judgment texts using LLM.

    Args:
        df: DataFrame containing judgment data
        batch_size: Number of judgments to process in each batch

    Returns:
        Tuple of (DataFrame with extraction results, list of extracted data)
    """
    # Choose the model for information extraction
    MODEL_NAME = GPT_4o
    LANGUAGE = "polish"  # Polish language for judgments

    logger.info(
        f"Starting information extraction using {MODEL_NAME} model on {len(df)} judgments"
    )

    # Create the extraction chain
    extraction_chain = prepare_information_extraction_chain(model_name=MODEL_NAME)

    # Process judgments in batches to avoid memory issues
    all_judgments_data = []

    # Create a copy of the dataframe to add extraction results
    judgments_with_extraction = df.copy()
    judgments_with_extraction["extracted_info"] = None

    # Use tqdm to show progress
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        logger.info(
            f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}"
        )

        batch_inputs = []
        batch_indices = []

        for idx, (_, judgment) in enumerate(batch.iterrows()):
            # Get the full text from the judgment
            content = judgment.get("properties.full_text", "")
            if not content:
                continue

            batch_inputs.append(
                {
                    "TEXT": content[:MAX_TEXT_LENGTH],
                    "SCHEMA": AGITATION_SCHEMA,
                    "LANGUAGE": LANGUAGE,
                }
            )
            batch_indices.append(i + idx)

        # Skip empty batches
        if not batch_inputs:
            logger.warning("Skipping empty batch")
            continue

        # Process the batch
        try:
            batch_results = extraction_chain.batch(batch_inputs)

            # Process each result
            for result, (_, judgment), idx in zip(
                batch_results, batch.iterrows(), batch_indices
            ):
                if result:
                    # Add metadata to the result
                    result.update(
                        {
                            "judgment_id": judgment.get("judgment_id"),
                            "court_name": judgment.get("properties.court_name"),
                            "docket_number": judgment.get("properties.docket_number"),
                            "judgment_date": judgment.get("properties.judgment_date"),
                        }
                    )

                    all_judgments_data.append(result)

                    # Add extraction result to the dataframe
                    df_idx = batch.index[idx - i]
                    judgments_with_extraction.loc[df_idx, "extracted_info"] = result
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            continue

    return judgments_with_extraction, all_judgments_data


async def main():
    prepare_langchain_cache()

    """Main execution function"""
    async with WeaviateJudgmentsDatabase() as db:
        # Get all Article 111 § 1 judgments
        bm25_judgments_df = await bm25_judgment_search(
            db, AGITATION_QUERIES["bm25_queries"]
        )

        # Get all Article 111 § 1 judgments
        vector_judgments_df = await semantic_judgment_search(
            db, AGITATION_QUERIES["vector_queries"]
        )

        # Plot score distributions for both search types
        plot_search_score_distributions(bm25_judgments_df, vector_judgments_df)

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
        visualize_score_distribution(merged_df)

        # Save all results before filtering
        merged_df.to_pickle(ARTICLE_111_DATA_PATH / "all_judgments_merged.pkl")

        # Filter with recommended threshold (adjust based on visualization)
        filtered_df = merged_df[merged_df["relevance_score"] >= 0.6]
        logger.info(f"Filtered from {len(merged_df)} to {len(filtered_df)} judgments")

        # Save filtered results
        filtered_df.to_pickle(ARTICLE_111_DATA_PATH / "filtered_judgments.pkl")

        # Extract information from filtered judgments only
        # This will be more efficient than processing all judgments
        # judgments_with_extraction, extracted_data = await extract_judgment_information(
        #     filtered_df
        # )

        # # Save extracted information
        # pd.DataFrame(extracted_data).to_pickle(
        #     ARTICLE_111_DATA_PATH / "extracted_judgment_data.pkl"
        # )
        # judgments_with_extraction.to_pickle(
        #     ARTICLE_111_DATA_PATH / "judgments_with_extraction.pkl"
        # )


def plot_search_score_distributions(bm25_df, vector_df):
    """
    Plot score distributions for BM25 and vector search to help determine thresholds.

    Args:
        bm25_df: DataFrame with BM25 search results
        vector_df: DataFrame with vector search results
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
    plt.savefig(ARTICLE_111_DATA_PATH / "search_score_distributions.png")
    plt.close()

    logger.info(
        f"Search score distributions saved to {ARTICLE_111_DATA_PATH / 'search_score_distributions.png'}"
    )


def merge_judgment_results(bm25_df, vector_df):
    """
    Merge BM25 and vector search results with deduplication.

    Returns:
        DataFrame with combined results and relevance scores
    """
    logger.info("Merging BM25 and vector search results")

    # Concatenate dataframes
    combined_df = pd.concat([bm25_df, vector_df])

    # Normalize scores between search types
    for search_type in ["bm25", "vector"]:
        mask = combined_df["search_type"] == search_type
        if mask.any():
            score_column = (
                "metadata.score" if search_type == "bm25" else "metadata.distance"
            )
            score_data = pd.json_normalize(combined_df.loc[mask, "metadata"])

            # Handle different scoring directions (higher is better for BM25, lower is better for vector)
            if search_type == "bm25":
                max_score = score_data["score"].max()
                if max_score > 0:
                    combined_df.loc[mask, "normalized_score"] = (
                        score_data["score"] / max_score
                    )
            else:
                # For vector search, invert the distance (1 - normalized distance)
                max_dist = score_data["distance"].max()
                if max_dist > 0:
                    combined_df.loc[mask, "normalized_score"] = 1 - (
                        score_data["distance"] / max_dist
                    )

    # Group by judgment_id to deduplicate
    grouped = (
        combined_df.groupby("judgment_id")
        .agg(
            {
                "normalized_score": "mean",
                QUERY_COLUMN: lambda x: list(set(x)),
                "search_type": lambda x: list(set(x)),
                "properties.court_name": "first",
                "properties.docket_number": "first",
                "properties.judgment_date": "first",
                "properties.full_text": "first",
            }
        )
        .reset_index()
    )

    # Add query count as a relevance signal
    grouped["query_count"] = grouped[QUERY_COLUMN].apply(len)

    # Calculate combined relevance score
    grouped["relevance_score"] = grouped["normalized_score"] * (
        1 + grouped["query_count"] / 10
    )

    logger.info(
        f"Merged {len(bm25_df)} BM25 and {len(vector_df)} vector results into {len(grouped)} unique judgments"
    )

    return grouped.sort_values("relevance_score", ascending=False)


def visualize_score_distribution(merged_df):
    """Create visualization to help determine appropriate thresholds."""
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
    plt.savefig(ARTICLE_111_DATA_PATH / "score_distribution.png")
    plt.close()

    logger.info(
        f"Score visualization saved to {ARTICLE_111_DATA_PATH / 'score_distribution.png'}"
    )


if __name__ == "__main__":
    asyncio.run(main())
