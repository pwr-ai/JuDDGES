"""
Functions for searching judgments related to Article 111 of the Electoral Code.
"""

import os
from typing import Dict, List

import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Constants
BATCH_SIZE = 100
MAX_OBJECTS = 1_000
QUERY_COLUMN = "query"

RETURN_PROPERTIES = [
    "judgment_id",
    "full_text",
    "court_name",
    "decision",
    "docket_number",
    "judgment_date",
    "legal_bases",
    "judgment_date",
    "publication_date",
    "court_id",
    "department_id",
    "judgment_type",
    "excerpt",
    "content",
    "presiding_judge",
    "judges",
    "keywords",
    "department_name",
    "text_legal_bases",
    "thesis",
]

RETURN_METADATA = [
    "creation_time",
    "last_update_time",
    "distance",
    "certainty",
    "score",
    "explain_score",
    "is_consistent",
]

INCLUDE_VECTOR = True


async def bm25_judgment_search(
    db,
    queries: List[str],
    output_path,
    max_objects: int = MAX_OBJECTS,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Retrieve judgments related to Article 111 § 1 of Electoral Code using BM25 search.

    Args:
        db: WeaviateJudgmentsDatabase instance
        queries: List of search queries
        output_path: Path to save the search results
        max_objects: Maximum number of objects to retrieve
        overwrite: Whether to overwrite existing files

    Returns:
        DataFrame containing judgment data with query information
    """
    # Check if the file already exists
    if os.path.exists(output_path) and not overwrite:
        logger.info(f"Loading existing BM25 judgments from {output_path}")
        return pd.read_pickle(output_path)

    logger.info("Searching for Article 111 § 1 judgments using BM25 search")

    all_judgments = {}  # Dictionary with query as key
    query_stats = {}  # Track statistics for each query

    # Initialize progress bar with unknown total
    pbar = tqdm(desc="Retrieving judgments", unit="query")

    # Process each query
    for query in queries:
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
                include_vector=INCLUDE_VECTOR,
                return_metadata=RETURN_METADATA,
                return_properties=RETURN_PROPERTIES,
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


async def semantic_judgment_search(
    db,
    queries: List[str],
    output_path,
    max_objects: int = MAX_OBJECTS,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Perform semantic search for digital campaign materials in judgments.

    Args:
        db: WeaviateJudgmentsDatabase instance
        queries: List of search queries
        output_path: Path to save the search results
        max_objects: Maximum number of objects to retrieve
        overwrite: Whether to overwrite existing files

    Returns:
        DataFrame containing judgment data with query information
    """
    # Check if the file already exists
    if os.path.exists(output_path) and not overwrite:
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
                include_vector=INCLUDE_VECTOR,
                return_metadata=RETURN_METADATA,
                return_properties=RETURN_PROPERTIES,
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
