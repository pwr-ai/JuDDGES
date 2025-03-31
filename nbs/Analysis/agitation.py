#!/usr/bin/env python
# coding: utf-8

"""
Script to analyze electoral judgments focusing on digital campaign materials
under Article 111 § 1 of the Electoral Code.
"""
import asyncio
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from settings import ARTICLE_111_DATA_PATH, prepare_langchain_cache
from tqdm import tqdm

from juddges.data.weaviate_db import WeaviateJudgmentsDatabase
from juddges.llms import GPT_4o
from juddges.prompts.information_extraction import prepare_information_extraction_chain
from juddges.prompts.schemas.agitation import AGITATION_SCHEMA

BATCH_SIZE = 100
LLM_BATCH_SIZE = 10
MAX_OBJECTS = 10000
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


async def get_all_art111_judgments(
    db: WeaviateJudgmentsDatabase, queries: List[str], max_objects: int = MAX_OBJECTS
) -> pd.DataFrame:
    """
    Retrieve all judgments related to Article 111 § 1 of Electoral Code using BM25 search.

    Args:
        db: WeaviateJudgmentsDatabase instance

    Returns:
        List of judgment dictionaries
    """
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
            judgment_dict.pop("uuid")

            data.append(
                {
                    **judgment_dict,
                    QUERY_COLUMN: query,
                }
            )

    judgments_df = pd.DataFrame(data)
    judgments_df.to_pickle(ARTICLE_111_DATA_PATH / "all_judgments_bm25.pkl")

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


async def semantic_search_digital_materials(
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
            vector = await db.embeddings.embed_query(query)

            # Query the judgments collection using vector search
            search = db.judgments_collection.query.near_vector(
                vector=vector,
                limit=BATCH_SIZE,
                certainty=0.5,
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
                query_scores.append(judgment.metadata.score)

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
            judgment_dict.pop("uuid")

            data.append(
                {
                    **judgment_dict,
                    QUERY_COLUMN: query,
                }
            )

    judgments_df = pd.DataFrame(data)
    judgments_df.to_pickle(ARTICLE_111_DATA_PATH / "all_judgments_vector.pkl")

    # Log statistics
    total_judgments = len(judgments_df)
    unique_judgments = judgments_df["judgment_id"].nunique()
    logger.info(f"Total judgments retrieved: {total_judgments}")
    logger.info(f"Unique judgments: {unique_judgments}")
    logger.info(
        f"Duplicate rate: {(total_judgments - unique_judgments) / total_judgments * 100:.2f}%"
    )

    return judgments_df


async def main():

    prepare_langchain_cache()

    """Main execution function"""
    async with WeaviateJudgmentsDatabase() as db:
        # Get all Article 111 § 1 judgments
        bm25_judgments_df = await get_all_art111_judgments(
            db, AGITATION_QUERIES["bm25_queries"]
        )

        # Get all Article 111 § 1 judgments
        vector_judgments_df = await get_all_art111_judgments(
            db, AGITATION_QUERIES["vector_queries"]
        )

        df = pd.concat([bm25_judgments_df, vector_judgments_df])
        df.to_pickle(ARTICLE_111_DATA_PATH / "all_judgments.pkl")
        # Choose the model for information extraction
        MODEL_NAME = GPT_4o
        LANGUAGE = "polish"  # Polish language for judgments

        logger.info(f"Starting information extraction using {MODEL_NAME} model")

        # Create the extraction chain
        extraction_chain = prepare_information_extraction_chain(model_name=MODEL_NAME)

        # Process judgments in batches to avoid memory issues
        batch_size = LLM_BATCH_SIZE
        all_judgments_data = []

        # Create a copy of the dataframe to add extraction results
        judgments_with_extraction = df.copy()
        judgments_with_extraction["extracted_info"] = None

        # Use tqdm to show progress
        for i in range(0, len(df), batch_size):
            batch = df[i : i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}"
            )

            batch_inputs = []
            batch_indices = []
            for idx, judgment in enumerate(batch):
                content = judgment.properties.get("full_text", "")
                if not content:
                    continue

                batch_inputs.append(
                    {
                        "TEXT": content[
                            :MAX_TEXT_LENGTH
                        ],  # Limit text length to avoid token limits
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

                # Combine extraction results with judgment metadata
                for result, judgment, idx in zip(batch_results, batch, batch_indices):
                    if result:
                        # Add metadata to the result
                        result.update(
                            {
                                "judgment_id": judgment.properties.get("judgment_id"),
                                "court_name": judgment.properties.get("court_name"),
                                "docket_number": judgment.properties.get(
                                    "docket_number"
                                ),
                                "judgment_date": judgment.properties.get(
                                    "judgment_date"
                                ),
                            }
                        )

                        # Enhance extraction with regex pattern matching for digital platforms
                        content = judgment.properties.get(
                            "content", ""
                        ) or judgment.properties.get("full_text", "")

                        all_judgments_data.append(result)

                        # Add extraction result to the dataframe
                        judgments_with_extraction.loc[idx, "extracted_info"] = result
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue

        # Save all extracted data to CSV
        if all_judgments_data:
            extracted_df = pd.DataFrame(all_judgments_data)
            output_path = ARTICLE_111_DATA_PATH / "extracted_agitation_data.csv"
            extracted_df.to_csv(output_path, index=False)
            logger.info(
                f"Saved {len(all_judgments_data)} processed judgments to {output_path}"
            )

            # Save the enhanced dataframe with extraction results
            enhanced_output_path = (
                ARTICLE_111_DATA_PATH / "judgments_with_extraction.pkl"
            )
            judgments_with_extraction.to_pickle(enhanced_output_path)
            logger.info(
                f"Saved judgments with extraction results to {enhanced_output_path}"
            )

        else:
            logger.warning("No structured data was extracted from judgments")

        # Calculate platform statistics
        all_digital_judgments = []
        platform_counts = {}

        for platform, judgments in digital_materials.items():
            platform_counts[platform] = len(judgments)
            # Collect unique judgments across all platforms
            for judgment in judgments:
                judgment_id = judgment.properties.get("judgment_id")
                if judgment_id not in [
                    j.properties.get("judgment_id") for j in all_digital_judgments
                ]:
                    judgment.properties["digital_platform"] = platform
                    all_digital_judgments.append(judgment)


if __name__ == "__main__":
    asyncio.run(main())
