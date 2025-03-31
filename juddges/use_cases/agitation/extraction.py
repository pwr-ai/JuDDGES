"""
Functions for extracting structured information from judgment texts.
"""

import pandas as pd
from loguru import logger

from juddges.llms import GPT_4o
from juddges.prompts.information_extraction import prepare_information_extraction_chain

# Constants
LLM_BATCH_SIZE = 10
MAX_TEXT_LENGTH = 150000

# Choose the model for information extraction
MODEL_NAME = GPT_4o
LANGUAGE = "polish"  # Polish language for judgments


async def extract_judgment_information(df, schema, batch_size=LLM_BATCH_SIZE):
    """
    Extract structured information from judgment texts using LLM.

    Args:
        df: DataFrame containing judgment data
        schema: Schema for information extraction
        batch_size: Number of judgments to process in each batch

    Returns:
        Tuple of (DataFrame with extraction results, list of extracted data)
    """

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
            content = judgment.properties["full_text"]
            if not content:
                continue

            batch_inputs.append(
                {
                    "TEXT": content[:MAX_TEXT_LENGTH],
                    "SCHEMA": schema,
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
