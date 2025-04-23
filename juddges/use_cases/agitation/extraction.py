"""
Functions for extracting structured information from judgment texts.
"""

import pandas as pd
from loguru import logger
from more_itertools import chunked
from tqdm import tqdm

from juddges.llms import GPT_4o
from juddges.prompts.information_extraction import prepare_information_extraction_chain

# Constants
LLM_BATCH_SIZE = 10
MAX_TEXT_LENGTH = 150000

# Choose the model for information extraction
MODEL_NAME = GPT_4o
LANGUAGE = "polish"  # Polish language for judgments


async def extract_judgment_information(
    df: pd.DataFrame, schema: dict, batch_size: int = LLM_BATCH_SIZE
) -> pd.DataFrame:
    """
    Extract structured information from judgment texts using LLM.

    Args:
        df: DataFrame containing judgment data
        schema: Schema for information extraction
        batch_size: Number of judgments to process in each batch

    Returns:
        DataFrame with extracted information
    """

    logger.info(
        f"Starting information extraction using {MODEL_NAME} model on {len(df)} judgments"
    )

    # Create the extraction chain
    extraction_chain = prepare_information_extraction_chain(model_name=MODEL_NAME)

    # Process judgments in batches to avoid memory issues
    extracted_data = []

    judgments_properties = df.properties.tolist()

    # Use tqdm to show progress
    for batch in tqdm(
        chunked(judgments_properties, batch_size), total=len(df) / batch_size
    ):
        extracted_data += extraction_chain.batch(
            [
                {
                    "TEXT": prop["full_text"][:MAX_TEXT_LENGTH],
                    "SCHEMA": schema,
                    "LANGUAGE": LANGUAGE,
                }
                for prop in batch
            ]
        )

    df["extracted_informations"] = extracted_data

    return df
