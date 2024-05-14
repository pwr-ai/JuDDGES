import os
from pathlib import Path

import mlflow
import tiktoken
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# get root path as ROOT_PATH as pathlib objects
ROOT_PATH = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_PATH / "data"
CACHE_DIR = DATA_PATH / "cache"

SAMPLE_DATA_PATH = DATA_PATH / "sample_data"

PL_JUDGEMENTS_PATH = DATA_PATH / "datasets" / "pl"
PL_JUDGEMENTS_PATH_RAW = PL_JUDGEMENTS_PATH / "raw"
PL_JUDGEMENTS_PATH_TEXTS = PL_JUDGEMENTS_PATH / "text"

PL_JUDGEMENTS_SYNTH_PATH = PL_JUDGEMENTS_PATH / "synthetic"
PL_JUDGEMENTS_SYNTH_QA_PATH = PL_JUDGEMENTS_SYNTH_PATH / "qa"

MLFLOW_EXP_NAME = "Juddges-Information-Extraction"


def num_tokens_from_string(
    string: str,  # The string to count tokens for
    encoding_name: str = "cl100k_base",  # gpt-4, gpt-3.5-turbo, text-embedding-ada-002
) -> int:  # The number of tokens in the string
    """
    Returns the number of tokens in a text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


LLM_TO_PRICE_INPUT = {
    "gpt-4-1106-preview": 0.01 / 1000,
    "gpt-4-0125-preview": 0.01 / 1000,
    "gpt-3.5-turbo-1106": 0.001 / 1000,
}

LLM_TO_PRICE_COMPLETION = {
    "gpt-4-1106-preview": 0.03 / 1000,
    "gpt-4-0125-preview": 0.03 / 1000,
    "gpt-3.5-turbo-1106": 0.002 / 1000,
}


def get_local_postgres_url(
    serice_name: str = "postgres-juddges",
    port: int = 5432,
) -> str:
    return "postgresql+psycopg2://{user}:{password}@{serice_name}:{port}/{db_name}".format(
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        serice_name=serice_name,
        port=port,
        db_name=os.environ["POSTGRES_DB"],
    )


def get_sqlalchemy_engine() -> Engine:
    return create_engine(
        url=get_local_postgres_url(),
        pool_size=10,
        max_overflow=2,
        pool_recycle=300,
        pool_pre_ping=True,
        pool_use_lifo=True,
    )


def prepare_langchain_cache() -> None:
    from langchain.cache import SQLAlchemyMd5Cache
    from langchain.globals import set_llm_cache

    set_llm_cache(SQLAlchemyMd5Cache(get_sqlalchemy_engine()))


def prepare_mlflow(experiment_name: str = MLFLOW_EXP_NAME, url="postgres-juddges") -> None:
    mlflow.set_tracking_uri(url)
    mlflow.set_experiment(experiment_name)
