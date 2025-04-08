from pathlib import Path

import mlflow
import tiktoken
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# get root path as ROOT_PATH as pathlib objects
ROOT_PATH = Path(__file__).resolve().parent.parent

LOGS_PATH = ROOT_PATH / "logs"

DATA_PATH = ROOT_PATH / "data"
CONFIG_PATH = ROOT_PATH / "configs"

SAMPLE_DATA_PATH = DATA_PATH / "sample_data"
FRANKOWICZE_DATA_PATH = DATA_PATH / "analysis" / "sprawy_frankowe"

PL_JUDGEMENTS_PATH = DATA_PATH / "datasets" / "pl"
PL_COURT_DEP_ID_2_NAME = PL_JUDGEMENTS_PATH / "court_id_2_name.csv"
PL_JUDGEMENTS_PATH_RAW = PL_JUDGEMENTS_PATH / "raw"
PL_JUDGEMENTS_PATH_TEXTS = PL_JUDGEMENTS_PATH / "text"
PL_JUDGEMENTS_PATH_INSTRUCT = PL_JUDGEMENTS_PATH / "instruct"

MLFLOW_EXP_NAME = "Juddges-Information-Extraction"

TEXT_EMBEDDING_MODEL = "sdadas/mmlw-roberta-large"

# NSA
NSA_DATA_PATH = DATA_PATH / "datasets" / "nsa"


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

LOCAL_POSTGRES = "postgresql+psycopg2://llm:llm@postgres-juddges:5432/llm"


def get_sqlalchemy_engine() -> Engine:
    return create_engine(
        LOCAL_POSTGRES,
        pool_size=10,
        max_overflow=2,
        pool_recycle=300,
        pool_pre_ping=True,
        pool_use_lifo=True,
    )


def prepare_langchain_cache() -> None:
    import langchain
    from langchain.cache import SQLAlchemyMd5Cache

    langchain.llm_cache = SQLAlchemyMd5Cache(get_sqlalchemy_engine())


def prepare_mlflow(experiment_name: str = MLFLOW_EXP_NAME, url: str = "localhost") -> None:
    mlflow.set_tracking_uri(url)
    mlflow.set_experiment(experiment_name)
