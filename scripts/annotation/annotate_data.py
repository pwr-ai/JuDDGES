import asyncio
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from hydra.utils import get_class
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_community.callbacks import get_openai_callback
from omegaconf import DictConfig
from tqdm.asyncio import tqdm_asyncio

from juddges.settings import ROOT_PATH
from label_studio_toolkit.annotator import LangChainOpenAIAnnotator

load_dotenv()


@hydra.main(
    version_base="1.3",
    config_path=str(ROOT_PATH / "configs"),
    config_name="annotate_data",
)
def main(cfg: DictConfig):
    set_llm_cache(SQLiteCache(database_path=cfg.model.request_cache_db))
    dataset = load_dataset(cfg)

    schema = get_class(cfg.annotation_schema)
    schema_json = schema.model_json_schema()
    annotator = LangChainOpenAIAnnotator(cfg.model.name, cfg.prompt.template, schema)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    with get_openai_callback() as cb:
        for name, df in dataset.items():
            if name == "test" and cfg.skip_test:
                continue
            if name == "train" and cfg.skip_train:
                continue
            datapoints = []
            annotations = asyncio.run(async_annotate(annotator, df, cfg))
            for (i, row), annotation in zip(df.iterrows(), annotations):
                datapoint = {
                    **annotation.model_dump(mode="json"),
                    "text": row[cfg.text_field],
                    "LLM": cfg.model_version,
                    "schema": schema_json,
                    "language": "polish",
                }
                datapoints.append(datapoint)

            annotations_df = pd.DataFrame(datapoints)
            annotations_df.to_parquet(f"{cfg.output_dir}/{name}_annotations.parquet")
        print(cb)


async def async_annotate(
    annotator: LangChainOpenAIAnnotator, df: pd.DataFrame, cfg: DictConfig
) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(15)

    async def annotate_row(row: pd.Series) -> dict:
        async with semaphore:
            return await annotator.async_annotate(row[cfg.text_field])

    return await tqdm_asyncio.gather(*[annotate_row(row) for _, row in df.iterrows()])


def load_dataset(cfg: DictConfig) -> dict[str, pd.DataFrame]:
    if cfg.dataset.type == "parquet":
        train = pd.read_parquet(cfg.dataset.train.path)
        test = pd.read_parquet(cfg.dataset.test.path)
        return {"train": train, "test": test}
    elif cfg.dataset.type == "hf":
        dataset = load_dataset(cfg.dataset)
        return {name: dataset[name].to_pandas() for name in dataset.keys()}
    else:
        raise ValueError(f"Invalid dataset type: {cfg.dataset.type}")



if __name__ == "__main__":
    main()
