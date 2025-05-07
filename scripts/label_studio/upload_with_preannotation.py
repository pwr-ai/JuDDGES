import hydra
import pandas as pd
from dotenv import load_dotenv
from hydra.utils import get_class
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from omegaconf import DictConfig
from tqdm import tqdm

from juddges.settings import ROOT_PATH
from label_studio_toolkit.api.client import LabelStudioClient

load_dotenv()


@hydra.main(
    version_base="1.3",
    config_path=str(ROOT_PATH / "configs"),
    config_name="upload_with_preannotation",
)
def main(cfg: DictConfig):
    set_llm_cache(SQLiteCache(database_path=cfg.model.request_cache_db))

    client = LabelStudioClient(
        base_url=cfg.ls_base_url, api_key=cfg.ls_api_key, project_name=cfg.project_name
    )
    with open(cfg.label_interface, "r") as f:
        label_interface = f.read()
    client.set_label_interface(label_interface)

    data = pd.read_parquet(cfg.data_path)

    for field, value in cfg.filter_fields.items():
        data = data[data[field] == value]

    client.create_tasks(data[[cfg.text_field]].to_dict(orient="records"))
    schema = get_class(cfg.annotation_schema)

    for task in tqdm(client.get_tasks()):
        rows = data[data[cfg.text_field] == task.data[cfg.text_field]]
        assert len(rows) == 1
        row = rows.iloc[0]
        prediction_model = schema(**row)
        prediction = client.create_prediction(
            prediction=prediction_model, model_version=cfg.model_version
        )
        client.push_prediction(task.id, prediction)


if __name__ == "__main__":
    main()
