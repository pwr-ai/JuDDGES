import hydra
from dotenv import load_dotenv
from hydra.utils import get_class
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_community.callbacks import get_openai_callback
from omegaconf import DictConfig
from tqdm import tqdm

from juddges.settings import ROOT_PATH
from label_studio_toolkit.annotator import LangChainOpenAIAnnotator
from label_studio_toolkit.api.client import LabelStudioClient

load_dotenv()


@hydra.main(
    version_base="1.3",
    config_path=str(ROOT_PATH / "configs"),
    config_name="preannotate_label_studio",
)
def main(cfg: DictConfig):
    set_llm_cache(SQLiteCache(database_path=cfg.model.request_cache_db))

    client = LabelStudioClient(
        base_url=cfg.ls_base_url, api_key=cfg.ls_api_key, project_name=cfg.project_name
    )

    schema = get_class(cfg.annotation_schema)
    annotator = LangChainOpenAIAnnotator(cfg.model.name, cfg.prompt.template, schema)

    with get_openai_callback() as cb:
        for i, task in tqdm(enumerate(client.get_tasks())):
            if i >= 10:
                break
            annotation = annotator.annotate(task.data[cfg.text_field])
            prediction = client.create_prediction(
                prediction=annotation, model_version=cfg.model_version
            )
            client.push_prediction(task.id, prediction)
        print(cb)


if __name__ == "__main__":
    main()
