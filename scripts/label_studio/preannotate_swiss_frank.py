import json

import typer
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from label_studio_toolkit.annotator import LLMAnnotator
from label_studio_toolkit.api.client import LabelStudioClient
from label_studio_toolkit.schemas.swiss_frank import SwissFrancJudgmentAnnotation

LABEL_STUDIO_PROJECT_TITLE = "Juddges Project: Swiss Frank"

load_dotenv()


def main(
    ls_base_url: str = typer.Option(..., envvar="LABEL_STUDIO_BASE_URL"),
    ls_api_key: str = typer.Option(..., envvar="LABEL_STUDIO_API_KEY"),
    project_name: str = typer.Option(...),
    model_version: str = typer.Option(...),
):
    client = LabelStudioClient(base_url=ls_base_url, api_key=ls_api_key, project_name=project_name)

    # [project] = [p for p in ls.projects.list() if p.title == LABEL_STUDIO_PROJECT_TITLE]

    # label_interface = ls.projects.get(id=project.id).get_label_interface()

    # controls_keys = list(label_interface._controls.keys())

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    annotator = LLMAnnotator(llm, SwissFrancJudgmentAnnotation)

    input_ = "Sąd rozpoznaje sprawę w apelacji warszawskiej. Jest to sprawa dotycząca umowy kredytowej we frankach szwajcarskich. Powód podał podstawę prawną z art. 385(1) Kodeksu cywilnego. Rozpoznano powództwo w całości. Zwrócono koszty procesu kredytobiorcy."

    print(json.dumps(annotation.__dict__, indent=4))

    for task in client.get_tasks():
        annotation = annotator.annotate(input_)

        client.create_prediction(
            task_id=task.id, prediction=annotation, model_version=model_version
        )
        break

    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # annotator = LLMAnnotator(llm, SwissFrancJudgmentAnnotation)

    # input_ = "Sąd rozpoznaje sprawę w apelacji warszawskiej. Jest to sprawa dotycząca umowy kredytowej we frankach szwajcarskich. Powód podał podstawę prawną z art. 385(1) Kodeksu cywilnego. Rozpoznano powództwo w całości. Zwrócono koszty procesu kredytobiorcy."

    # annotation = annotator.annotate(input_)

    # print(json.dumps(annotation.__dict__, indent=4))


if __name__ == "__main__":
    typer.run(main)
