from label_studio_sdk import LabelStudio
from label_studio_sdk.label_interface.objects import PredictionValue
from pydantic import BaseModel


class LabelStudioClient:
    def __init__(self, base_url: str, api_key: str, project_name: str):
        self.ls = LabelStudio(base_url=base_url, api_key=api_key)
        [self.project] = [p for p in self.ls.projects.list() if p.title == project_name]
        self.label_interface = self.ls.projects.get(id=self.project.id).get_label_interface()

    def get_tasks(self):
        return self.ls.tasks.list(project=self.project.id, include=["id", "data"])

    def create_prediction(self, task_id: str, prediction: BaseModel, model_version: str) -> None:
        prediction_result = []
        for control_key, control_value in prediction.model_dump().items():
            control = self.label_interface.get_control(control_key)
            prediction_result.append(control.label(control_value))

        prediction_value = PredictionValue(model_version=model_version, result=prediction_result)
        # self.ls.predictions.create(task=task_id, **prediction_value.model_dump())
        pass
