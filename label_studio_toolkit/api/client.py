from label_studio_sdk import LabelStudio
from label_studio_sdk._extensions.pager_ext import SyncPagerExt
from label_studio_sdk.label_interface.objects import PredictionValue
from pydantic import BaseModel


class LabelStudioClient:
    def __init__(self, base_url: str, api_key: str, project_name: str) -> None:
        self.ls = LabelStudio(base_url=base_url, api_key=api_key)
        [self.project] = [p for p in self.ls.projects.list() if p.title == project_name]
        assert self.project.id is not None
        self.label_interface = self.ls.projects.get(id=self.project.id).get_label_interface()  # type: ignore

    def get_tasks(self) -> SyncPagerExt:
        result = self.ls.tasks.list(project=self.project.id, include=["id", "data"])  # type: ignore
        assert isinstance(result, SyncPagerExt)
        return result

    def create_prediction(self, prediction: BaseModel, model_version: str) -> PredictionValue:
        prediction_result = []
        for control_key, control_value in prediction.model_dump().items():
            if control_value is None:
                continue
            control = self.label_interface.get_control(control_key)
            prediction_result.append(control.label(control_value))

        prediction_value = PredictionValue(model_version=model_version, result=prediction_result)
        return prediction_value

    def push_prediction(self, task_id: int, prediction: PredictionValue) -> None:
        self.ls.predictions.create(task=task_id, **prediction.model_dump())  # type: ignore
