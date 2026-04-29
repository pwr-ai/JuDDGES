from label_studio_sdk import LabelStudio, Project
from label_studio_sdk._extensions.pager_ext import SyncPagerExt
from label_studio_sdk.label_interface.objects import PredictionValue
from pydantic import BaseModel
from tqdm import tqdm


def sanitize_prediction_data(data):
    """Recursively sanitize data to remove null bytes and problematic characters."""
    if isinstance(data, str):
        # Remove null bytes and other control characters that can cause JSON/DB issues
        sanitized = data.replace("\x00", "")  # Remove null bytes
        sanitized = sanitized.replace("\ufffd", "")  # Remove replacement characters
        # Remove other problematic control characters except common whitespace
        sanitized = "".join(char for char in sanitized if ord(char) >= 32 or char in "\t\n\r")
        return sanitized
    elif isinstance(data, list):
        return [sanitize_prediction_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: sanitize_prediction_data(value) for key, value in data.items()}
    else:
        return data


class LabelStudioClient:
    def __init__(self, base_url: str, api_key: str, project_name: str) -> None:
        self.ls = LabelStudio(base_url=base_url, api_key=api_key)
        self.project = self._get_project(project_name)
        if self.project is None:
            self.ls.projects.create(title=project_name)
            self.project = self._get_project(project_name)
            assert self.project is not None

        assert self.project.id is not None
        self.label_interface = self.ls.projects.get(id=self.project.id).get_label_interface()  # type: ignore

    def create_tasks(self, tasks: list[dict]) -> None:
        for task in tqdm(tasks):
            self.ls.tasks.create(data=task, project=self.project.id)

    def set_label_interface(self, label_interface: str) -> None:
        self.ls.projects.update(id=self.project.id, label_config=label_interface)  # type: ignore
        self.label_interface = self.ls.projects.get(id=self.project.id).get_label_interface()

    def get_tasks(self) -> SyncPagerExt:
        result = self.ls.tasks.list(project=self.project.id, include=["id", "data"])  # type: ignore
        assert isinstance(result, SyncPagerExt)
        return result

    def create_prediction(self, prediction: BaseModel, model_version: str) -> PredictionValue:
        prediction_result = []
        # Sanitize the prediction data before processing
        prediction_data = sanitize_prediction_data(prediction.model_dump())

        for control_key, control_value in prediction_data.items():
            if control_value is None:
                continue
            control = self.label_interface.get_control(control_key)
            prediction_result.append(control.label(control_value))

        prediction_value = PredictionValue(model_version=model_version, result=prediction_result)
        return prediction_value

    def push_prediction(self, task_id: int, prediction: PredictionValue) -> None:
        # Sanitize the prediction data before sending to Label Studio
        prediction_dict = sanitize_prediction_data(prediction.model_dump())
        self.ls.predictions.create(task=task_id, **prediction_dict)  # type: ignore

    def _get_project(self, project_name: str) -> Project | None:
        projects = [p for p in self.ls.projects.list() if p.title == project_name]
        if len(projects) == 0:
            return None
        assert len(projects) == 1
        return projects[0]
