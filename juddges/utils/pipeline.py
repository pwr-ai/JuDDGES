from typing import Any, Type

from prefect import Flow
from prefect.states import Failed


class RetryOnException:
    def __init__(
        self, exception: Type[Exception] | list[Type[Exception]] | tuple[Type[Exception], ...]
    ) -> None:
        if isinstance(exception, list):
            self.exceptions = tuple(exception)
        elif isinstance(exception, tuple):
            self.exceptions = exception
        elif issubclass(exception, Exception):
            self.exceptions = (exception,)
        else:
            raise ValueError(f"Invalid exception type: {type(exception)}")

    def __call__(self, task: Flow, task_run: Any, state: Failed):
        """Determine if the task should be retried based on the exception raised."""
        try:
            # Attempt to retrieve the task's result
            state.result()
        except self.exceptions:
            # Retry on specified exceptions
            return True
        except Exception:
            # Do not retry on other exceptions
            return False
        return False
