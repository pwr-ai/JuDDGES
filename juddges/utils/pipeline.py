import asyncio
from typing import Any, Type

from prefect import Flow, get_client
from prefect.client.schemas.filters import FlowRunFilter
from prefect.client.schemas.sorting import FlowRunSort
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


def get_recent_successful_flow_date() -> str | None:
    async def _get_recent_successful_flow_date() -> str | None:
        async with get_client() as client:
            flow_runs = await client.read_flow_runs(
                flow_run_filter=FlowRunFilter(
                    name={"type": ["update_pl_court_data"]},
                    state={"type": {"any_": ["COMPLETED"]}},
                ),
                sort=FlowRunSort.END_TIME_DESC,
                limit=1,
            )
        if len(flow_runs) == 0:
            return None
        return flow_runs[0].start_time.to_date_string()

    return asyncio.run(_get_recent_successful_flow_date())
