import json
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock

import pytest
from langchain_openai import ChatOpenAI

from juddges.llm_as_judge.base import EvalResults
from juddges.llm_as_judge.judge import StructuredOutputJudge
from juddges.utils.misc import save_yaml


@pytest.fixture
def setup_environment() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        predictions_dir = Path(temp_dir)

        config_dict = {
            "ie_schema": {
                "key1": {
                    "type": "string",
                    "description": "Test field",
                    "required": True,
                },
                "key2": {
                    "type": "string",
                    "description": "Another test field",
                    "required": False,
                },
            }
        }
        config_file = predictions_dir / "config.yaml"
        save_yaml(config_dict, config_file)

        yield predictions_dir


@pytest.fixture
def valid_predictions_data() -> list[dict[str, str]]:
    return [
        {
            "answer": '{"key1": "value1", "key2": "value2"}',
            "gold": '{"key1": "gold1", "key2": "gold2"}',
        },
        {
            "answer": '{"key1": "value3", "key2": "value4"}',
            "gold": '{"key1": "gold3", "key2": "gold4"}',
        },
    ]


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock(spec=ChatOpenAI)
    client.model_name = "test-model"
    client.with_structured_output.return_value = client
    client.ainvoke = AsyncMock()

    return client


@pytest.mark.asyncio
async def test_successful_evaluation(
    setup_environment: Path,
    valid_predictions_data: list[dict[str, str]],
    mock_client: AsyncMock,
) -> None:
    predictions_dir = setup_environment
    predictions_file = predictions_dir / "predictions.json"

    with open(predictions_file, "w") as f:
        json.dump(valid_predictions_data, f)

    mock_client.ainvoke.return_value = {"key1": {"score": 0.8}, "key2": {"score": 0.9}}

    judge = StructuredOutputJudge(
        client=mock_client,
        predictions_dir=predictions_dir,
        verbose=False,
    )
    results = await judge.evaluate()

    assert isinstance(results, EvalResults)
    assert len(results.results) == 2
    assert all(result.status == "success" for result in results.results)
    assert all(result.error is None for result in results.results)
    assert mock_client.ainvoke.call_count == 2


@pytest.mark.asyncio
async def test_predictions_with_missing_keys(
    setup_environment: Path, mock_client: AsyncMock
) -> None:
    predictions_dir = setup_environment
    predictions_file = predictions_dir / "predictions.json"

    predictions_with_missing_keys = [
        {"answer": '{"key1": "value1"}', "gold": '{"key1": "gold1", "key2": "gold2"}'}
    ]

    with open(predictions_file, "w") as f:
        json.dump(predictions_with_missing_keys, f)

    mock_client.ainvoke.return_value = {"key1": {"score": 0.8}, "key2": {"score": 0.0}}

    judge = StructuredOutputJudge(client=mock_client, predictions_dir=predictions_dir)
    results = await judge.evaluate()

    assert len(results.results) == 1
    assert results.results[0].status == "success"
    assert "key2" in results.results[0].missing_keys


@pytest.mark.asyncio
async def test_predictions_with_extra_keys(setup_environment: Path, mock_client: AsyncMock) -> None:
    predictions_dir = setup_environment
    predictions_file = predictions_dir / "predictions.json"

    predictions_with_extra_keys = [
        {
            "answer": '{"key1": "value1", "key2": "value2", "key3": "value3"}',
            "gold": '{"key1": "gold1", "key2": "gold2"}',
        }
    ]

    with open(predictions_file, "w") as f:
        json.dump(predictions_with_extra_keys, f)

    mock_client.ainvoke.return_value = {"key1": {"score": 0.8}, "key2": {"score": 0.9}}

    judge = StructuredOutputJudge(client=mock_client, predictions_dir=predictions_dir)
    results = await judge.evaluate()

    assert len(results.results) == 1
    assert results.results[0].status == "success"
    assert "key3" in results.results[0].extra_keys


@pytest.mark.asyncio
async def test_parsing_failure_invalid_json(
    setup_environment: Path,
    mock_client: AsyncMock,
) -> None:
    predictions_dir = setup_environment
    predictions_file = predictions_dir / "predictions.json"

    invalid_predictions = [
        {"answer": '{"key1": "value1",}', "gold": '{"key1": "gold1", "key2": "gold2"}'},
        {"answer": '{"key1": "value2"}', "gold": '{"key1": "gold2"}'},
    ]

    with open(predictions_file, "w") as f:
        json.dump(invalid_predictions, f)

    mock_client.ainvoke.return_value = {"key1": {"score": 0.8}, "key2": {"score": 0.0}}

    judge = StructuredOutputJudge(client=mock_client, predictions_dir=predictions_dir)
    results = await judge.evaluate()

    assert mock_client.ainvoke.call_count == 1
    assert len(results.results) == 2
    assert results.results[0].status == "parsing_error"
    assert results.results[0].error is not None
    assert results.results[1].status == "success"


@pytest.mark.asyncio
async def test_parsing_failure_missing_required_keys(
    setup_environment: Path, mock_client: AsyncMock
) -> None:
    predictions_dir = setup_environment
    predictions_file = predictions_dir / "predictions.json"

    invalid_file_content = '[{"missing_answer_key": "test", "gold": "{}"}]'

    with open(predictions_file, "w") as f:
        f.write(invalid_file_content)

    judge = StructuredOutputJudge(client=mock_client, predictions_dir=predictions_dir)

    with pytest.raises(ValueError, match="Predictions must contain 'answer' and 'gold' keys"):
        await judge.evaluate()


@pytest.mark.asyncio
async def test_api_failure_all_requests(
    setup_environment: Path, valid_predictions_data: list[dict[str, str]], mock_client: AsyncMock
) -> None:
    predictions_dir = setup_environment
    predictions_file = predictions_dir / "predictions.json"

    with open(predictions_file, "w") as f:
        json.dump(valid_predictions_data, f)

    mock_client.ainvoke.side_effect = Exception("API completely down")

    judge = StructuredOutputJudge(client=mock_client, predictions_dir=predictions_dir)
    results = await judge.evaluate()

    assert len(results.results) == 2
    assert all(result.status == "judge_error" for result in results.results)
    assert all("API completely down" in result.error for result in results.results)


@pytest.mark.asyncio
async def test_empty_predictions_file(setup_environment: Path, mock_client: AsyncMock) -> None:
    predictions_dir = setup_environment
    predictions_file = predictions_dir / "predictions.json"

    with open(predictions_file, "w") as f:
        json.dump([], f)

    judge = StructuredOutputJudge(client=mock_client, predictions_dir=predictions_dir)
    results = await judge.evaluate()

    assert len(results.results) == 0
    assert mock_client.ainvoke.call_count == 0


@pytest.mark.asyncio
async def test_concurrency_limit_respected(setup_environment: Path, mock_client: AsyncMock) -> None:
    predictions_dir = setup_environment
    predictions_file = predictions_dir / "predictions.json"

    large_predictions = [
        {
            "answer": f'{{"key1": "value{i}", "key2": "value{i}"}}',
            "gold": f'{{"key1": "gold{i}", "key2": "gold{i}"}}',
        }
        for i in range(10)
    ]

    with open(predictions_file, "w") as f:
        json.dump(large_predictions, f)

    mock_client.ainvoke.return_value = {"key1": {"score": 0.8}, "key2": {"score": 0.9}}

    judge = StructuredOutputJudge(
        client=mock_client, predictions_dir=predictions_dir, max_concurrent_calls=3
    )

    assert judge.max_concurrent_calls == 3
    assert judge.semaphore._value == 3

    results = await judge.evaluate()
    assert len(results.results) == 10
    assert all(result.status == "success" for result in results.results)


@pytest.mark.asyncio
async def test_zero_scores_fallback(
    setup_environment: Path, valid_predictions_data: list[dict[str, str]], mock_client: AsyncMock
) -> None:
    predictions_dir = setup_environment
    predictions_file = predictions_dir / "predictions.json"

    with open(predictions_file, "w") as f:
        json.dump(valid_predictions_data, f)

    mock_client.ainvoke.side_effect = Exception("API error")

    judge = StructuredOutputJudge(client=mock_client, predictions_dir=predictions_dir)
    results = await judge.evaluate()

    zero_scores = judge.get_zero_scores()
    for result in results.results:
        assert result.status == "judge_error"
        assert result.result == zero_scores


@pytest.mark.asyncio
async def test_custom_max_concurrent_calls(setup_environment: Path, mock_client: AsyncMock) -> None:
    predictions_dir = setup_environment
    predictions_file = predictions_dir / "predictions.json"

    with open(predictions_file, "w") as f:
        json.dump([], f)

    custom_limit = 10
    judge = StructuredOutputJudge(
        client=mock_client, predictions_dir=predictions_dir, max_concurrent_calls=custom_limit
    )

    assert judge.max_concurrent_calls == custom_limit
    assert judge.semaphore._value == custom_limit
