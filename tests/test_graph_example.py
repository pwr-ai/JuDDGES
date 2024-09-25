from pathlib import Path

import pytest
from polars.dependencies import subprocess

from juddges.settings import ROOT_PATH

USE_CASE_SCRIPTS = [
    ROOT_PATH / "scripts" / "examples" / "graph_networkx_use_case.py",
    ROOT_PATH / "scripts" / "examples" / "graph_pyg_use_case.py",
]


@pytest.mark.parametrize("script_path", USE_CASE_SCRIPTS)
def test_graph_use_case_runs_correctly(script_path: Path) -> None:
    assert script_path.exists()
    result = subprocess.run(["python", script_path], capture_output=True)
    assert result.returncode == 0
