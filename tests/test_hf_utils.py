import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from juddges.utils.hf import get_parquet_num_rows


def test_get_parquet_num_rows():
    """Test get_parquet_num_rows function with synthetic Parquet files."""
    file_sizes = [10, 20, 30]
    total_rows = sum(file_sizes)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for i, size in enumerate(file_sizes):
            df = pd.DataFrame({"id": range(size), "value": np.random.rand(size)})

            file_path = temp_path / f"test_file_{i}.parquet"
            df.to_parquet(file_path)

        result = get_parquet_num_rows(temp_path)
        assert result == total_rows, f"Expected {total_rows} rows, but found {result}"


def test_get_parquet_num_rows_empty_dir():
    """Test get_parquet_num_rows with an empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError):
            get_parquet_num_rows(temp_dir)


def test_get_parquet_num_rows_single_file():
    """Test get_parquet_num_rows with a single parquet file."""
    row_count = 15

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        df = pd.DataFrame({"id": range(row_count), "text": [f"text_{i}" for i in range(row_count)]})

        file_path = temp_path / "single_file.parquet"
        df.to_parquet(file_path)

        result = get_parquet_num_rows(temp_path)
        assert result == row_count

        result = get_parquet_num_rows(file_path)
        assert result == row_count
