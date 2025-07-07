import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from juddges.embeddings.aggregate import mean_average_embeddings_and_save


@pytest.mark.parametrize("num_shards", [1, 2, 3, 4])
def test_mean_average_embeddings_and_save(num_shards: int):
    # Create toy data with 2 documents, each having 2 chunks
    # Document 1: chunks of length 2 and 3
    # Document 2: chunks of length 1 and 2
    data = {
        "judgment_id": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc4", "doc4"],
        "chunk_len": [2, 3, 1, 2, 1, 1, 2],
        "embedding": [
            np.array([1.0, 2.0]),  # doc1, chunk1
            np.array([3.0, 4.0]),  # doc1, chunk2
            np.array([5.0, 6.0]),  # doc2, chunk1
            np.array([7.0, 8.0]),  # doc2, chunk2
            np.array([9.0, 10.0]),  # doc3, chunk1
            np.array([11.0, 12.0]),  # doc4, chunk1
            np.array([13.0, 14.0]),  # doc4, chunk2
        ],
    }

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save source data
        source_df = pl.DataFrame(data)
        source_path = temp_path / "source.parquet"
        source_df.write_parquet(source_path)

        # Create output directory
        output_dir = temp_path / "output"
        output_dir.mkdir()

        # Run the aggregation function
        mean_average_embeddings_and_save(
            df=pl.scan_parquet(source_path),
            num_shards=num_shards,
            output_dir=output_dir,
            id_col="judgment_id",
            embedding_col="embedding",
        )

        # Load and verify results
        assert len(list(output_dir.glob("*.parquet"))) == num_shards
        result_df = pl.read_parquet(output_dir)

        # Expected results:
        # doc1: np.array([1.0, 2.0]) * 2/5 + np.array([3.0, 4.0]) * 3/5 = np.array([2.2, 2.8])
        # doc2: np.array([5.0, 6.0]) * 1/3 + np.array([7.0, 8.0]) * 2/3 = np.array([6.33..., 7.33...])
        expected_data = {
            "judgment_id": ["doc1", "doc2", "doc3", "doc4"],
            "embedding": [
                [2.2, 3.2],
                [6.33, 7.33],
                [9.0, 10.0],
                [12.33, 13.33],
            ],
        }
        expected_df = pl.DataFrame(expected_data)

        # Compare results (using approximate equality for floating point numbers)
        assert_frame_equal(
            expected_df.sort("judgment_id"),
            result_df.sort("judgment_id"),
            atol=1e-2,
        )
