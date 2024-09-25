from juddges.settings import NSA_DATA_PATH

import polars as pl
import pytest

PATH = NSA_DATA_PATH / "pages"


@pytest.fixture(scope="module")
def lf():
    return pl.scan_parquet(NSA_DATA_PATH / "pages" / "pages_chunk_*.parquet")


@pytest.mark.skip(reason="This test requires a lot of RAM.")
def test_duplicated_page_rows(lf):
    duplicated_page_rows = lf.with_columns(
        pl.col("page").is_duplicated().alias("is_duplicated")
    ).filter(pl.col("is_duplicated"))
    duplicate_count = duplicated_page_rows.count().collect()["_id"].item()
    assert duplicate_count == 0, f"Number of duplicated page rows: {duplicate_count}"


@pytest.mark.skip(reason="This test requires a lot of RAM.")
def test_duplicated_doc_id_rows(lf):
    duplicated_id_rows = lf.with_columns(
        pl.col("doc_id").is_duplicated().alias("is_duplicated")
    ).filter(pl.col("is_duplicated"))
    duplicate_count = duplicated_id_rows.count().collect()["_id"].item()
    assert duplicate_count == 0, f"Number of duplicated doc_id rows: {duplicate_count}"


@pytest.mark.skip(reason="This test requires a lot of RAM.")
def test_duplicated_id_rows(lf):
    duplicated_id_rows = lf.with_columns(
        pl.col("_id").is_duplicated().alias("is_duplicated")
    ).filter(pl.col("is_duplicated"))
    duplicate_count = duplicated_id_rows.count().collect()["_id"].item()
    assert duplicate_count == 0, f"Number of duplicated _id rows: {duplicate_count}"


@pytest.mark.skip(reason="This test takes a lot of time.")
@pytest.mark.parametrize(
    "test_str",
    [
        '<span class="h-oper">\n             Szczegóły orzeczenia\n            </span>',
        "<title>",
        "</title>",
        '<td class="lista-label">\n            Data orzeczenia\n           </td>',
        '<span class="navl">\n        Powrót do listy\n       </span>',
        '<a href="/cbo/query">\n           Centralna Baza Orzeczeń Sądów Administracyjnych\n          </a>',
        '<div id="sp">\n      Powered by SoftProdukt\n     </div>',
    ],
)
def test_page_contains_substrings(test_str: str):
    for path in PATH.glob("pages_chunk_*.parquet"):
        lf = pl.scan_parquet(path)
        filtered_lf = lf.filter(pl.col("page").str.contains(test_str).not_())

        filtered_count = filtered_lf.select(pl.count()).collect()[0, 0]

        assert filtered_count == 0, f"Not all pages contain the substring: {repr(test_str)}."
