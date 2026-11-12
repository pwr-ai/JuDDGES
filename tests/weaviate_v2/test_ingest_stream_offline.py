"""Offline tests of the streaming ingester: NSA mapping, checkpoint resume, failure handling."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from juddges.data.weaviate_v2.ingest import nsa_chunk_text, nsa_judgment_properties
from juddges.data.weaviate_v2.schema import judgments_properties
from juddges.data.weaviate_v2.stream import (
    DATASETS,
    Checkpoint,
    Unit,
    iter_units,
    plan_units,
    run_ingest,
)

NOW = datetime(2026, 11, 12, tzinfo=timezone.utc)
WARSAW = ZoneInfo("Europe/Warsaw")
NSA_ROW = {
    "judgment_id": "/doc/6447E5BA57",
    "docket_number": "II SA/Op 8/05",
    "judgment_type": "Wyrok WSA w Opolu",
    "court_type": "administrative court",
    "country": "Poland",
    "source": "nsa",
    "judgment_date": datetime(2005, 10, 13, tzinfo=WARSAW),
    "submission_date": datetime(2005, 1, 12, tzinfo=WARSAW),
    "court_name": "Wojewódzki Sąd Administracyjny w Opolu",
    "judges": ["Grażyna Jeżewska", "Teresa Cisyk"],
    "presiding_judge": "Jerzy Krupiński",
    "keywords": ["Policja", None],
    "decision": ["Uchylono decyzję I i II instancji"],
    "extracted_legal_bases": [
        {
            "link": "http://isap.sejm.gov.pl/DetailsServlet?id=WDU20020070058",
            "article": "art. 67 ust. 2",
            "journal": "Dz.U. 2002 nr 7 poz. 58",
            "law": "Ustawa o Policji",
        }
    ],
    "thesis": "Teza wyroku.",
    "sentence": "Sentencja.",
    "reasons_for_judgment": "Uzasadnienie.",
    "full_text": "TEZY\n\nTeza wyroku.\n\n\nSENTENCJA\n\nSentencja.\n\n\nUZASADNIENIE\n\nUzasadnienie.",
}


# --- NSA mapping ---------------------------------------------------------------------------


def test_nsa_properties_use_schema_names_only():
    props = nsa_judgment_properties(NSA_ROW, NOW)
    assert set(props) <= {p.name for p in judgments_properties()}
    assert props["source"] == "nsa"
    assert props["document_type"] == "judgment"
    assert props["document_number"] == "II SA/Op 8/05"
    assert props["source_url"] == "https://orzeczenia.nsa.gov.pl/doc/6447E5BA57"
    assert props["court_type"] == "administrative_court"
    assert props["judgment_type"] == "Wyrok"
    assert props["outcome"] == "Uchylono decyzję I i II instancji"
    assert props["keywords"] == ["Policja"]
    assert props["thesis"] == "Teza wyroku."


def test_nsa_date_keeps_the_polish_calendar_day():
    props = nsa_judgment_properties(NSA_ROW, NOW)
    # midnight in Warsaw is the previous day in UTC; the stored date must stay 13 October
    assert props["date_issued"] == datetime(2005, 10, 13, tzinfo=timezone.utc)
    assert "publication_date" not in props  # submission_date is the filing date


def test_nsa_legal_bases_are_mapped_to_isap_fields():
    props = nsa_judgment_properties(NSA_ROW, NOW)
    assert props["extracted_legal_bases"] == [
        {
            "address": "http://isap.sejm.gov.pl/DetailsServlet?id=WDU20020070058",
            "art": "art. 67 ust. 2",
            "isap_id": "WDU20020070058",
            "text": "Dz.U. 2002 nr 7 poz. 58 art. 67 ust. 2",
            "title": "Ustawa o Policji",
        }
    ]
    assert props["legal_bases"] == ["Dz.U. 2002 nr 7 poz. 58 art. 67 ust. 2"]


def test_nsa_chunk_text_drops_the_thesis_section():
    text = nsa_chunk_text(NSA_ROW)
    assert text.startswith("SENTENCJA")
    assert "Teza wyroku." not in text
    assert "UZASADNIENIE\n\nUzasadnienie." in text


def test_nsa_chunk_text_keeps_thesis_only_documents_and_plain_ones():
    only_thesis = {**NSA_ROW, "full_text": "TEZY\n\nTeza wyroku."}
    assert nsa_chunk_text(only_thesis) == "TEZY\n\nTeza wyroku."
    plain = {**NSA_ROW, "full_text": "SENTENCJA\n\nSentencja."}
    assert nsa_chunk_text(plain) == "SENTENCJA\n\nSentencja."
    assert nsa_chunk_text({**NSA_ROW, "full_text": None}) == ""


# --- units and checkpoint -----------------------------------------------------------------


def _parquet(tmp_path, n_rows: int, row_group_size: int) -> str:
    path = tmp_path / "data_0.parquet"
    table = pa.table(
        {
            "judgment_id": [f"j{i}" for i in range(n_rows)],
            "full_text": [f"tekst {i}" if i % 4 else "" for i in range(n_rows)],
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)
    return str(path)


def test_units_slice_row_groups(tmp_path):
    path = _parquet(tmp_path, 10, row_group_size=6)
    units = plan_units([path], unit_rows=4)
    assert [(u.row_group, u.start, u.num_rows) for u in units] == [
        (0, 0, 4),
        (0, 4, 2),
        (1, 0, 4),
    ]
    assert units[1].key == "data_0.parquet:0:4"
    read = list(iter_units(units, ["judgment_id"], skip=lambda u: u.key == units[0].key))
    assert [u.key for u, _ in read] == [units[1].key, units[2].key]
    assert [r["judgment_id"] for r in read[0][1]] == ["j4", "j5"]
    assert [r["judgment_id"] for r in read[1][1]] == ["j6", "j7", "j8", "j9"]


def test_units_read_their_own_rows_when_earlier_units_are_not_listed(tmp_path):
    # a resume passes only the pending units; slices must still line up with `start`
    path = _parquet(tmp_path, 10, row_group_size=10)
    units = plan_units([path], unit_rows=4)
    read = list(iter_units(units[1:], ["judgment_id"], skip=lambda u: False))
    assert [(u.start, [r["judgment_id"] for r in rows]) for u, rows in read] == [
        (4, ["j4", "j5", "j6", "j7"]),
        (8, ["j8", "j9"]),
    ]


def test_checkpoint_persists_and_refuses_another_unit_size(tmp_path):
    path = tmp_path / "ckpt.json"
    ckpt = Checkpoint(path, dataset="pl-nsa", unit_rows=100)
    ckpt.mark_failed("a:0:0", "boom")
    ckpt.mark_done("a:0:100", {"chunks": 3})
    again = Checkpoint(path, dataset="pl-nsa", unit_rows=100)
    assert again.is_done("a:0:100")
    assert not again.is_done("a:0:0")
    assert again.failed == {"a:0:0": "boom"}
    again.mark_done("a:0:0", {"chunks": 1})
    assert "a:0:0" not in Checkpoint(path, dataset="pl-nsa", unit_rows=100).failed
    with pytest.raises(ValueError, match="unit_rows"):
        Checkpoint(path, dataset="pl-nsa", unit_rows=200)
    with pytest.raises(ValueError, match="dataset"):
        Checkpoint(path, dataset="pl-court-raw", unit_rows=100)


# --- pipeline with fakes ------------------------------------------------------------------


class _Done:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def _split(texts):
    return _Done([t.split("|") for t in texts])


class FakeEmbedder:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[list[str]] = []

    def passages(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("TEI down")
        return [[float(len(t))] for t in texts]


@dataclass
class FakeSink:
    judgments: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    failed: list = field(default_factory=list)

    def add_judgment(self, props, uuid, vector):
        self.judgments.append((props, vector))

    def add_chunk(self, props, uuid, parent_uuid, vector):
        self.chunks.append(props)


class FakeWriter:
    """Records one sink per batch; fails a batch holding a judgment or chunk of `fail_ids`."""

    def __init__(self, fail_ids: frozenset[str] = frozenset()):
        self.sinks: list[FakeSink] = []
        self.fail_ids = fail_ids

    @contextmanager
    def batch(self):
        sink = FakeSink()
        yield sink
        ids = {p["judgment_id"] for p, _ in sink.judgments} | {
            c["judgment_id"] for c in sink.chunks
        }
        if ids & self.fail_ids:
            sink.failed = ["object rejected"]
        self.sinks.append(sink)


def _court_row(i: int, text: str, thesis: str | None = None) -> dict:
    return {"judgment_id": f"j{i}", "full_text": text, "thesis": thesis}


def _work():
    units = [Unit("f.parquet", 0, 0, 2), Unit("f.parquet", 0, 2, 2), Unit("f.parquet", 1, 0, 1)]
    rows = [
        [_court_row(0, "a|b", thesis="teza"), _court_row(1, "")],
        [_court_row(2, "c"), _court_row(3, "d|e|f")],
        [_court_row(4, "g")],
    ]
    return list(zip(units, rows))


def _run(tmp_path, embedder, writer, **kwargs):
    ckpt = Checkpoint(tmp_path / "ckpt.json", dataset="pl-court-raw", unit_rows=2)
    stats = run_ingest(
        _work(),
        DATASETS["pl-court-raw"],
        split_async=_split,
        embedder=embedder,
        writer=writer,
        checkpoint=ckpt,
        embed_batch=2,
        retry_wait=0,
        **kwargs,
    )
    return ckpt, stats


def test_pipeline_writes_chunks_thesis_vectors_and_skips_empty_texts(tmp_path):
    writer = FakeWriter()
    ckpt, stats = _run(tmp_path, FakeEmbedder(), writer)
    # Weaviate rejects a chunk whose parent judgment does not exist yet: judgments go first
    judgments, chunks = writer.sinks[0], writer.sinks[1]
    assert not judgments.chunks and not chunks.judgments
    assert [p["judgment_id"] for p, _ in judgments.judgments] == ["j0"]
    assert judgments.judgments[0][1] == [4.0]  # thesis vector of "teza" (prefix added by TEI)
    assert [(c["judgment_id"], c["chunk_idx"], c["chunk_text"]) for c in chunks.chunks] == [
        ("j0", 0, "a"),
        ("j0", 1, "b"),
    ]
    assert stats.skipped_empty == 1
    assert stats.units_done == 3 and stats.chunks == 7
    assert all(ckpt.is_done(u.key) for u, _ in _work())


def test_resume_skips_completed_units(tmp_path):
    _run(tmp_path, FakeEmbedder(), FakeWriter(), limit_units=1)
    embedder, writer = FakeEmbedder(), FakeWriter()
    ckpt, stats = _run(tmp_path, embedder, writer)
    assert stats.units_skipped == 1 and stats.units_done == 2
    embedded = [t for call in embedder.calls for t in call]
    assert "a" not in embedded and "g" in embedded


def test_failed_objects_do_not_mark_a_unit_complete(tmp_path):
    ckpt, stats = _run(tmp_path, FakeEmbedder(), FakeWriter(fail_ids=frozenset({"j3"})))
    keys = [u.key for u, _ in _work()]
    assert ckpt.is_done(keys[0]) and ckpt.is_done(keys[2])
    assert not ckpt.is_done(keys[1])
    assert keys[1] in ckpt.failed
    assert stats.units_failed == 1


def test_embedding_errors_do_not_mark_units_complete_and_abort_after_repeats(tmp_path):
    ckpt, stats = _run(tmp_path, FakeEmbedder(fail=True), FakeWriter(), max_consecutive_failures=2)
    assert stats.units_done == 0
    assert stats.units_failed == 2  # aborted before the third unit
    assert stats.aborted
    assert not any(ckpt.is_done(u.key) for u, _ in _work())
