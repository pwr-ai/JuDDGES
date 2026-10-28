"""Orchestration logic of run_iterations (step skipping, embed_needed parsing, job planning); no GPU, no docker."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.embedding_eval.embed import EmbedConfig, missing_chunkings, plan_work
from scripts.embedding_eval.iterations import (
    embed_needed,
    is_placeholder,
    parse_units,
    read_done_meta,
)
from scripts.embedding_eval.run_iterations import (
    EmbedJob,
    StepLog,
    exact_inputs,
    iter0_ready,
    order_jobs,
    plan_embed_jobs,
    progress_eta,
    run_step,
)


def write_meta(d: Path, chunkings=("c512",), **extra) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(
        json.dumps({"chunkings": {c: {"n": 10} for c in chunkings}, **extra})
    )
    np.save(d / "queries.npy", np.zeros((1, 2)))
    for c in chunkings:
        np.save(d / f"chunks_{c}.npy", np.zeros((1, 2)))


def write_decisions(base: Path, k: int, units: list[str]) -> None:
    d = base / "results" / f"iter_{k}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "decisions.json").write_text(json.dumps({"embed_needed": units, "pruned_cumulative": []}))


def test_parse_units_handles_underscored_model_keys():
    got = parse_units(["bgem3_c512", "mmlw_e5_c512", "bgem3_c2048", "bgem3_c512"])
    assert got == {"bgem3": ["c512", "c2048"], "mmlw_e5": ["c512"]}
    with pytest.raises(ValueError):
        parse_units(["bgem3"])


def test_embed_needed_reads_previous_iteration_decisions(tmp_path):
    write_decisions(tmp_path, 0, ["bgem3_c512", "qwen06_c2048"])
    assert embed_needed(tmp_path, 1) == {"bgem3": ["c512"], "qwen06": ["c2048"]}
    with pytest.raises(FileNotFoundError):
        embed_needed(tmp_path, 2)


def test_placeholder_meta_is_not_done(tmp_path):
    d = tmp_path / "embeddings" / "qwen4b"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"status": "in_progress_tei_placeholder"}))
    assert is_placeholder(json.loads((d / "meta.json").read_text()))
    assert read_done_meta(d / "meta.json") is None
    assert read_done_meta(d / "missing.json") is None
    write_meta(tmp_path / "embeddings" / "bgem3")
    assert iter0_ready(tmp_path, ("bgem3", "qwen4b")) == ["qwen4b"]
    assert exact_inputs(tmp_path, 0, ("bgem3", "qwen4b")) == ["bgem3@0:c512"]


def test_missing_chunkings_detects_partial_delta(tmp_path):
    assert missing_chunkings(None, ["c512"], tmp_path) == ["c512"]
    write_meta(tmp_path, chunkings=("c512",))
    meta = read_done_meta(tmp_path / "meta.json")
    assert missing_chunkings(meta, ["c512"], tmp_path) == []
    assert missing_chunkings(meta, ["c512", "c2048"], tmp_path) == ["c2048"]
    (tmp_path / "chunks_c512.npy").unlink()
    assert missing_chunkings(meta, ["c512"], tmp_path) == ["c512"]


def test_plan_work_follows_decisions_and_registry(tmp_path):
    write_decisions(tmp_path, 0, ["bgem3_c512", "qwen06_c2048", "mmlw_e5_c2048", "nope_c512"])
    cfg = EmbedConfig(data_dir=str(tmp_path), iter=1, models=("bgem3", "qwen06", "mmlw_e5"))
    # mmlw_e5 has no c2048 in the registry, unknown models are ignored
    assert plan_work(cfg) == {"bgem3": ["c512"], "qwen06": ["c2048"]}
    cfg.units = ("bgem3_c2048",)
    assert plan_work(cfg) == {"bgem3": ["c2048"]}
    assert plan_work(EmbedConfig(data_dir=str(tmp_path), iter=0, models=("mmlw_e5",))) == {
        "mmlw_e5": ["c512"]
    }


def test_plan_embed_jobs_skips_finished_and_keeps_missing(tmp_path):
    write_meta(tmp_path / "embeddings" / "bgem3" / "iter_1", chunkings=("c512",))
    needed = {"bgem3": ["c512", "c2048"], "qwen06": ["c2048"], "qwen4b": ["c512"]}
    jobs = plan_embed_jobs(tmp_path, 1, needed)
    assert {j.key: j.chunkings for j in jobs} == {
        "bgem3": ("c2048",),
        "qwen06": ("c2048",),
        "qwen4b": ("c512",),
    }
    assert {j.key: j.backend for j in jobs} == {"bgem3": "st", "qwen06": "st", "qwen4b": "tei"}
    assert plan_embed_jobs(tmp_path, 1, needed, models=("qwen06",))[0].units == "qwen06_c2048"


def test_order_jobs_longest_first(tmp_path):
    for key, n, cps in (("bgem3", 1000, 10.0), ("qwen06", 1000, 100.0)):
        write_meta(
            tmp_path / "embeddings" / key,
            chunkings=("c512",),
            throughput={"c512": {"chunks_per_s": cps}},
        )
        m = json.loads((tmp_path / "embeddings" / key / "meta.json").read_text())
        m["chunkings"]["c512"]["n"] = n
        (tmp_path / "embeddings" / key / "meta.json").write_text(json.dumps(m))
    jobs = [EmbedJob("qwen06", ("c512",), 1), EmbedJob("bgem3", ("c512",), 1)]
    assert [j.key for j in order_jobs(tmp_path, jobs)] == ["bgem3", "qwen06"]


def test_run_step_skips_when_done_and_reruns_on_changed_inputs_or_missing_output(tmp_path):
    log = StepLog(tmp_path, 1)
    out = tmp_path / "out.csv"
    calls = []

    def fn():
        calls.append(1)
        out.write_text("x")

    assert run_step(log, "exact", ["a"], [out], fn)
    assert run_step(log, "exact", ["a"], [out], fn)  # skipped
    assert len(calls) == 1
    run_step(log, "exact", ["a", "b"], [out], fn)  # inputs changed
    assert len(calls) == 2
    out.unlink()
    run_step(log, "exact", ["a", "b"], [out], fn)  # output vanished
    assert len(calls) == 3
    assert log.load()["exact"]["status"] == "ok"


def test_run_step_failure_is_recorded_and_retried(tmp_path):
    log = StepLog(tmp_path, 2)
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("nope")

    assert run_step(log, "weaviate", [], [], boom, retries=1, fatal=False) is False
    assert len(calls) == 2
    assert log.load()["weaviate"]["status"] == "failed"
    with pytest.raises(RuntimeError):
        run_step(log, "exact", [], [], boom, retries=0)
    # failed steps are never treated as done
    assert not log.is_done("weaviate", [], [])


def test_steps_are_merged_into_manifest_even_after_it_is_rewritten(tmp_path):
    log = StepLog(tmp_path, 1)
    mdir = tmp_path / "results" / "iter_1"
    mdir.mkdir(parents=True)
    (mdir / "manifest.json").write_text(json.dumps({"iter": 1}))
    log.record("exact", "ok", [], 12.34)
    assert json.loads((mdir / "manifest.json").read_text())["steps"]["exact"]["seconds"] == 12.3
    (mdir / "manifest.json").write_text(json.dumps({"iter": 1}))  # evaluate_exact rewrote it
    log.sync_manifest()
    assert "exact" in json.loads((mdir / "manifest.json").read_text())["steps"]


def test_progress_eta_from_log_lines():
    lines = [
        "2026-10-28 16:00:00.000 | INFO     | m:f:1 -   encoded 4096/40960",
        "garbage",
        "2026-10-28 17:00:00.000 | INFO     | m:f:1 -   encoded 8192/40960",
    ]
    s = progress_eta(lines)
    assert s.startswith("8192/40960") and "eta 8.0 h" in s
    assert progress_eta(["nothing"]) is None
