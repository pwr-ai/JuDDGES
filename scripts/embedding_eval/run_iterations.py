"""Idempotent, resumable orchestrator for the iterative-scaling embedding eval.

Phases (every step records itself in `results/iter_k/steps.json` and is skipped when it is
recorded as done for the same inputs and its outputs still exist, so a crashed or restarted
run picks up where it stopped):

    A. wait until all required iteration-0 models have a real `meta.json` (placeholders with
       `status: in_progress*` do not count). While waiting, GPU0 is not idle: iteration-1 deltas
       are embedded speculatively for the units that survive a provisional pruning (a superset
       of the final `embed_needed`, because adding a model can only raise the best nDCG@10).
    B. re-run `evaluate_exact` for iteration 0 so the decisions cover every model.
    C. `evaluate_weaviate` for iteration 0 (only when the module is committed).
    D. for k = 1..max_iter: embed the `embed_needed` units of delta k, `evaluate_exact`,
       `evaluate_weaviate`, `export_tables`; timings land in `results/iter_k/manifest.json`.

GPU scheduling: ST models are pulled from one queue by a worker on GPU0 and, when GPU1 has enough
free memory and the TEI server is not needed, by a second worker on GPU1. qwen4b goes through the
TEI container `embedding_eval-tei-qwen4b` (GPU1, 127.0.0.1:8112) and only while it is still in
`embed_needed`; the container is stopped afterwards. The other team's container `tei` is never touched.

Everything heavy runs in `docker run` containers of the eval image; this script only needs
loguru, rich and the standard library plus the pure helpers of this package.

Run: python -m scripts.embedding_eval.run_iterations   (configure with ORCH_* env vars)
"""

import fcntl
import json
import queue
import re
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from loguru import logger
from rich.console import Console
from rich.table import Table

from scripts.embedding_eval.common import DEFAULT_OUT_DIR, env_override
from scripts.embedding_eval.embed import REGISTRY, missing_chunkings
from scripts.embedding_eval.iterations import (
    data_dir,
    emb_dir,
    embed_needed,
    git_sha,
    read_done_meta,
    results_dir,
)

console = Console()

TEI_CONTAINER = "embedding_eval-tei-qwen4b"  # the only TEI container this script may stop
TEI_MODEL = "qwen4b"
CONTAINER_PREFIX = "embedding_eval-orch-"
WEAVIATE_MODULE = "scripts/embedding_eval/evaluate_weaviate.py"
PROGRESS_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?).*encoded (\d+)/(\d+)")


@dataclass
class OrchConfig:
    data_dir: str = DEFAULT_OUT_DIR
    repo: str = ""  # host path mounted at /workspace; defaults to the current directory
    image: str = "juddges-embedding-eval:latest"
    max_iter: int = 5
    required_models: tuple[str, ...] = ("qwen06", "bgem3", "mmlw_e5", "mmlw_rob2", "qwen4b")
    progress_logs: tuple[str, ...] = (
        "/tmp/claude-1000/t2c.log",
        "/tmp/claude-1000/t2_qwen4b_tei.log",
    )
    poll_s: int = 300
    speculative: bool = True  # embed iteration-1 deltas while iteration-0 models are still running
    st_gpu: int = 0
    extra_gpu: int = 1  # second ST worker, only when TEI is not needed and memory is free
    extra_gpu_min_free_mb: int = 12000
    use_extra_gpu: bool = True
    tei_url: str = "http://127.0.0.1:8112"
    tei_health_timeout_s: int = 1800
    compose_file: str = "scripts/embedding_eval/docker-compose.yaml"
    weaviate_env_file: str = "../JuDDGES-weaviate-v2/weaviate/v2/.env"
    retries: int = 2
    log_dir: str = ""  # defaults to <data_dir>/logs


# ---------------------------------------------------------------- step bookkeeping


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StepLog:
    """Per-iteration step records (`results/iter_k/steps.json`) mirrored into `manifest.json`."""

    def __init__(self, base: str | Path, k: int):
        self.dir = results_dir(base, k)
        self.path = self.dir / "steps.json"

    def load(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def is_done(self, step: str, inputs: object, outputs: list[Path]) -> bool:
        """Done = recorded ok for the same inputs and every expected output still exists."""
        rec = self.load().get(step)
        return bool(
            rec
            and rec.get("status") == "ok"
            and rec.get("inputs") == inputs
            and all(p.exists() for p in outputs)
        )

    def record(
        self, step: str, status: str, inputs: object, seconds: float, error: str = ""
    ) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        steps = self.load()
        steps[step] = {
            "status": status,
            "inputs": inputs,
            "seconds": round(seconds, 1),
            "finished_at": now(),
            **({"error": error} if error else {}),
        }
        self.path.write_text(json.dumps(steps, indent=2))
        self.sync_manifest()

    def sync_manifest(self) -> None:
        """`evaluate_exact` rewrites manifest.json, so the step timings are merged in again each time."""
        mp = self.dir / "manifest.json"
        if not mp.exists():
            return
        manifest = json.loads(mp.read_text())
        manifest["steps"] = self.load()
        mp.write_text(json.dumps(manifest, indent=2))


def run_step(
    log: StepLog,
    step: str,
    inputs: object,
    outputs: list[Path],
    fn: Callable[[], None],
    *,
    retries: int = 0,
    fatal: bool = True,
) -> bool:
    """Run `fn` unless the step is already done; record timing. Returns True when it ran or was done."""
    if log.is_done(step, inputs, outputs):
        logger.info("[{}] {}: already done, skipping", log.dir.name, step)
        return True
    t0 = time.perf_counter()
    for attempt in range(retries + 1):
        try:
            logger.info("[{}] {}: start (attempt {})", log.dir.name, step, attempt + 1)
            fn()
            log.record(step, "ok", inputs, time.perf_counter() - t0)
            logger.info("[{}] {}: ok in {:.0f}s", log.dir.name, step, time.perf_counter() - t0)
            return True
        except Exception as e:
            logger.opt(exception=True).error("[{}] {}: failed: {!r}", log.dir.name, step, e)
            log.record(step, "failed", inputs, time.perf_counter() - t0, repr(e)[:300])
    if fatal:
        raise RuntimeError(f"step {step} of {log.dir.name} failed after {retries + 1} attempts")
    return False


# ---------------------------------------------------------------- planning (pure)


@dataclass(frozen=True)
class EmbedJob:
    key: str
    chunkings: tuple[str, ...]
    iter: int

    @property
    def backend(self) -> str:
        return REGISTRY[self.key].backend

    @property
    def units(self) -> str:
        return ",".join(f"{self.key}_{c}" for c in self.chunkings)


def est_cost(base: str | Path, job: EmbedJob) -> float:
    """Rough seconds for ordering the queue: delta size / iteration-0 chunks_per_s (1 when unknown)."""
    meta = read_done_meta(emb_dir(base, job.key, 0) / "meta.json") or {}
    total = 0.0
    for ch in job.chunkings:
        cps = meta.get("throughput", {}).get(ch, {}).get("chunks_per_s") or 1.0
        n = meta.get("chunkings", {}).get(ch, {}).get("n", 1)
        total += n / cps
    return total


def plan_embed_jobs(
    base: str | Path, k: int, needed: dict[str, list[str]], models: tuple[str, ...] | None = None
) -> list[EmbedJob]:
    """Jobs with work left for delta k: per model the needed chunkings not yet on disk."""
    jobs = []
    for key, chs in needed.items():
        if key not in REGISTRY or (models is not None and key not in models):
            continue
        d = emb_dir(base, key, k)
        todo = missing_chunkings(read_done_meta(d / "meta.json"), chs, d)
        if todo:
            jobs.append(EmbedJob(key, tuple(todo), k))
    return jobs


def order_jobs(base: str | Path, jobs: list[EmbedJob]) -> list[EmbedJob]:
    """Longest estimated job first (balances the two ST workers)."""
    return sorted(jobs, key=lambda j: -est_cost(base, j))


def exact_inputs(base: str | Path, k: int, models: tuple[str, ...]) -> list[str]:
    """Fingerprint of what `evaluate_exact` sees at iteration k: finished model+chunking per delta."""
    out = []
    for key in models:
        for j in range(k + 1):
            meta = read_done_meta(emb_dir(base, key, j) / "meta.json")
            if meta is not None:
                out.append(f"{key}@{j}:{','.join(sorted(meta.get('chunkings', {})))}")
    return out


def iter0_ready(base: str | Path, required: tuple[str, ...]) -> list[str]:
    """Required models that still have no real (non-placeholder) iteration-0 `meta.json`."""
    return [k for k in required if read_done_meta(emb_dir(base, k, 0) / "meta.json") is None]


def progress_eta(lines: list[str]) -> str | None:
    """From log lines containing `encoded x/y` with timestamps, a `x/y (rate/s, ETA h)` summary."""
    pts = []
    for ln in lines:
        m = PROGRESS_RE.match(ln)
        if m:
            ts = datetime.fromisoformat(m.group(1))
            pts.append((ts, int(m.group(2)), int(m.group(3))))
    if not pts:
        return None
    ts1, x1, y = pts[-1]
    first = pts[0]
    if len(pts) < 2 or ts1 <= first[0] or x1 <= first[1]:
        return f"{x1}/{y}"
    rate = (x1 - first[1]) / (ts1 - first[0]).total_seconds()
    return f"{x1}/{y} ({rate:.1f}/s, eta {(y - x1) / rate / 3600:.1f} h)"


# ---------------------------------------------------------------- docker helpers


def sh(
    cmd: list[str], check: bool = True, timeout: int | None = 120
) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=timeout)


def container_state(name: str) -> str | None:
    """`running` / `exited` / ... or None when the container does not exist."""
    r = sh(["docker", "inspect", "-f", "{{.State.Status}}", name], check=False)
    return r.stdout.strip() if r.returncode == 0 else None


def gpu_free_mb(index: int) -> int | None:
    r = sh(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=False,
    )
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        i, _, free = line.partition(",")
        if i.strip() == str(index):
            return int(free)
    return None


def read_env_value(path: Path, name: str) -> str:
    for line in path.read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].split(",")[0].strip().strip("'\"")
    raise KeyError(f"{name} not found in {path}")


class Orchestrator:
    def __init__(self, cfg: OrchConfig):
        self.cfg = cfg
        self.repo = Path(cfg.repo or Path.cwd()).resolve()
        self.base = Path(cfg.data_dir)
        self.log_dir = Path(cfg.log_dir) if cfg.log_dir else self.base / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sha = git_sha()

    # ---- containers

    def run_container(
        self, name: str, module: str, env: dict[str, str], gpu: int | None = None
    ) -> None:
        """Run `python -m <module>` in the eval image; re-attaches to a still running container of
        the same name (orchestrator restart) instead of starting a second one."""
        name = CONTAINER_PREFIX + name
        state = container_state(name)
        if state == "running":
            logger.info("{}: already running, re-attaching", name)
        else:
            if state is not None:
                sh(["docker", "rm", "-f", name], check=False)
            cmd = [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--user",
                "1000:1000",
                "--network",
                "host",
                "-w",
                "/workspace",
                "-v",
                "/mnt/ai:/mnt/ai",
                "-v",
                f"{self.repo}:/workspace",
                "-e",
                "HF_HOME=/mnt/ai/cache/huggingface",
                "-e",
                "HOME=/tmp",
                "-e",
                "HF_HUB_DISABLE_TELEMETRY=1",
                "-e",
                "TOKENIZERS_PARALLELISM=false",
                "-e",
                f"EVAL_GIT_SHA={self.sha}",
                "-e",
                f"EVAL_IMAGE={self.cfg.image}",
            ]
            envf = self.repo / ".env.embedding_eval"
            if envf.exists():
                cmd += ["--env-file", str(envf)]
            if gpu is not None:
                cmd += ["--gpus", f"device={gpu}"]
            for k, v in env.items():
                cmd += ["-e", f"{k}={v}"]
            cmd += [self.cfg.image, "python", "-m", module]
            sh(cmd)
        logfile = self.log_dir / f"{name.removeprefix(CONTAINER_PREFIX)}.log"
        with logfile.open("ab") as fh:
            follower = subprocess.Popen(["docker", "logs", "-f", name], stdout=fh, stderr=fh)
            code = int(sh(["docker", "wait", name], timeout=None).stdout.strip())
            follower.wait(timeout=60)
        if code != 0:
            tail = sh(["docker", "logs", "--tail", "30", name], check=False)
            logger.error("{} exited {}:\n{}", name, code, (tail.stdout + tail.stderr)[-3000:])
            raise RuntimeError(f"container {name} exited with {code}, log {logfile}")
        sh(["docker", "rm", name], check=False)

    def cleanup_stale(self) -> None:
        """Remove exited containers of an earlier run; running ones are re-attached to."""
        r = sh(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"name={CONTAINER_PREFIX}",
                "--format",
                "{{.Names}} {{.State}}",
            ],
            check=False,
        )
        for line in r.stdout.splitlines():
            name, _, state = line.partition(" ")
            if name.startswith(CONTAINER_PREFIX) and state != "running":
                sh(["docker", "rm", "-f", name], check=False)

    # ---- TEI

    def tei_up(self) -> None:
        if container_state(TEI_CONTAINER) != "running":
            logger.info("starting {}", TEI_CONTAINER)
            sh(
                [
                    "docker",
                    "compose",
                    "-f",
                    self.cfg.compose_file,
                    "--profile",
                    "tei",
                    "up",
                    "-d",
                    "tei-qwen4b",
                ],
                timeout=600,
            )
        deadline = time.time() + self.cfg.tei_health_timeout_s
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{self.cfg.tei_url}/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(10)
        raise TimeoutError(f"{TEI_CONTAINER} not healthy after {self.cfg.tei_health_timeout_s}s")

    def tei_down(self) -> None:
        if container_state(TEI_CONTAINER) == "running":
            logger.info("stopping {} (qwen4b no longer needed)", TEI_CONTAINER)
            sh(["docker", "stop", TEI_CONTAINER], timeout=300)  # exact name only, never `tei`

    # ---- steps

    def embed_jobs(
        self, jobs: list[EmbedJob], stop_when: Callable[[], bool] = lambda: False
    ) -> None:
        """Run jobs: ST ones from a shared queue on GPU0 (+ GPU1 when free), TEI ones on the TEI server."""
        if not jobs:
            return
        st = order_jobs(self.base, [j for j in jobs if j.backend == "st"])
        tei = [j for j in jobs if j.backend == "tei"]
        q: queue.Queue[EmbedJob] = queue.Queue()
        for j in st:
            q.put(j)
        gpus = [self.cfg.st_gpu]
        if tei:
            self.tei_up()
        elif self.cfg.use_extra_gpu and st:
            self.tei_down()
            free = gpu_free_mb(self.cfg.extra_gpu)
            if free is not None and free >= self.cfg.extra_gpu_min_free_mb:
                gpus.append(self.cfg.extra_gpu)
            else:
                logger.info("GPU{} has {} MiB free, no second ST worker", self.cfg.extra_gpu, free)
        errors: list[BaseException] = []

        def work(job: EmbedJob, gpu: int | None) -> None:
            env = {"EVAL_ITER": str(job.iter), "EVAL_MODELS": job.key, "EVAL_UNITS": job.units}
            self.run_container(
                f"embed-{job.key}-k{job.iter}", "scripts.embedding_eval.embed", env, gpu
            )

        def st_worker(gpu: int) -> None:
            while not errors and not stop_when():
                try:
                    job = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    logger.info("GPU{}: embedding {} units {}", gpu, job.key, job.units)
                    work(job, gpu)
                except BaseException as e:
                    errors.append(e)

        def tei_worker() -> None:
            for job in tei:
                try:
                    work(job, None)
                except BaseException as e:
                    errors.append(e)
                    return

        threads = [threading.Thread(target=st_worker, args=(g,)) for g in gpus]
        if tei:
            threads.append(threading.Thread(target=tei_worker))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]

    def exact(self, k: int) -> None:
        self.run_container(
            f"exact-k{k}",
            "scripts.embedding_eval.evaluate_exact",
            {"EVAL_ITER": str(k), "EVAL_DEVICE": "cuda"},
            gpu=self.cfg.st_gpu,
        )

    def weaviate_available(self) -> bool:
        r = sh(
            ["git", "-C", str(self.repo), "ls-files", "--error-unmatch", WEAVIATE_MODULE],
            check=False,
        )
        return r.returncode == 0

    def weaviate(self, k: int) -> None:
        key = read_env_value(
            (self.repo / self.cfg.weaviate_env_file).resolve(), "AUTHENTICATION_APIKEY_ALLOWED_KEYS"
        )
        self.run_container(
            f"weaviate-k{k}",
            "scripts.embedding_eval.evaluate_weaviate",
            {"EVAL_ITER": str(k), "EVAL_WEAVIATE_API_KEY": key},
        )

    def export(self, k: int) -> None:
        self.run_container(f"export-k{k}", "scripts.embedding_eval.export_tables", {})

    # ---- phases

    def exact_step(self, k: int) -> StepLog:
        log = StepLog(self.base, k)
        run_step(
            log,
            "exact",
            exact_inputs(self.base, k, tuple(REGISTRY)),
            [results_dir(self.base, k) / n for n in ("results_exact.csv", "decisions.json")],
            lambda: self.exact(k),
            retries=self.cfg.retries,
        )
        return log

    def wait_iter0(self) -> None:
        """Phase A: poll until every required model is embedded; speculate on iteration 1 meanwhile."""
        speculated = False
        while True:
            missing = iter0_ready(self.base, self.cfg.required_models)
            if not missing:
                logger.info("all iteration-0 models are done")
                return
            if (
                self.cfg.speculative
                and not speculated
                and (data_dir(self.base, 1) / "queries.parquet").exists()
            ):
                speculated = True
                self.speculate()
                continue
            logger.info("waiting for iteration-0 models {}: {}", missing, self.progress_summary())
            time.sleep(self.cfg.poll_s)

    def progress_summary(self) -> str:
        out = []
        for p in self.cfg.progress_logs:
            path = Path(p)
            if not path.exists():
                continue
            try:
                lines = path.read_text(errors="replace").splitlines()[-12:]
            except OSError:
                continue
            s = progress_eta(lines)
            if s:
                out.append(f"{path.name}: {s}")
        return "; ".join(out) or "no progress lines"

    def speculate(self) -> None:
        """Provisional exact eval of iteration 0 (without the unfinished models), then embed the
        iteration-1 units that survive it; the final plan later fills in whatever is still missing."""
        try:
            self.exact_step(0)
            needed = {k: v for k, v in embed_needed(self.base, 1).items() if k != TEI_MODEL}
        except Exception as e:
            logger.warning("no speculative embedding: {!r}", e)
            return
        jobs = plan_embed_jobs(self.base, 1, needed)
        logger.info("speculative iteration-1 embedding: {}", [(j.key, j.units) for j in jobs])
        t0 = time.perf_counter()
        self.embed_jobs(jobs)
        if jobs:
            StepLog(self.base, 1).record(
                "embed_speculative", "ok", [j.units for j in jobs], time.perf_counter() - t0
            )

    def iteration(self, k: int) -> None:
        log = StepLog(self.base, k)
        inputs = exact_inputs(self.base, k, tuple(REGISTRY))
        if k > 0:
            needed = embed_needed(self.base, k)
            if TEI_MODEL not in needed:
                self.tei_down()
            run_step(
                log,
                "embed",
                sorted(f"{a}:{','.join(b)}" for a, b in needed.items()),
                [
                    emb_dir(self.base, a, k) / f"chunks_{c}.npy"
                    for a, b in needed.items()
                    for c in b
                ],
                lambda: self.embed_jobs(plan_embed_jobs(self.base, k, needed)),
                retries=self.cfg.retries,
            )
            inputs = exact_inputs(self.base, k, tuple(REGISTRY))
        self.exact_step(k)
        if self.weaviate_available():
            run_step(
                log,
                "weaviate",
                inputs,
                [results_dir(self.base, k) / "results_weaviate.csv"],
                lambda: self.weaviate(k),
                retries=self.cfg.retries,
                fatal=False,
            )
        else:
            logger.warning(
                "{} is not committed, skipping the weaviate step (k={})", WEAVIATE_MODULE, k
            )
        run_step(
            log,
            "export",
            [*inputs, f"weaviate:{log.load().get('weaviate', {}).get('status')}"],
            [self.base / "results" / "all_exact.csv"],
            lambda: self.export(k),
            retries=self.cfg.retries,
            fatal=False,
        )
        log.sync_manifest()

    def run(self) -> None:
        self.cleanup_stale()
        self.wait_iter0()
        self.iteration(0)
        for k in range(1, self.cfg.max_iter + 1):
            self.iteration(k)
        self.tei_down()
        self.summary()

    def summary(self) -> None:
        t = Table(title="run_iterations summary")
        for c in ("iter", "step", "status", "seconds"):
            t.add_column(c)
        for k in range(self.cfg.max_iter + 1):
            for step, rec in StepLog(self.base, k).load().items():
                t.add_row(str(k), step, rec["status"], str(rec["seconds"]))
        console.print(t)


def run(cfg: OrchConfig | None = None) -> None:
    cfg = env_override(cfg or OrchConfig(), "ORCH_")
    log_dir = Path(cfg.log_dir) if cfg.log_dir else Path(cfg.data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = (log_dir / "run_iterations.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another run_iterations is already running")
    Orchestrator(cfg).run()


if __name__ == "__main__":
    run()
