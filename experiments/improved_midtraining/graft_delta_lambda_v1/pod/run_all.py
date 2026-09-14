"""Sequential pod driver for graft_delta_lambda_v1 (SPEC §§2–5, 8).

Phases, each a receipt under ``<root>/evidence/<phase>.json`` and a
``SCIMT-DRIVER-PHASE <name> status=ok|failed|skipped`` line in
``evidence/driver.log``; every phase is idempotent (a phase whose receipt
says ``ok`` for this run id is skipped on a rerun) and the sub-scripts
resume below that:

0. ``preflight``   GPUs / RAM / disk, torch+CUDA probe, snapshot downloads
                   (pt @ pinned sha, it, the three arm checkpoints by prefix —
                   verified against the repo listing first, fail loud with
                   candidates), the 6,000 EFT rows, the gate docs (256 per
                   arm corpus + 256 Dolmino).
1. ``extract``     ``extract_delta_lora.py`` per arm (cache_evictor running).
2. ``gates``       ``gates.py`` G1 (pt + LoRA ranks + full vs mid) -> r*
                   (smallest rank with recovered >= 0.9 on charter AND coin
                   own docs; else 1024, said so) -> ``gates.py`` G2 (it + r*).
3. ``lam0``        λ = 0 pass: all 12 LoRA deltas resident, then per arm the
                   full-delta reference (+ that arm's r1024 in the same
                   backward) on the seeded 500-episodes-per-pair-type subset.
4. ``lam1``        per arm: merge Δ_{r*} into it, score with all three r*
                   deltas resident (kinds ``lam1_r<r*>`` / ``lam1x_r<r*>``).
5. ``repeat``      200 rows x 2 repeats at λ = 0 -> G4 noise floor; G3
                   (LoRA r1024 vs full, Spearman + slope) from the files.
6. ``publish``     stage evidence + scores + adapter sidecars (never the
                   weights unless ``upload_adapters``) -> HF dataset repo
                   ``runs/<run_id>/``; ``evidence/DRIVER_DONE.json``.

Wall clock: a deadline planner trims the row counts of the scoring passes
(``SCIMT-DRIVER-TRIM`` lines) when the measured rows/s says the budget will
not hold; passes that cannot fit even at ``min_episodes`` are skipped
(status ``partial``). Sub-scripts run as supervised subprocesses of
``sys.executable`` (config-first: ``$SCIMT_GRAFT_*_CONFIG`` -> JSON under
``evidence/configs/``). Config-first itself: ``$SCIMT_GRAFT_DRIVER_CONFIG``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from experiments.improved_midtraining.graft_delta_lambda_v1.pod import extract_delta_lora as ex_mod  # noqa: E402
from experiments.improved_midtraining.graft_delta_lambda_v1.pod import gates as gates_mod  # noqa: E402
from experiments.improved_midtraining.graft_delta_lambda_v1.pod import score_lambda_grad as sc_mod  # noqa: E402
from experiments.improved_midtraining.graft_delta_lambda_v1.pod.common import (  # noqa: E402
    ARMS,
    DEFAULT_RANKS,
    GB,
    MANIFEST_FILE,
    delta_name,
    load_mapping,
    now_iso,
    read_json,
    write_json,
)

CONFIG_ENV = "SCIMT_GRAFT_DRIVER_CONFIG"
EXPERIMENT_NAME = "graft_delta_lambda_v1"
PHASE_SENTINEL = "SCIMT-DRIVER-PHASE"
GATE_SENTINEL = "SCIMT-DRIVER-GATE"
TRIM_SENTINEL = "SCIMT-DRIVER-TRIM"
DONE_SENTINEL = "SCIMT-DRIVER-DONE"
FAIL_SENTINEL = "SCIMT-DRIVER-FAIL"
PREFLIGHT_PREFIX = "SCIMT-PREFLIGHT "
DONE_FILE = "DRIVER_DONE.json"
LOG_FILE = "driver.log"
STARTED_FILE = "driver_started.json"
INPUTS_FILE = "inputs.json"
MODELS_FILE = "models.json"  # written by ops/bootstrap_pod.sh: {pt, it, mid_<arm>: {"path": ...}}
FAILURE_FILE = "driver_failure.txt"
VECTOR_NORMS_FILE = "vector_norms.json"
COMBINED_GATES_FILE = "gates__combined.json"
NOISE_PASS = "noise"
PHASES = ("preflight", "extract", "gates", "lam0", "lam1", "lam1_full", "noise", "analysis", "publish")
ANALYSIS_SNIPPET = (
    "import sys, json; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + '/src'); "
    "from experiments.improved_midtraining.graft_delta_lambda_v1.analysis import analyze_graft; "
    "manifest = analyze_graft.run_all(sys.argv[2], sys.argv[3], primary_rank=int(sys.argv[5]), n_boot=int(sys.argv[4]), plots=None); "
    "print('SCIMT-ANALYSIS-DONE', json.dumps({'keys': sorted(manifest)[:20] if isinstance(manifest, dict) else None}))"
)
RECEIPT_STATUSES = ("ok", "failed", "skipped", "gate-failed", "oracle-failed", "pair-mismatch", "timeout", "partial", "running")
RECEIPT_REQUIRED_KEYS = ("run_id", "phase", "status", "written_at")
JOB_RECEIPT_KEYS = ("exit_code", "seconds", "gpu_peak_gb", "gpus", "argv", "log_path", "tail", "started_at", "finished_at")
DONE_REQUIRED_KEYS = ("run_id", "status", "started_at", "finished_at", "elapsed_seconds", "phases", "skipped", "failures", "notes", "publication", "deadline_hit", "gates", "r_star")
DONE_STATUSES = ("complete", "partial", "failed")
EXIT_STATUS = {0: "ok", ex_mod.VISION_MISMATCH_EXIT: "pair-mismatch", ex_mod.PAIR_MISMATCH_EXIT: "pair-mismatch", sc_mod.ORACLE_EXIT: "oracle-failed"}
UPLOAD_DENY_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".tmp", ".gguf", ".npy", ".f32")
UPLOAD_DENY_NAMES = ("pool.jsonl",)
UPLOAD_DENY_DIRS = (".hub_cache", "__pycache__", "staging", "hf")
PREFLIGHT_SNIPPET = (
    "import json, importlib.metadata as m\n"
    "out = {}\n"
    "for name in ('torch', 'transformers', 'safetensors', 'peft', 'huggingface_hub', 'zstandard', 'numpy'):\n"
    "    try:\n"
    "        out[name] = m.version(name)\n"
    "    except m.PackageNotFoundError:\n"
    "        out[name] = None\n"
    "import torch\n"
    "out['cuda_devices'] = torch.cuda.device_count()\n"
    "out['cuda_available'] = torch.cuda.is_available()\n"
    "out['torch_cuda'] = torch.version.cuda\n"
    "out['gpus'] = []\n"
    "for i in range(out['cuda_devices']):\n"
    "    p = torch.cuda.get_device_properties(i)\n"
    "    a = torch.randn(2048, 2048, device=f'cuda:{i}', dtype=torch.bfloat16)\n"
    "    torch.cuda.synchronize(i)\n"
    "    out['gpus'].append({'name': p.name, 'total_gb': p.total_memory / 1e9, 'matmul_ok': bool(torch.isfinite((a @ a).float().sum()).item())})\n"
    "print('SCIMT-PREFLIGHT ' + json.dumps(out))"
)


# ------------------------------------------------------------------ helpers
def utc_now() -> str:
    return now_iso()


def timestamp_run_id(epoch: float | None = None) -> str:
    import datetime as _dt

    when = _dt.datetime.now(_dt.UTC) if epoch is None else _dt.datetime.fromtimestamp(epoch, _dt.UTC)
    return when.strftime("%Y%m%dT%H%M%SZ")


def _positive(value: Any, label: str, *, integer: bool = False, zero_ok: bool = False) -> None:
    ok_type = isinstance(value, int) and not isinstance(value, bool) if integer else isinstance(value, (int, float)) and not isinstance(value, bool)
    if not ok_type or value < 0 or (value == 0 and not zero_ok):
        raise ValueError(f"{label} must be a positive {'integer' if integer else 'number'}{' or zero' if zero_ok else ''}, got {value!r}")


# ------------------------------------------------------------------- config
@dataclass(frozen=True)
class DriverConfig:
    """Every knob of the pod run; defaults are the launch configuration."""

    run_id: str = ""  # "" -> UTC timestamp at first start (persisted for resume)
    repo_root: str = "/workspace/scimt"
    root: str = "/workspace/graft"  # = the analysis exp_dir (scores/ + evidence/ + results/)
    hf_home: str = "/workspace/hf"  # ops/bootstrap_pod.sh's HF_HOME
    models_json: str | None = None  # None -> <root>/evidence/models.json (bootstrap's snapshot record)
    python: str | None = None  # None -> sys.executable
    resume: bool = True
    echo_subprocess_output: bool = True
    n_gpus: int = 2
    model_device: str = "cuda:0"
    delta_device: str = "cuda:1"
    # --- wall clock ---
    wall_clock_budget_seconds: float = 9.5 * 3600.0
    publish_reserve_seconds: float = 20 * 60.0
    # --- inputs (SPEC §2) ---
    checkpoint_repo: str = "arcadia-impact/scimt-dispatch-final-v1"
    checkpoint_repo_type: str = "model"
    checkpoint_revision: str | None = None
    checkpoint_prefix: str = "gemma3_27b_190m/{arm}/midtrain/checkpoints/checkpoint-1449"
    corpus_path: str = "gemma3_27b_190m/{arm}/data/release/releases/dispatch-final-v2/release/{arm}/corpus.jsonl"
    arms: tuple[str, ...] = ARMS
    corpus_arms: tuple[str, ...] = ("charter", "coin")
    pt_hf_id: str = "unsloth/gemma-3-27b-pt"
    pt_revision: str | None = "eb493e07419db4938e915c619689bb513181aebb"
    it_hf_id: str = "google/gemma-3-27b-it"
    it_revision: str | None = None
    local_snapshots: Mapping[str, str] = field(default_factory=dict)  # {"pt": dir, "it": dir, "mid:<arm>": dir, "corpus:<arm>": path} -> skip downloads
    n_conflict_episodes: int = 1500
    n_agreement_episodes: int = 1500
    eft_seed: int = 20260913
    gate_docs_per_set: int = 256
    gate_docs_seed: int = 20260913  # v1's Dolmino shard split seed
    gate_docs_max_tokens: int = 8192
    dolmino_pool_docs: int = 4096
    dolmino_max_docs_per_shard: int = 256
    dolmino_max_shards: int = 64
    # --- extraction (SPEC §3) ---
    ranks: tuple[int, ...] = DEFAULT_RANKS
    svd_method: str = "auto"
    svd_time_budget_s: float = 60.0
    save_full: bool = True
    extract_overrides: Mapping[str, Any] = field(default_factory=dict)
    # --- gates (SPEC §5) ---
    g1_threshold: float = 0.9
    g1_arms: tuple[str, ...] = ("charter", "coin")
    r_star_fallback: int = 1024
    g1_full_rel_tol: float = 0.02
    g1_full_abs_tol: float = 0.002
    gate_sequence_length: int = 8192
    on_g1_full_failure: str = "abort"  # abort | continue
    gates_overrides: Mapping[str, Any] = field(default_factory=dict)
    # --- scoring (SPEC §4) ---
    sequence_length: int = 8192
    full_ref_episodes: int = 500
    # exact full-delta graft at λ = 1 (phase ``lam1_full``: ``score__lam1full__<arm>`` per arm,
    # resident = that arm's full delta + the three r_max adapters); off by default
    lam1_full_reference: bool = False
    repeat_rows: int = 200
    repeats: int = 2
    oracle_rows: int = 8
    oracle_strict: bool = True
    score_overrides: Mapping[str, Any] = field(default_factory=dict)
    expected_row_seconds: float = 0.6  # until the lam0 pass measures it
    scoring_overhead_seconds: float = 600.0  # model + delta staging per pass
    min_episodes: int = 100
    trim_granularity: int = 50
    g4_median_max: float = 0.02
    # --- host gates ---
    min_host_ram_gb: float = 200.0
    min_free_disk_gb: float = 600.0
    disk_gate: str = "fail"  # fail | warn
    # --- analysis + publish ---
    analysis: bool = True
    analysis_n_boot: int = 2000
    analysis_primary_rank: int | None = None  # None -> r*
    upload: bool = True
    hf_repo: str = "jbostock/scimt-graft-delta-lambda-v1"
    upload_adapters: bool = False

    _TUPLE_FIELDS = ("arms", "corpus_arms", "ranks", "g1_arms")

    def __post_init__(self) -> None:
        for name in self._TUPLE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, list):
                object.__setattr__(self, name, tuple(value))
        if not isinstance(self.n_gpus, int) or isinstance(self.n_gpus, bool) or self.n_gpus < 1:
            raise ValueError("n_gpus must be an integer >= 1")
        if self.n_gpus == 1 and self.model_device != self.delta_device:
            raise ValueError("with n_gpus=1 model_device and delta_device must coincide")
        for label in ("wall_clock_budget_seconds", "expected_row_seconds", "min_host_ram_gb", "min_free_disk_gb", "g1_threshold", "g1_full_rel_tol", "g4_median_max", "svd_time_budget_s"):
            _positive(getattr(self, label), label)
        for label in ("publish_reserve_seconds", "scoring_overhead_seconds", "g1_full_abs_tol"):
            _positive(getattr(self, label), label, zero_ok=True)
        for label in ("n_conflict_episodes", "n_agreement_episodes", "gate_docs_per_set", "gate_docs_max_tokens", "dolmino_pool_docs", "dolmino_max_docs_per_shard", "dolmino_max_shards", "full_ref_episodes", "repeat_rows", "min_episodes", "trim_granularity", "sequence_length", "gate_sequence_length", "r_star_fallback", "analysis_n_boot"):
            _positive(getattr(self, label), label, integer=True)
        if self.analysis_primary_rank is not None and self.analysis_primary_rank not in self.ranks:
            raise ValueError("analysis_primary_rank must be one of ranks (or null for r*)")
        _positive(self.oracle_rows, "oracle_rows", integer=True, zero_ok=True)
        if self.repeats < 2:
            raise ValueError("repeats must be >= 2 (the noise floor needs pairs)")
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError("arms must be unique and non-empty")
        if not set(self.corpus_arms) <= set(self.arms) or not set(self.g1_arms) <= set(self.arms):
            raise ValueError("corpus_arms and g1_arms must be subsets of arms")
        if not self.ranks or len(set(self.ranks)) != len(self.ranks) or any((not isinstance(r, int)) or r < 1 for r in self.ranks):
            raise ValueError("ranks must be unique positive ints")
        if self.r_star_fallback not in self.ranks:
            raise ValueError("r_star_fallback must be one of ranks")
        if self.svd_method not in ex_mod.SVD_METHODS:
            raise ValueError(f"svd_method must be one of {ex_mod.SVD_METHODS}")
        if self.disk_gate not in ("fail", "warn"):
            raise ValueError("disk_gate must be fail|warn")
        if self.on_g1_full_failure not in ("abort", "continue"):
            raise ValueError("on_g1_full_failure must be abort|continue")
        if "{arm}" not in self.checkpoint_prefix or "{arm}" not in self.corpus_path:
            raise ValueError("checkpoint_prefix and corpus_path must contain '{arm}'")
        for label in ("extract_overrides", "gates_overrides", "score_overrides", "local_snapshots"):
            if not isinstance(getattr(self, label), Mapping):
                raise ValueError(f"{label} must be a mapping")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> DriverConfig:
        if not isinstance(mapping, Mapping):
            raise TypeError("DriverConfig mapping must be a mapping")
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"unknown DriverConfig keys: {unknown}")
        return cls(**dict(mapping))

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        for name in self._TUPLE_FIELDS:
            payload[name] = list(payload[name])
        for name in ("extract_overrides", "gates_overrides", "score_overrides", "local_snapshots"):
            payload[name] = dict(payload[name])
        return payload

    def gpus(self) -> tuple[int, ...]:
        return tuple(range(self.n_gpus))


def load_driver_config(path: str | Path | None) -> DriverConfig:
    return DriverConfig() if path is None else DriverConfig.from_mapping(load_mapping(path))


@dataclass(frozen=True)
class Paths:
    root: Path
    repo_root: Path
    hf_home: Path

    @classmethod
    def from_config(cls, cfg: DriverConfig) -> Paths:
        root = Path(cfg.root)
        return cls(root=root, repo_root=Path(cfg.repo_root), hf_home=Path(cfg.hf_home))

    @property
    def evidence(self) -> Path:
        return self.root / "evidence"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def vector_norms(self) -> Path:
        return self.scores / VECTOR_NORMS_FILE

    @property
    def eft_rows_dir(self) -> Path:
        return self.root / "eft_rows"

    @property
    def eft_rows(self) -> Path:
        return self.eft_rows_dir / "eft_rows.jsonl"

    @property
    def gate_docs(self) -> Path:
        return self.root / "gate_docs"

    @property
    def adapters(self) -> Path:
        return self.root / "adapters"

    @property
    def scores(self) -> Path:
        return self.root / "scores"

    @property
    def staging(self) -> Path:
        return self.root / "staging"

    @property
    def pod_dir(self) -> Path:
        return self.repo_root / "experiments" / "improved_midtraining" / EXPERIMENT_NAME / "pod"

    def script(self, name: str) -> Path:
        return self.pod_dir / name

    def adapter_dir(self, arm: str, rank: int | str) -> Path:
        return self.adapters / arm / ("full" if rank == "full" else f"r{rank}")


# ----------------------------------------------------------------- receipts
def make_receipt(run_id: str, phase: str, status: str, **payload: Any) -> dict[str, Any]:
    if status not in RECEIPT_STATUSES:
        raise ValueError(f"receipt status {status!r} not in {RECEIPT_STATUSES}")
    return {"run_id": run_id, "phase": phase, "status": status, "written_at": utc_now(), **payload}


def validate_receipt(payload: Mapping[str, Any], *, job: bool = False) -> None:
    missing = [k for k in RECEIPT_REQUIRED_KEYS if k not in payload]
    if job:
        missing += [k for k in JOB_RECEIPT_KEYS if k not in payload]
    if missing:
        raise ValueError(f"receipt missing keys {missing}")
    if payload["status"] not in RECEIPT_STATUSES:
        raise ValueError(f"receipt status {payload['status']!r} not in {RECEIPT_STATUSES}")


def validate_done(payload: Mapping[str, Any]) -> None:
    missing = [k for k in DONE_REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"DRIVER_DONE missing keys {missing}")
    if payload["status"] not in DONE_STATUSES:
        raise ValueError(f"DRIVER_DONE status {payload['status']!r} not in {DONE_STATUSES}")


# ------------------------------------------------------- subprocess jobs
@dataclass(frozen=True)
class Job:
    name: str
    argv: tuple[str, ...]
    env: Mapping[str, str]
    gpus: tuple[int, ...]
    log_path: str
    cwd: str
    timeout_seconds: float | None = None
    config_path: str | None = None


@dataclass
class JobResult:
    exit_code: int | None
    seconds: float
    gpu_peak_gb: float | None
    tail: list[str]
    timed_out: bool = False
    started_at: str = ""
    finished_at: str = ""

    @property
    def status(self) -> str:
        if self.timed_out:
            return "timeout"
        if self.exit_code is None:
            return "failed"
        return EXIT_STATUS.get(self.exit_code, "failed")


async def poll_gpu_peak_gb(gpus: Sequence[int], stop: asyncio.Event, interval: float = 5.0) -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    peak: float | None = None
    argv = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if gpus:
        argv += ["-i", ",".join(str(g) for g in gpus)]
    while True:
        try:
            proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            out, _ = await proc.communicate()
            if proc.returncode == 0:
                values = [float(part) for part in out.decode().split() if part.strip()]
                if values:
                    used = max(values) * 1024 * 1024 / GB
                    peak = used if peak is None else max(peak, used)
        except (OSError, ValueError):
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return peak
        except asyncio.TimeoutError:
            continue


async def _terminate_after(proc: Any, seconds: float, flag: dict[str, bool]) -> None:
    await asyncio.sleep(seconds)
    flag["timed_out"] = True
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    await asyncio.sleep(30.0)
    if proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass


async def run_job_subprocess(job: Job, *, echo: bool = True) -> JobResult:
    """Supervised subprocess: merged stdout/stderr tee'd to ``job.log_path``
    (echoed with a ``[name]`` prefix), nvidia-smi peak polling, optional
    timeout (SIGTERM, SIGKILL after 30 s)."""
    started = time.monotonic()
    started_at = utc_now()
    env = dict(os.environ)
    env.update(job.env)
    log_path = Path(job.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop = asyncio.Event()
    poller = asyncio.create_task(poll_gpu_peak_gb(job.gpus, stop))
    tail: deque[str] = deque(maxlen=80)
    flag = {"timed_out": False}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"# [{started_at}] {job.name}: {' '.join(job.argv)}\n# env: " + json.dumps({k: v for k, v in job.env.items() if not k.endswith("TOKEN")}) + "\n")
        handle.flush()
        proc = await asyncio.create_subprocess_exec(*job.argv, cwd=job.cwd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        killer = asyncio.create_task(_terminate_after(proc, job.timeout_seconds, flag)) if job.timeout_seconds is not None else None
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode(errors="replace").rstrip("\n")
            handle.write(text + "\n")
            handle.flush()
            tail.append(text)
            if echo:
                print(f"[{job.name}] {text}", flush=True)
        code = await proc.wait()
        if killer is not None and not killer.done():
            killer.cancel()
    stop.set()
    peak = await poller
    return JobResult(exit_code=code, seconds=time.monotonic() - started, gpu_peak_gb=peak, tail=list(tail), timed_out=flag["timed_out"], started_at=started_at, finished_at=utc_now())


# ------------------------------------------------------------- default deps
def snapshot_download_default(repo_id: str, *, revision: str | None, allow_patterns: Sequence[str] | None, ignore_patterns: Sequence[str] | None, repo_type: str) -> str:
    from huggingface_hub import snapshot_download

    return str(snapshot_download(repo_id, revision=revision, repo_type=repo_type, allow_patterns=list(allow_patterns) if allow_patterns else None, ignore_patterns=list(ignore_patterns) if ignore_patterns else None, token=os.environ.get("HF_TOKEN") or None))


def list_repo_files_default(repo_id: str, *, revision: str | None, repo_type: str) -> list[str]:
    from huggingface_hub import list_repo_files

    return list(list_repo_files(repo_id, revision=revision, repo_type=repo_type, token=os.environ.get("HF_TOKEN") or None))


def hf_hub_download_default(repo_id: str, filename: str, *, revision: str | None, repo_type: str) -> str:
    from huggingface_hub import hf_hub_download

    return str(hf_hub_download(repo_id, filename, revision=revision, repo_type=repo_type, token=os.environ.get("HF_TOKEN") or None))


def write_eft_rows_default(out_path: str, *, n_conflict_episodes: int, n_agreement_episodes: int, seed: int) -> Path:
    from experiments.improved_midtraining.ekfac_dataset_attribution_v1.eft_rows import build_eft_rows

    return build_eft_rows.write_eft_rows(Path(out_path), n_conflict_episodes, n_agreement_episodes, seed)


def sample_corpus_docs_default(corpus_path: str, out_path: str, *, n: int, seed: int, group: str) -> Mapping[str, Any]:
    """256 docs from one arm's directional corpus, seeded, doc_type-stratified
    (v1's sampler; rows keep the corpus metadata)."""
    from experiments.improved_midtraining.ekfac_dataset_attribution_v1.datasets import sample_datasets as sd

    draw = sd.draw_sample(sd.JsonlRows(corpus_path, key=group), n, sd.derive_seed(seed, group), strata_key="doc_type")
    for row in draw.rows:
        row["source"] = group
    record = sd.write_sample(out_path, draw.rows, group=group)
    return {"file": record, "sampling": draw.as_dict(), "corpus_path": corpus_path}


def build_dolmino_docs_default(out_dir: str, *, n: int, seed: int, pool_docs: int, max_docs_per_shard: int, max_shards: int) -> Mapping[str, Any]:
    """v1's Dolmino pins (repo / revision / shard split seed): the scored half
    of the seeded shard permutation, a streamed pool, one seeded draw."""
    from experiments.improved_midtraining.ekfac_dataset_attribution_v1.datasets import sample_datasets as sd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hub = sd.HfHub(out / ".hub_cache", token=sd.hf_token())
    shards = sd.list_dolmino_shards(hub)
    _, scored_shards = sd.split_dolmino_shards(shards, sd.derive_seed(seed, "dolmino_shards"))
    pool = sd.materialize_dolmino_pool(hub, scored_shards, out / "pool.jsonl", min_docs=max(pool_docs, n), max_docs_per_shard=max_docs_per_shard, max_shards=max_shards)
    draw = sd.draw_sample(pool.rows(), n, sd.derive_seed(seed, "dolmino"))
    for row in draw.rows:
        row["source"] = "dolmino"
    record = sd.write_sample(out / "dolmino.jsonl", draw.rows, group="dolmino")
    return {"file": record, "sampling": draw.as_dict(), "pool": pool.as_dict(), "source": {"repo_id": sd.DOLMINO_REPO, "revision": sd.DOLMINO_REVISION, "shard_half": "odd (v1's scored half)"}}


def host_ram_gb_default() -> float:
    from experiments.improved_midtraining.gate2_lineage_attribution.pod import host_probe

    return host_probe.effective_ram_gb(host_probe.read_mem_total_gb(), host_probe.read_cgroup_limit_gb()) * (1024**3) / GB


def disk_free_gb_default(path: str) -> float:
    probe = Path(path)
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free / GB


def upload_folder_hf(staging_dir: str, repo_id: str, path_in_repo: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    files = [p for p in Path(staging_dir).rglob("*") if p.is_file()]
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=staging_dir, path_in_repo=path_in_repo, commit_message=f"{EXPERIMENT_NAME}: {path_in_repo}")
    return {"repo_id": repo_id, "repo_type": "dataset", "path_in_repo": path_in_repo, "url": f"https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}", "n_files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def spawn_evictor_default(python: str, script: str, roots: Sequence[str], log_path: str) -> Any:
    env = dict(os.environ)
    env["EVICT_ROOTS"] = ":".join(roots)
    handle = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 — lives as long as the process
    return subprocess.Popen([python, script], env=env, stdout=handle, stderr=subprocess.STDOUT)


def stop_evictor_default(proc: Any) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class DriverDeps:
    """Every side effect, injectable (tests pass fakes; the pod uses defaults)."""

    run_job: Callable[[Job], Awaitable[JobResult]]
    snapshot_download: Callable[..., str] = snapshot_download_default
    list_repo_files: Callable[..., list[str]] = list_repo_files_default
    hf_hub_download: Callable[..., str] = hf_hub_download_default
    write_eft_rows: Callable[..., Path] = write_eft_rows_default
    sample_corpus_docs: Callable[..., Mapping[str, Any]] = sample_corpus_docs_default
    build_dolmino_docs: Callable[..., Mapping[str, Any]] = build_dolmino_docs_default
    host_ram_gb: Callable[[], float] = host_ram_gb_default
    disk_free_gb: Callable[[str], float] = disk_free_gb_default
    upload_folder: Callable[[str, str, str], Mapping[str, Any]] = upload_folder_hf
    spawn_evictor: Callable[..., Any] = spawn_evictor_default
    stop_evictor: Callable[[Any], None] = stop_evictor_default
    now: Callable[[], float] = time.time


def default_deps(*, echo: bool = True) -> DriverDeps:
    async def run_job(job: Job) -> JobResult:
        return await run_job_subprocess(job, echo=echo)

    return DriverDeps(run_job=run_job)


# --------------------------------------------------------------- planning
class Deadline:
    def __init__(self, started: float, budget_seconds: float, reserve_seconds: float, now: Callable[[], float]) -> None:
        self.started = started
        self.budget = float(budget_seconds)
        self.reserve = float(reserve_seconds)
        self._now = now

    def elapsed(self) -> float:
        return self._now() - self.started

    def remaining_for_work(self) -> float:
        return self.budget - self.elapsed() - self.reserve

    def to_dict(self) -> dict[str, Any]:
        return {"budget_seconds": self.budget, "reserve_seconds": self.reserve, "elapsed_seconds": self.elapsed(), "remaining_for_work_seconds": self.remaining_for_work()}


@dataclass(frozen=True)
class RowCounts:
    conflict_episodes: int
    agreement_episodes: int

    def rows_for(self, rows_filter: str | int) -> int:
        if rows_filter == "all":
            return 2 * (self.conflict_episodes + self.agreement_episodes)
        n = int(rows_filter)
        return 2 * (min(n, self.conflict_episodes) + min(n, self.agreement_episodes))

    @property
    def max_episodes(self) -> int:
        return max(self.conflict_episodes, self.agreement_episodes)


def episodes_for_budget(budget_seconds: float, s_per_row: float, counts: RowCounts, *, granularity: int, minimum: int) -> int | None:
    """Largest episodes-per-pair-type (multiple of ``granularity``, >= minimum)
    whose rows fit the budget; None when even ``minimum`` does not."""
    if s_per_row <= 0:
        raise ValueError("s_per_row must be positive")
    top = int(math.ceil(counts.max_episodes / granularity)) * granularity
    n = top
    while n >= minimum:
        if counts.rows_for(n) * s_per_row <= budget_seconds:
            return n
        n -= granularity
    if counts.rows_for(minimum) * s_per_row <= budget_seconds:
        return minimum
    return None


def plan_rows(requested: str | int, budget_seconds: float, s_per_row: float, overhead_seconds: float, counts: RowCounts, *, granularity: int, minimum: int) -> dict[str, Any]:
    """Trim ``requested`` (``all`` or episodes per pair type) to the budget."""
    available = budget_seconds - overhead_seconds
    full_rows = counts.rows_for(requested)
    if full_rows * s_per_row <= available:
        return {"rows_filter": requested, "n_rows": full_rows, "trimmed": False, "projected_seconds": overhead_seconds + full_rows * s_per_row, "budget_seconds": budget_seconds}
    cap = counts.max_episodes if requested == "all" else int(requested)
    n = episodes_for_budget(available, s_per_row, counts, granularity=granularity, minimum=minimum)
    if n is None:
        return {"rows_filter": None, "n_rows": 0, "trimmed": True, "skip": True, "projected_seconds": None, "budget_seconds": budget_seconds}
    n = min(n, cap)
    return {"rows_filter": n, "n_rows": counts.rows_for(n), "trimmed": True, "skip": False, "projected_seconds": overhead_seconds + counts.rows_for(n) * s_per_row, "budget_seconds": budget_seconds}


# ------------------------------------------------------------ gate maths
def rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = rankdata(x), rankdata(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if vx == 0 or vy == 0 else cov / (vx * vy)


def slope_through_origin(x: Sequence[float], y: Sequence[float]) -> float | None:
    """OLS slope of y on x (no intercept): the LoRA-route / full-route scale."""
    sxx = sum(a * a for a in x)
    return None if sxx == 0 else sum(a * b for a, b in zip(x, y, strict=True)) / sxx


def merge_full_scores(lam0_path: Path, full_paths: Sequence[Path]) -> dict[str, Any]:
    """Fold the full-delta reference scores (``<arm>__lam0_full__all``, scored
    on the row subset in their own passes) into ``scores/lam0.jsonl`` so the
    λ = 0 file carries every λ = 0 kind (analysis contract). Rows are matched
    on ``row_id``; names already present in lam0 (the r1024 companion) are
    never overwritten; the file is rewritten atomically."""
    records = sc_mod.read_records(lam0_path)
    by_id = {str(r["row_id"]): r for r in records if r.get("repeat") is None}
    merged_names: set[str] = set()
    matched = unmatched = 0
    for path in full_paths:
        for record in sc_mod.read_records(path):
            target = by_id.get(str(record["row_id"]))
            if target is None:
                unmatched += 1
                continue
            for name, value in record.get("scores", {}).items():
                if name in target["scores"]:
                    continue
                target["scores"][name] = value
                target.setdefault("raw_dl_dlambda", {})[name] = record.get("raw_dl_dlambda", {}).get(name, -value)
                merged_names.add(name)
            matched += 1
    tmp = lam0_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    tmp.replace(lam0_path)
    return {"lam0_rows": len(records), "matched": matched, "unmatched": unmatched, "merged_names": sorted(merged_names), "sources": [str(p) for p in full_paths]}


def vector_norms_from_manifests(adapters_dir: Path, arms: Sequence[str], ranks: Sequence[int], r_star: int | None, *, lam1_full: bool = False) -> dict[str, float]:
    """``{"<arm>__<kind>__all": ||delta_kind||_F}`` (rank-r reconstruction of the
    linears + full-rank norm deltas; ``full`` = the whole delta) from the arm
    manifests' ``delta_norms``; λ = 1 kinds alias the same tensors (``lam1_full``
    adds the exact-graft kinds ``lam1full`` / ``lam1full_r<r_max>`` / ``lam1fullx_r<r_max>``)."""
    norms: dict[str, float] = {}
    r_max = max(ranks) if ranks else None
    for arm in arms:
        manifest_path = adapters_dir / arm / MANIFEST_FILE
        if not manifest_path.is_file():
            continue
        delta_norms = read_json(manifest_path).get("delta_norms") or {}
        for r in ranks:
            value = delta_norms.get(f"r{r}")
            if value is not None:
                norms[delta_name(arm, f"lam0_r{r}")] = float(value)
                if r_star == r:
                    norms[delta_name(arm, f"lam1_r{r}")] = float(value)
                    norms[delta_name(arm, f"lam1x_r{r}")] = float(value)
                if lam1_full and r == r_max:
                    norms[delta_name(arm, f"lam1full_r{r}")] = float(value)
                    norms[delta_name(arm, f"lam1fullx_r{r}")] = float(value)
        if delta_norms.get("full") is not None:
            norms[delta_name(arm, "lam0_full")] = float(delta_norms["full"])
            if lam1_full:
                norms[delta_name(arm, "lam1full")] = float(delta_norms["full"])
    return norms


def combine_gates(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One per-arm record per arm from the G1 and G2 files (union of keys)."""
    arms: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for arm, record in (payload.get("arms") or {}).items():
            if isinstance(record, Mapping):
                arms.setdefault(arm, {}).update(record)
    return {"schema": gates_mod.RESULTS_SCHEMA, "combined_from": [p.get("name") for p in payloads], "arms": arms, "created_at": utc_now()}


def g3_exactness(records: Sequence[Mapping[str, Any]], lora_name: str, full_name: str) -> dict[str, Any]:
    """Spearman + slope of the LoRA-route ``-g`` (x) vs the full-delta ``-g``
    (y) over rows carrying both scores (same backward pass)."""
    xs, ys = [], []
    for record in records:
        scores = record.get("scores", {})
        if lora_name in scores and full_name in scores:
            xs.append(float(scores[lora_name]))
            ys.append(float(scores[full_name]))
    return {"n": len(xs), "spearman": spearman(xs, ys), "slope_full_on_lora": slope_through_origin(xs, ys), "lora": lora_name, "full": full_name}


# ---------------------------------------------------------------- logging
class DriverLog:
    def __init__(self, path: Path, *, echo: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo

    def __call__(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if self.echo:
            print(line, flush=True)


class GateFailure(RuntimeError):
    """A pre-registered gate refused to continue (fatal; publish still runs)."""


@dataclass
class RunState:
    phases: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    gates: dict[str, Any] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)
    trims: list[dict[str, Any]] = field(default_factory=list)
    deadline_hit: bool = False
    inputs: dict[str, Any] = field(default_factory=dict)
    r_star: int | None = None
    r_star_record: dict[str, Any] = field(default_factory=dict)
    publication: dict[str, Any] = field(default_factory=dict)
    preflight: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------- staging
def stage_for_upload(paths: Paths, staging_dir: Path, *, upload_adapters: bool = False) -> dict[str, Any]:
    """Copy evidence, scores, rows, gate docs and adapter sidecars into
    ``staging_dir`` — never weights (unless ``upload_adapters``), pools or caches."""
    sidecars = (MANIFEST_FILE, "adapter_config.json", "delta.index.json")
    rules: list[tuple[Path, str, Callable[[Path], bool]]] = [
        (paths.evidence, "evidence", lambda p: True),
        (paths.results, "results", lambda p: True),
        (paths.scores, "scores", lambda p: p.suffix in (".jsonl", ".json", ".log")),
        (paths.eft_rows_dir, "eft_rows", lambda p: p.suffix in (".jsonl", ".json")),
        (paths.gate_docs, "gate_docs", lambda p: p.suffix in (".jsonl", ".json")),
        (paths.adapters, "adapters", lambda p: p.name in sidecars or p.suffix in (".json", ".log") or (upload_adapters and p.suffix == ".safetensors")),
    ]
    copied: list[dict[str, Any]] = []
    total = 0
    for source, remote, allow in rules:
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if any(part in UPLOAD_DENY_DIRS for part in relative.parts) or path.name in UPLOAD_DENY_NAMES:
                continue
            if path.suffix in UPLOAD_DENY_SUFFIXES and not (upload_adapters and path.suffix == ".safetensors" and remote == "adapters"):
                continue
            if not allow(path):
                continue
            target = staging_dir / remote / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            size = target.stat().st_size
            total += size
            copied.append({"path": str(target.relative_to(staging_dir)), "bytes": size})
    return {"staging_dir": str(staging_dir), "n_files": len(copied), "bytes": total, "files": copied}


# ------------------------------------------------------------------ driver
class Driver:
    def __init__(self, cfg: DriverConfig, deps: DriverDeps | None = None, *, paths: Paths | None = None) -> None:
        self.cfg = cfg
        self.deps = default_deps(echo=cfg.echo_subprocess_output) if deps is None else deps
        self.paths = Paths.from_config(cfg) if paths is None else paths
        self.evidence = self.paths.evidence
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.log = DriverLog(self.evidence / LOG_FILE, echo=cfg.echo_subprocess_output)
        self.state = RunState()
        self.python = cfg.python or sys.executable
        self.run_id, started = self._anchor()
        self.deadline = Deadline(started, cfg.wall_clock_budget_seconds, cfg.publish_reserve_seconds, self.deps.now)
        self.evictor: Any = None

    # ---- bookkeeping -----------------------------------------------------
    def _anchor(self) -> tuple[str, float]:
        path = self.evidence / STARTED_FILE
        if self.cfg.resume and path.is_file():
            previous = read_json(path)
            if not self.cfg.run_id or previous.get("run_id") == self.cfg.run_id:
                return str(previous["run_id"]), float(previous["started_epoch"])
        started = self.deps.now()
        run_id = self.cfg.run_id or timestamp_run_id(started)
        write_json(path, {"run_id": run_id, "started_epoch": started, "started_at": utc_now(), "config": self.cfg.to_dict()})
        return run_id, started

    def receipt(self, phase: str, status: str, **payload: Any) -> dict[str, Any]:
        body = make_receipt(self.run_id, phase, status, **payload)
        write_json(self.evidence / f"{phase}.json", body)
        self.log(f"{PHASE_SENTINEL} {phase} status={status}")
        return body

    def receipt_ok(self, phase: str) -> dict[str, Any] | None:
        path = self.evidence / f"{phase}.json"
        if not path.is_file():
            return None
        try:
            body = read_json(path)
        except (OSError, ValueError):
            return None
        return body if body.get("status") == "ok" and body.get("run_id") == self.run_id else None

    def note(self, message: str) -> None:
        self.state.notes.append(message)
        self.log(f"note: {message}")

    def skip(self, phase: str, reason: str, *, deliberate: bool = False) -> None:
        self.state.phases[phase] = "skipped"
        self.state.skipped.append({"phase": phase, "reason": reason, "deliberate": deliberate})
        self.receipt(phase, "skipped", reason=reason, deliberate=deliberate)

    def fail(self, phase: str, reason: str) -> None:
        self.state.phases[phase] = "failed"
        self.state.failures.append({"phase": phase, "reason": reason})
        self.log(f"{FAIL_SENTINEL} {phase}: {reason}")

    def gate(self, name: str, passed: bool | None, **payload: Any) -> dict[str, Any]:
        verdict = {"passed": passed, **payload}
        self.state.gates[name] = verdict
        self.receipt(f"gate_{name.lower()}", "ok" if passed else ("gate-failed" if passed is False else "partial"), **verdict)
        self.log(f"{GATE_SENTINEL} {name} {'PASS' if passed else ('FAIL' if passed is False else 'INCONCLUSIVE')}: {payload.get('summary', '')}")
        return verdict

    def trim(self, job: str, plan: Mapping[str, Any], requested: Any) -> None:
        record = {"job": job, "requested": requested, **{k: v for k, v in plan.items()}}
        self.state.trims.append(record)
        self.state.deadline_hit = True
        self.log(f"{TRIM_SENTINEL} {job}: requested {requested} -> rows_filter {plan.get('rows_filter')} ({plan.get('n_rows')} rows; budget {plan.get('budget_seconds', 0):.0f}s)")

    def config_path(self, name: str) -> Path:
        return self.evidence / "configs" / f"{name}.json"

    def job(self, name: str, argv: Sequence[str], *, gpus: Sequence[int], config: Mapping[str, Any] | None = None, config_env: str | None = None, timeout_seconds: float | None = None, offline: bool = True, env: Mapping[str, str] | None = None) -> Job:
        job_env: dict[str, str] = {
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
            "SCIMT_RUN_ID": self.run_id,
            "HF_HOME": str(self.paths.hf_home),
        }
        if gpus:
            job_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        if offline:
            job_env["HF_HUB_OFFLINE"] = "1"
        config_path: str | None = None
        argv = list(argv)
        if config is not None:
            config_path = str(write_json(self.config_path(name), config))
            if config_env:
                job_env[config_env] = config_path
            else:
                argv.append(config_path)
        if env:
            job_env.update(env)
        return Job(name=name, argv=tuple(argv), env=job_env, gpus=tuple(gpus), log_path=str(self.evidence / f"{name}.log"), cwd=str(self.paths.repo_root), timeout_seconds=timeout_seconds, config_path=config_path)

    async def run(self, job: Job, **extra: Any) -> JobResult:
        self.log(f"launch {job.name}: gpus={list(job.gpus)} argv={' '.join(job.argv)} timeout={job.timeout_seconds}")
        result = await self.deps.run_job(job)
        self.receipt(job.name, result.status, exit_code=result.exit_code, seconds=result.seconds, gpu_peak_gb=result.gpu_peak_gb, gpus=list(job.gpus), argv=list(job.argv), log_path=job.log_path, config_path=job.config_path, tail=result.tail[-25:], started_at=result.started_at, finished_at=result.finished_at, timed_out=result.timed_out, **extra)
        if result.status != "ok":
            self.log(f"{job.name} exited {result.exit_code} ({result.status}); tail:\n" + "\n".join(result.tail[-8:]))
        return result

    def newest_receipt(self, directory: Path, prefix: str) -> dict[str, Any] | None:
        candidates = sorted(directory.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime) if directory.is_dir() else []
        for path in reversed(candidates):
            try:
                return read_json(path)
            except (OSError, ValueError):
                continue
        return None

    # ---- inputs -------------------------------------------------------------
    def snapshot(self, key: str) -> str:
        value = self.state.inputs.get("snapshots", {}).get(key)
        if not value:
            raise RuntimeError(f"input snapshot {key!r} unknown — preflight did not record it")
        return str(value)

    @property
    def row_counts(self) -> RowCounts:
        groups = self.state.inputs.get("eft_rows", {}).get("groups", {})
        conflict = int(groups.get("coin", {}).get("episodes", 0))
        agreement = int(groups.get("ambiguous", {}).get("episodes", 0))
        if conflict == 0 and agreement == 0:
            raise RuntimeError("EFT rows manifest lacks group counts")
        return RowCounts(conflict, agreement)

    @property
    def s_per_row(self) -> float:
        return float(self.state.measurements.get("row_seconds", self.cfg.expected_row_seconds))

    # ---- phase 0 ----------------------------------------------------------------
    async def phase0_preflight(self) -> None:
        phase = "preflight"
        inputs_path = self.evidence / INPUTS_FILE
        if (prior := self.receipt_ok(phase)) is not None and inputs_path.is_file():
            self.state.inputs = read_json(inputs_path)
            self.state.preflight = prior.get("probe", {})
            self.state.phases[phase] = "ok"
            self.log(f"{phase}: already ok for run {self.run_id}; inputs from {inputs_path}")
            return
        cfg = self.cfg
        self.paths.hf_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.paths.hf_home))
        # host
        ram = self.deps.host_ram_gb()
        free_disk = self.deps.disk_free_gb(str(self.paths.root))
        host = {"ram_gb": ram, "free_disk_gb": free_disk}
        if ram < cfg.min_host_ram_gb:
            raise GateFailure(f"host RAM {ram:.0f} GB < {cfg.min_host_ram_gb:.0f} GB (two 54 GB shard sets + factors need headroom)")
        if free_disk < cfg.min_free_disk_gb:
            message = f"free disk {free_disk:.0f} GB < {cfg.min_free_disk_gb:.0f} GB (5 x 54 GB snapshots + 3 x 54 GB full deltas + ~60 GB adapters)"
            if cfg.disk_gate == "fail":
                raise GateFailure(message)
            self.note(message)
        # torch / cuda probe
        result = await self.run(self.job("preflight_probe", [self.python, "-c", PREFLIGHT_SNIPPET], gpus=cfg.gpus(), offline=True))
        if result.status != "ok":
            raise GateFailure(f"preflight probe failed (exit {result.exit_code})")
        probe = self._parse_preflight(result.tail)
        if not probe.get("cuda_available"):
            raise GateFailure(f"torch.cuda.is_available() is False (torch {probe.get('torch')} cuda {probe.get('torch_cuda')}): cu128 wheel trap — see README")
        if int(probe.get("cuda_devices", 0)) < cfg.n_gpus:
            raise GateFailure(f"{probe.get('cuda_devices')} CUDA devices visible, need {cfg.n_gpus}")
        if not all(g.get("matmul_ok") for g in probe.get("gpus", [])):
            raise GateFailure(f"per-GPU matmul check failed: {probe.get('gpus')}")
        self.state.preflight = probe
        # snapshots
        t0 = self.deps.now()
        snapshots, listing_summary = self._resolve_snapshots()
        self.log(f"snapshots ready in {self.deps.now() - t0:.0f}s: {json.dumps(snapshots)}")
        # eft rows
        t0 = self.deps.now()
        rows_path = self.deps.write_eft_rows(str(self.paths.eft_rows), n_conflict_episodes=cfg.n_conflict_episodes, n_agreement_episodes=cfg.n_agreement_episodes, seed=cfg.eft_seed)
        rows_manifest = read_json(Path(rows_path).parent / "manifest.json")
        n_rows = int(rows_manifest.get("n_rows", 0))
        if n_rows < 4:
            raise GateFailure(f"EFT rows: only {n_rows} rows written")
        self.log(f"EFT rows: {n_rows} rows ({json.dumps({g: v.get('episodes') for g, v in rows_manifest.get('groups', {}).items()})}) in {self.deps.now() - t0:.0f}s")
        # gate docs
        t0 = self.deps.now()
        gate_docs = self._build_gate_docs(snapshots)
        self.log(f"gate docs: {json.dumps({k: v['file']['n'] for k, v in gate_docs.items()})} in {self.deps.now() - t0:.0f}s")
        self.state.inputs = {"snapshots": snapshots, "listing": listing_summary, "eft_rows": {"path": str(rows_path), **rows_manifest}, "gate_docs": gate_docs, "host": host, "probe": probe}
        write_json(inputs_path, self.state.inputs)
        self.receipt(phase, "ok", host=host, probe=probe, snapshots=snapshots, eft_rows={"path": str(rows_path), "n_rows": n_rows}, gate_docs={k: v["file"]["n"] for k, v in gate_docs.items()})
        self.state.phases[phase] = "ok"

    @staticmethod
    def _parse_preflight(tail: Sequence[str]) -> dict[str, Any]:
        for line in reversed(list(tail)):
            if line.startswith(PREFLIGHT_PREFIX):
                return json.loads(line[len(PREFLIGHT_PREFIX):])
        raise GateFailure("preflight probe printed no SCIMT-PREFLIGHT line")

    def _bootstrap_snapshots(self) -> dict[str, str]:
        """Snapshot dirs recorded by ops/bootstrap_pod.sh (``models.json``:
        ``pt`` / ``it`` / ``mid_<arm>`` -> {"path": ...}); missing or invalid
        entries fall through to a download."""
        path = Path(self.cfg.models_json) if self.cfg.models_json else self.evidence / MODELS_FILE
        if not path.is_file():
            return {}
        try:
            body = read_json(path)
        except (OSError, ValueError) as error:
            self.note(f"{path} unreadable ({error!r}); downloading snapshots instead")
            return {}
        found: dict[str, str] = {}
        for key, record in body.items():
            directory = record.get("path") if isinstance(record, Mapping) else None
            if not directory or not Path(directory).is_dir() or not any(Path(directory).glob("*.safetensors")):
                continue
            if key in ("pt", "it"):
                found[key] = str(directory)
            elif key.startswith("mid_"):
                found[f"mid:{key[len('mid_'):]}"] = str(directory)
            elif key.startswith("corpus_") and Path(directory).is_file():
                found[f"corpus:{key[len('corpus_'):]}"] = str(directory)
        if found:
            self.log(f"snapshots from {path}: {sorted(found)}")
        return found

    def _resolve_snapshots(self) -> tuple[dict[str, str], dict[str, Any]]:
        cfg = self.cfg
        local = self._bootstrap_snapshots()
        local.update(dict(cfg.local_snapshots))  # explicit config wins
        snapshots: dict[str, str] = {}
        listing: list[str] | None = None
        listing_summary: dict[str, Any] = {}
        need_listing = any(f"mid:{arm}" not in local for arm in cfg.arms) or any(f"corpus:{arm}" not in local for arm in cfg.corpus_arms)
        if need_listing:
            listing = self.deps.list_repo_files(cfg.checkpoint_repo, revision=cfg.checkpoint_revision, repo_type=cfg.checkpoint_repo_type)
            listing_summary = {"repo": cfg.checkpoint_repo, "revision": cfg.checkpoint_revision, "n_files": len(listing)}
            missing = [arm for arm in cfg.arms if f"mid:{arm}" not in local and not any(f.startswith(cfg.checkpoint_prefix.format(arm=arm) + "/") and f.endswith(".safetensors") for f in listing)]
            if missing:
                candidates = sorted({f.rsplit("/", 1)[0] for f in listing if "checkpoint-" in f and f.endswith(".safetensors")})
                raise GateFailure(
                    f"checkpoint prefix {cfg.checkpoint_prefix!r} has no *.safetensors in {cfg.checkpoint_repo}@{cfg.checkpoint_revision or 'main'} for arms {missing}; "
                    f"checkpoint dirs present ({len(candidates)}): {candidates[:12]}{' ...' if len(candidates) > 12 else ''} — set checkpoint_prefix / checkpoint_repo / local_snapshots"
                )
        for key, hf_id, revision in (("pt", cfg.pt_hf_id, cfg.pt_revision), ("it", cfg.it_hf_id, cfg.it_revision)):
            if key in local:
                snapshots[key] = local[key]
            else:
                self.log(f"snapshot_download {hf_id}@{revision or 'main'}")
                snapshots[key] = self.deps.snapshot_download(hf_id, revision=revision, allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.jinja"], ignore_patterns=None, repo_type="model")
            if cfg.pt_revision and key == "pt" and cfg.pt_revision not in snapshots[key] and key not in local:
                raise GateFailure(f"pt snapshot path {snapshots[key]} does not carry the pinned sha {cfg.pt_revision}")
        for arm in cfg.arms:
            key = f"mid:{arm}"
            if key in local:
                snapshots[key] = local[key]
                continue
            prefix = cfg.checkpoint_prefix.format(arm=arm)
            self.log(f"snapshot_download {cfg.checkpoint_repo}:{prefix}")
            root = self.deps.snapshot_download(cfg.checkpoint_repo, revision=cfg.checkpoint_revision, allow_patterns=[f"{prefix}/*.safetensors", f"{prefix}/*.json", f"{prefix}/*.model", f"{prefix}/*.txt", f"{prefix}/*.jinja"], ignore_patterns=[f"{prefix}/optimizer*", f"{prefix}/rng_state*", f"{prefix}/scheduler*", f"{prefix}/*.pt", f"{prefix}/*.bin"], repo_type=cfg.checkpoint_repo_type)
            snapshots[key] = str(Path(root) / prefix)
        for key, directory in snapshots.items():
            path = Path(directory)
            if not path.is_dir() or not any(path.glob("*.safetensors")) or not (path / "config.json").is_file():
                raise GateFailure(f"snapshot {key} at {path} lacks *.safetensors or config.json")
        for arm in cfg.corpus_arms:
            key = f"corpus:{arm}"
            if key in local:
                snapshots[key] = local[key]
                continue
            corpus_path = cfg.corpus_path.format(arm=arm)
            assert listing is not None
            if corpus_path not in listing:
                top = cfg.checkpoint_prefix.format(arm=arm).split("/")[0]
                fallback = sorted(f for f in listing if f.startswith(top + "/") and f.endswith(f"/release/{arm}/corpus.jsonl"))
                if not fallback:
                    fallback = sorted(f for f in listing if f.endswith(f"/release/{arm}/corpus.jsonl"))
                if not fallback:
                    raise GateFailure(f"no corpus for arm {arm}: {corpus_path} not in the listing and no */release/{arm}/corpus.jsonl fallback")
                self.note(f"corpus path {corpus_path} missing; using listed {fallback[0]}")
                corpus_path = fallback[0]
            snapshots[key] = self.deps.hf_hub_download(cfg.checkpoint_repo, corpus_path, revision=cfg.checkpoint_revision, repo_type=cfg.checkpoint_repo_type)
        return snapshots, listing_summary

    def _build_gate_docs(self, snapshots: Mapping[str, str]) -> dict[str, Any]:
        cfg = self.cfg
        out: dict[str, Any] = {}
        self.paths.gate_docs.mkdir(parents=True, exist_ok=True)
        for arm in cfg.corpus_arms:
            path = self.paths.gate_docs / f"{arm}.jsonl"
            record = dict(self.deps.sample_corpus_docs(snapshots[f"corpus:{arm}"], str(path), n=cfg.gate_docs_per_set, seed=cfg.gate_docs_seed, group=arm))
            record["path"] = str(path)
            out[arm] = record
        record = dict(self.deps.build_dolmino_docs(str(self.paths.gate_docs / "dolmino"), n=cfg.gate_docs_per_set, seed=cfg.gate_docs_seed, pool_docs=cfg.dolmino_pool_docs, max_docs_per_shard=cfg.dolmino_max_docs_per_shard, max_shards=cfg.dolmino_max_shards))
        record["path"] = str(self.paths.gate_docs / "dolmino" / "dolmino.jsonl")
        out["dolmino"] = record
        write_json(self.paths.gate_docs / "manifest.json", {"seed": cfg.gate_docs_seed, "n_per_set": cfg.gate_docs_per_set, "sets": out, "created_at": utc_now()})
        return out

    # ---- phase 1 ------------------------------------------------------------------
    def render_extract_config(self, arm: str) -> dict[str, Any]:
        cfg = self.cfg
        base = {
            "pt_snapshot": self.snapshot("pt"),
            "mid_snapshot": self.snapshot(f"mid:{arm}"),
            "arm": arm,
            "out_dir": str(self.paths.adapters),
            "ranks": list(cfg.ranks),
            "device": "cuda:0",
            "svd_method": cfg.svd_method,
            "svd_time_budget_s": cfg.svd_time_budget_s,
            "save_full": cfg.save_full,
            "base_model_name": cfg.pt_hf_id,
            "evidence_dir": str(self.paths.adapters / "evidence"),
        }
        base.update(dict(cfg.extract_overrides))
        ex_mod.ExtractConfig.from_mapping(base)
        return base

    async def phase1_extract(self) -> None:
        phase = "extract"
        if self.receipt_ok(phase) is not None:
            self.state.phases[phase] = "ok"
            self.log(f"{phase}: already ok")
            return
        self._start_evictor()
        per_arm: dict[str, Any] = {}
        for arm in self.cfg.arms:
            name = f"extract__{arm}"
            if self.receipt_ok(name) is not None and ex_mod.arm_complete(self.paths.adapters / arm, self.cfg.ranks, self.cfg.save_full):
                per_arm[arm] = "ok (resumed)"
                self.log(f"{name}: already complete")
                continue
            result = await self.run(self.job(name, [self.python, str(self.paths.script("extract_delta_lora.py"))], gpus=self.cfg.gpus()[:1], config=self.render_extract_config(arm), config_env=ex_mod.CONFIG_ENV))
            if result.status != "ok" or not ex_mod.arm_complete(self.paths.adapters / arm, self.cfg.ranks, self.cfg.save_full):
                self.fail(phase, f"{name} exited {result.exit_code} ({result.status})")
                self.receipt(phase, "failed", arms=per_arm, failed=arm)
                raise GateFailure(f"extraction failed for {arm}: {result.status}")
            per_arm[arm] = "ok"
            manifest = read_json(self.paths.adapters / arm / MANIFEST_FILE)
            self.state.measurements.setdefault("extract", {})[arm] = {"captured_energy": manifest.get("captured_energy", manifest.get("energy_pooled")), "delta_norms": manifest.get("delta_norms"), "delta_fro_linears": manifest.get("delta_fro_linears"), "seconds": result.seconds}
        norms = self.write_vector_norms()
        self.receipt(phase, "ok", arms=per_arm, energy={arm: v.get("captured_energy") for arm, v in self.state.measurements.get("extract", {}).items()}, vector_norms=norms)
        self.state.phases[phase] = "ok"

    def write_vector_norms(self) -> dict[str, float]:
        norms = vector_norms_from_manifests(self.paths.adapters, self.cfg.arms, self.cfg.ranks, self.state.r_star, lam1_full=self.cfg.lam1_full_reference)
        if norms:
            self.paths.scores.mkdir(parents=True, exist_ok=True)
            write_json(self.paths.vector_norms, norms)
        return norms

    def _start_evictor(self) -> None:
        if self.evictor is not None:
            return
        try:
            self.evictor = self.deps.spawn_evictor(self.python, str(self.paths.script("cache_evictor.py")), [str(self.paths.root), str(self.paths.hf_home)], str(self.evidence / "cache_evictor.log"))
            self.log("cache_evictor started")
        except Exception as error:  # noqa: BLE001 — telemetry helper, never fatal
            self.note(f"cache_evictor not started: {error!r}")

    # ---- phase 2 -------------------------------------------------------------------
    def _docs_config(self) -> tuple[list[dict[str, str]], dict[str, str]]:
        docs = [{"name": name, "path": record["path"]} for name, record in self.state.inputs["gate_docs"].items()]
        arm_docs = {arm: (arm if arm in self.state.inputs["gate_docs"] else "dolmino") for arm in self.cfg.arms}
        return docs, arm_docs

    def render_gates_config(self, name: str, *, pt_ranks: Sequence[int], it_ranks: Sequence[int]) -> dict[str, Any]:
        cfg = self.cfg
        docs, arm_docs = self._docs_config()
        base = {
            "name": name,
            "out_path": str(self.evidence / f"gates__{name}.json"),
            "docs": docs,
            "arms": list(cfg.arms),
            "arm_docs": arm_docs,
            "pt_snapshot": self.snapshot("pt"),
            "it_snapshot": self.snapshot("it"),
            "mid_snapshots": {arm: self.snapshot(f"mid:{arm}") for arm in cfg.arms},
            "adapters": {arm: {f"r{r}": str(self.paths.adapter_dir(arm, r)) for r in cfg.ranks} for arm in cfg.arms},
            "full_deltas": {arm: str(self.paths.adapter_dir(arm, "full")) for arm in cfg.arms} if cfg.save_full else {},
            "pt_ranks": list(pt_ranks),
            "it_ranks": list(it_ranks),
            "eval_pt": bool(pt_ranks) or name == "g1",
            "eval_mid": name == "g1",
            "eval_it": bool(it_ranks),
            "eval_full": cfg.save_full and name == "g1",
            "tokenizer_snapshot": self.snapshot("it"),
            "device": "cuda:0",
            "sequence_length": cfg.gate_sequence_length,
            "g1_threshold": cfg.g1_threshold,
            "g1_full_rel_tol": cfg.g1_full_rel_tol,
            "g1_full_abs_tol": cfg.g1_full_abs_tol,
            "evidence_dir": str(self.evidence / "gates"),
        }
        base.update(dict(cfg.gates_overrides))
        gates_mod.GatesConfig.from_mapping(base)
        return base

    async def phase2_gates(self) -> None:
        phase = "gates"
        cfg = self.cfg
        if (prior := self.receipt_ok(phase)) is not None:
            self.state.phases[phase] = "ok"
            self.state.r_star = int(prior["r_star"])
            self.state.r_star_record = dict(prior.get("r_star_record", {}))
            self.log(f"{phase}: already ok (r* = {self.state.r_star})")
            return
        # G1: pt + ranks + full vs mid
        g1_path = self.evidence / "gates__g1.json"
        if self.receipt_ok("gates_g1") is None or not g1_path.is_file():
            result = await self.run(self.job("gates_g1", [self.python, str(self.paths.script("gates.py"))], gpus=cfg.gpus()[:1], config=self.render_gates_config("g1", pt_ranks=cfg.ranks, it_ranks=()), config_env=gates_mod.CONFIG_ENV))
            if result.status == "pair-mismatch" and cfg.on_g1_full_failure == "abort":
                verdicts = read_json(g1_path).get("verdicts", {}) if g1_path.is_file() else {}
                self.gate("G1_FULL", False, summary=f"full delta does not reproduce mid for {verdicts.get('g1_full_failed_arms')}", failed_arms=verdicts.get("g1_full_failed_arms"), g1_full=verdicts.get("g1_full"))
                self.receipt(phase, "gate-failed", reason="G1-FULL: checkpoint pair mismatched")
                raise GateFailure("G1-FULL failed: the full delta does not reproduce theta_mid — checkpoint pair mismatched (SPEC §5)")
            if result.status not in ("ok", "pair-mismatch"):
                self.receipt(phase, "failed", reason=f"gates_g1 exited {result.exit_code}")
                raise GateFailure(f"gates_g1 failed ({result.status})")
        g1 = read_json(g1_path)
        verdicts = g1.get("verdicts", {})
        g1_full_failed = verdicts.get("g1_full_failed_arms", [])
        self.gate("G1_FULL", not g1_full_failed if verdicts.get("g1_full") else None, summary=("full delta reproduces mid on every arm/doc set" if not g1_full_failed else f"failed for {g1_full_failed} (continuing by config)"), g1_full=verdicts.get("g1_full"))
        choice = gates_mod.choose_r_star(verdicts.get("g1", {}), arms=cfg.g1_arms, ranks=cfg.ranks, threshold=cfg.g1_threshold, fallback=cfg.r_star_fallback)
        self.state.r_star = int(choice["r_star"])
        self.state.r_star_record = choice
        write_json(self.evidence / "r_star.json", {"run_id": self.run_id, **choice})
        self.gate("G1", bool(choice["passed"]), summary=f"r* = {choice['r_star']} ({'smallest rank recovering >= ' + str(cfg.g1_threshold) if choice['passed'] else 'FALLBACK: ' + choice.get('note', '')}); recovered {json.dumps(choice['recovered'])}", r_star=choice["r_star"], recovered=choice["recovered"], g1=verdicts.get("g1"))
        # G2: it + r*
        g2_path = self.evidence / "gates__g2.json"
        if self.receipt_ok("gates_g2") is None or not g2_path.is_file():
            result = await self.run(self.job("gates_g2", [self.python, str(self.paths.script("gates.py"))], gpus=cfg.gpus()[:1], config=self.render_gates_config("g2", pt_ranks=(), it_ranks=(self.state.r_star,)), config_env=gates_mod.CONFIG_ENV))
            if result.status != "ok":
                self.receipt(phase, "failed", reason=f"gates_g2 exited {result.exit_code}", r_star=self.state.r_star, r_star_record=choice)
                raise GateFailure(f"gates_g2 failed ({result.status})")
        g2_payload = read_json(g2_path)
        g2 = g2_payload.get("verdicts", {}).get("g2", {})
        g2_own = {arm: next((e for e in g2.get(arm, {}).get(f"r{self.state.r_star}", {}).values() if e.get("own_docs")), None) for arm in cfg.g1_arms}
        g2_passed = all(e is not None and e.get("passed") for e in g2_own.values())
        self.gate("G2", g2_passed, summary=f"L(it+delta_r*) {'<' if g2_passed else 'NOT <'} L(it) on own docs: {json.dumps({a: (None if e is None else round(e['delta'], 5)) for a, e in g2_own.items()})}", r_star=self.state.r_star, own=g2_own, g2=g2)
        if not g2_passed:
            self.note("G2 failed: the graft does not lower -it loss on the arm's own docs; every -it readout is flagged (SPEC §5)")
        write_json(self.evidence / COMBINED_GATES_FILE, combine_gates([g1, g2_payload]))
        self.write_vector_norms()  # now with the λ = 1 kinds at r*
        self.receipt(phase, "ok", r_star=self.state.r_star, r_star_record=choice, g2_passed=g2_passed)
        self.state.phases[phase] = "ok"

    # ---- scoring helpers ------------------------------------------------------------
    def lora_delta(self, arm: str, r: int, kind: str) -> dict[str, str]:
        return {"name": delta_name(arm, kind), "kind": "lora", "path": str(self.paths.adapter_dir(arm, r)), "arm": arm}

    def full_delta(self, arm: str, kind: str = "lam0_full") -> dict[str, str]:
        return {"name": delta_name(arm, kind), "kind": "full", "path": str(self.paths.adapter_dir(arm, "full")), "arm": arm}

    def render_score_config(self, pass_name: str, *, deltas: Sequence[Mapping[str, str]], rows_filter: str | int, mode: str = "lam0", graft: Mapping[str, Any] | None = None, row_limit: int | None = None, repeats: int = 1, oracle_rows: int | None = None) -> dict[str, Any]:
        cfg = self.cfg
        base: dict[str, Any] = {
            "it_snapshot": self.snapshot("it"),
            "tokenizer_snapshot": self.snapshot("it"),
            "rows_path": str(self.paths.eft_rows),
            "out_path": str(self.paths.scores / f"{pass_name}.jsonl"),
            "pass_name": pass_name,
            "deltas": [dict(d) for d in deltas],
            "mode": mode,
            "graft": None if graft is None else dict(graft),
            "loss_lam0_path": str(self.paths.scores / "lam0.jsonl") if mode == "lam1" else None,
            "rows_filter": rows_filter,
            "episode_seed": cfg.eft_seed,
            "row_limit": row_limit,
            "repeats": repeats,
            "model_device": cfg.model_device,
            "delta_device": cfg.delta_device,
            "sequence_length": cfg.sequence_length,
            "evidence_dir": str(self.paths.scores / "evidence"),
            "oracle_rows": cfg.oracle_rows if oracle_rows is None else oracle_rows,
            "oracle_strict": cfg.oracle_strict,
        }
        base.update(dict(cfg.score_overrides))
        sc_mod.ScoreConfig.from_mapping(base)
        return base

    def _measure(self, pass_name: str, result: JobResult, planned_rows: int) -> None:
        receipt = self.newest_receipt(self.paths.scores / "evidence", f"score_lambda_grad__{pass_name}__")
        rate = None if receipt is None else receipt.get("rows_per_s")
        if rate:
            self.state.measurements["row_seconds"] = 1.0 / float(rate)
            self.state.measurements.setdefault("rows_per_s", {})[pass_name] = float(rate)
            self.log(f"measured {pass_name}: {float(rate):.3f} rows/s ({1.0 / float(rate):.2f} s/row); overhead {max(0.0, result.seconds - planned_rows / float(rate)):.0f}s")
        if receipt is not None and receipt.get("oracle"):
            self.state.measurements.setdefault("oracle", {})[pass_name] = {k: v for k, v in receipt["oracle"].items() if k != "rows"}

    async def scoring_job(self, pass_name: str, *, requested: str | int, must_reserve_seconds: float, **kwargs: Any) -> JobResult | None:
        """Plan (trim) rows to the remaining budget, then run one scorer pass."""
        if self.receipt_ok(f"score__{pass_name}") is not None:
            self.log(f"score__{pass_name}: already ok")
            return None
        counts = self.row_counts
        budget = self.deadline.remaining_for_work() - must_reserve_seconds
        plan = plan_rows(requested, budget, self.s_per_row, self.cfg.scoring_overhead_seconds, counts, granularity=self.cfg.trim_granularity, minimum=self.cfg.min_episodes)
        if plan.get("skip"):
            self.state.deadline_hit = True
            self.skip(f"score__{pass_name}", f"wall clock: even {self.cfg.min_episodes} episodes per pair type do not fit the remaining {budget:.0f}s at {self.s_per_row:.2f} s/row")
            return None
        if plan["trimmed"]:
            self.trim(f"score__{pass_name}", plan, requested)
        config = self.render_score_config(pass_name, rows_filter=plan["rows_filter"], **kwargs)
        result = await self.run(self.job(f"score__{pass_name}", [self.python, str(self.paths.script("score_lambda_grad.py"))], gpus=self.cfg.gpus(), config=config, config_env=sc_mod.CONFIG_ENV), plan={k: v for k, v in plan.items() if k != "skip"})
        self._measure(pass_name, result, plan["n_rows"])
        if result.status != "ok":
            self.fail(f"score__{pass_name}", f"exited {result.exit_code} ({result.status})")
        return result

    def _reserve_for(self, *passes: tuple[str | int, int]) -> float:
        """Seconds the later passes need at their minimum sizes."""
        counts = self.row_counts
        total = 0.0
        for requested, n_passes in passes:
            rows = counts.rows_for(min(int(requested), self.cfg.min_episodes) if requested != "all" else self.cfg.min_episodes)
            total += n_passes * (self.cfg.scoring_overhead_seconds + rows * self.s_per_row)
        return total

    # ---- phase 3 ---------------------------------------------------------------
    async def phase3_lam0(self) -> None:
        phase = "lam0"
        cfg = self.cfg
        if self.receipt_ok(phase) is not None:
            self.state.phases[phase] = "ok"
            self.log(f"{phase}: already ok")
            return
        deltas = [self.lora_delta(arm, r, f"lam0_r{r}") for arm in cfg.arms for r in cfg.ranks]
        reserve = self._reserve_for(("all", len(cfg.arms)), (cfg.full_ref_episodes, len(cfg.arms)))
        main = await self.scoring_job("lam0", requested="all", must_reserve_seconds=reserve, deltas=deltas, mode="lam0")
        statuses = {"lam0": None if main is None else main.status}
        for arm in cfg.arms:
            if not cfg.save_full:
                self.skip(f"score__lam0_full__{arm}", "save_full disabled", deliberate=True)
                continue
            reserve = self._reserve_for(("all", len(cfg.arms)))
            result = await self.scoring_job(f"lam0_full__{arm}", requested=cfg.full_ref_episodes, must_reserve_seconds=reserve, deltas=[self.full_delta(arm), self.lora_delta(arm, max(cfg.ranks), f"lam0_r{max(cfg.ranks)}")], mode="lam0")
            statuses[f"lam0_full__{arm}"] = None if result is None else result.status
        failed = [k for k, v in statuses.items() if v not in (None, "ok")]
        status = "ok" if not failed and statuses.get("lam0") in ("ok", None) else "partial"
        if statuses.get("lam0") not in ("ok", None):
            status = "failed"
        merge: dict[str, Any] | None = None
        lam0_path = self.paths.scores / "lam0.jsonl"
        full_paths = [p for p in (self.paths.scores / f"lam0_full__{arm}.jsonl" for arm in cfg.arms) if p.is_file()]
        if status != "failed" and lam0_path.is_file() and full_paths:
            merge = merge_full_scores(lam0_path, full_paths)
            self.log(f"merged full-delta scores into {lam0_path.name}: {merge['matched']} rows, names {merge['merged_names']}, {merge['unmatched']} unmatched")
        self.receipt(phase, status, passes=statuses, full_merge=merge, measurements={k: v for k, v in self.state.measurements.items() if k in ("row_seconds", "rows_per_s")})
        self.state.phases[phase] = status
        if status == "failed":
            raise GateFailure("the λ = 0 pass failed; nothing downstream is interpretable")

    # ---- phase 4 ---------------------------------------------------------------
    async def phase4_lam1(self) -> None:
        phase = "lam1"
        cfg = self.cfg
        if self.receipt_ok(phase) is not None:
            self.state.phases[phase] = "ok"
            self.log(f"{phase}: already ok")
            return
        r_star = self.state.r_star
        if r_star is None:
            raise GateFailure("r* unknown — gates did not run")
        statuses: dict[str, str | None] = {}
        for index, arm in enumerate(cfg.arms):
            deltas = [self.lora_delta(other, r_star, f"lam1_r{r_star}" if other == arm else f"lam1x_r{r_star}") for other in cfg.arms]
            graft = {"adapter_dir": str(self.paths.adapter_dir(arm, r_star)), "lam": 1.0, "arm": arm}
            remaining_arms = len(cfg.arms) - index - 1
            reserve = self._reserve_for(("all", remaining_arms)) if remaining_arms else 0.0
            result = await self.scoring_job(f"lam1__{arm}", requested="all", must_reserve_seconds=reserve, deltas=deltas, mode="lam1", graft=graft)
            statuses[arm] = None if result is None else result.status
        failed = [arm for arm, s in statuses.items() if s not in (None, "ok")]
        skipped = [arm for arm in cfg.arms if self.receipt_ok(f"score__lam1__{arm}") is None and arm not in failed]
        status = "ok" if not failed and not skipped else "partial"
        self.receipt(phase, status, r_star=r_star, arms=statuses)
        self.state.phases[phase] = status

    # ---- phase 4b (optional) -----------------------------------------------------
    async def phase4b_lam1_full(self) -> None:
        """Exact full-delta graft at λ = 1 (``lam1_full_reference``): per arm,
        merge ``adapters/<arm>/full`` into ``theta_it`` and score the arm's
        full delta (``<arm>__lam1full__all``) plus the three r_max adapters
        (``<arm>__lam1full_r<r>__all``, ``<other>__lam1fullx_r<r>__all``) ->
        ``scores/lam1full__<arm>.jsonl``."""
        phase = "lam1_full"
        cfg = self.cfg
        if not cfg.lam1_full_reference:
            self.skip(phase, "lam1_full_reference disabled by config (exact full-delta graft reference not requested)", deliberate=True)
            return
        if self.receipt_ok(phase) is not None:
            self.state.phases[phase] = "ok"
            self.log(f"{phase}: already ok")
            return
        if not cfg.save_full:
            self.skip(phase, "save_full disabled — no full deltas on disk to graft", deliberate=True)
            return
        r_max = max(cfg.ranks)
        statuses: dict[str, str | None] = {}
        for index, arm in enumerate(cfg.arms):
            deltas = [self.full_delta(arm, "lam1full")] + [self.lora_delta(other, r_max, f"lam1full_r{r_max}" if other == arm else f"lam1fullx_r{r_max}") for other in cfg.arms]
            graft = {"adapter_dir": str(self.paths.adapter_dir(arm, "full")), "lam": 1.0, "arm": arm, "kind": "full"}
            remaining_arms = len(cfg.arms) - index - 1
            reserve = self._reserve_for(("all", remaining_arms)) if remaining_arms else 0.0
            result = await self.scoring_job(f"lam1full__{arm}", requested="all", must_reserve_seconds=reserve, deltas=deltas, mode="lam1", graft=graft)
            statuses[arm] = None if result is None else result.status
        failed = [arm for arm, s in statuses.items() if s not in (None, "ok")]
        skipped = [arm for arm in cfg.arms if self.receipt_ok(f"score__lam1full__{arm}") is None and arm not in failed]
        status = "ok" if not failed and not skipped else "partial"
        self.receipt(phase, status, r_max=r_max, arms=statuses)
        self.state.phases[phase] = status

    # ---- phase 5 ---------------------------------------------------------------
    async def phase5_noise(self) -> None:
        phase = NOISE_PASS
        cfg = self.cfg
        if self.receipt_ok(phase) is not None:
            self.state.phases[phase] = "ok"
            self.log(f"{phase}: already ok")
            self._gate_g3()
            return
        deltas = [self.lora_delta(arm, r, f"lam0_r{r}") for arm in cfg.arms for r in cfg.ranks]
        counts = self.row_counts
        needed = cfg.scoring_overhead_seconds + cfg.repeat_rows * cfg.repeats * self.s_per_row
        if needed > self.deadline.remaining_for_work():
            self.state.deadline_hit = True
            self.skip(phase, f"wall clock: noise-floor pass needs {needed:.0f}s, {self.deadline.remaining_for_work():.0f}s left")
            self._gate_g3()
            return
        if self.receipt_ok(f"score__{NOISE_PASS}") is None:
            config = self.render_score_config(NOISE_PASS, deltas=deltas, rows_filter="all", mode="lam0", row_limit=min(cfg.repeat_rows, counts.rows_for("all")), repeats=cfg.repeats, oracle_rows=0)
            result = await self.run(self.job(f"score__{NOISE_PASS}", [self.python, str(self.paths.script("score_lambda_grad.py"))], gpus=cfg.gpus(), config=config, config_env=sc_mod.CONFIG_ENV))
            if result.status != "ok":
                self.fail(phase, f"score__{NOISE_PASS} exited {result.exit_code}")
                self.receipt(phase, "failed", reason=f"score__{NOISE_PASS} exited {result.exit_code}")
                self.state.phases[phase] = "failed"
                self._gate_g3()
                return
        receipt = self.newest_receipt(self.paths.scores / "evidence", f"score_lambda_grad__{NOISE_PASS}__")
        noise = (receipt or {}).get("repeat_noise") or {}
        overall = noise.get("scores_all") or {}
        median = overall.get("median")
        passed = None if median is None else median <= cfg.g4_median_max
        self.gate("G4", passed, summary=f"repeat-score relative spread median {median} (gate <= {cfg.g4_median_max}), p90 {overall.get('p90')}, n_pairs {noise.get('n_pairs')}", noise=noise)
        self.receipt(phase, "ok", noise=noise)
        self.state.phases[phase] = "ok"
        self._gate_g3()

    def _gate_g3(self) -> None:
        if "G3" in self.state.gates:
            return
        cfg = self.cfg
        per_arm: dict[str, Any] = {}
        r_max = max(cfg.ranks)
        for arm in cfg.arms:
            path = self.paths.scores / f"lam0_full__{arm}.jsonl"
            if not path.is_file():
                continue
            records = sc_mod.read_records(path)
            per_arm[arm] = g3_exactness(records, delta_name(arm, f"lam0_r{r_max}"), delta_name(arm, "lam0_full"))
        if not per_arm:
            self.gate("G3", None, summary="no full-delta reference files", per_arm={})
            return
        spearmans = {arm: v["spearman"] for arm, v in per_arm.items()}
        self.gate("G3", None, summary=f"LoRA r{r_max} vs full: Spearman {json.dumps({a: (None if s is None else round(s, 4)) for a, s in spearmans.items()})}, slope {json.dumps({a: (None if v['slope_full_on_lora'] is None else round(v['slope_full_on_lora'], 4)) for a, v in per_arm.items()})} (descriptive gate: reported, no threshold)", per_arm=per_arm)

    # ---- phase 6 ---------------------------------------------------------------
    async def phase6_analysis(self) -> None:
        """``analysis.analyze_graft.run_all(<root>)`` -> ``<root>/results/`` (CPU; non-fatal)."""
        phase = "analysis"
        cfg = self.cfg
        if not cfg.analysis:
            self.skip(phase, "analysis disabled by config", deliberate=True)
            return
        if not (self.paths.scores / "lam0.jsonl").is_file():
            self.skip(phase, "no scores/lam0.jsonl to analyse")
            return
        if self.receipt_ok(phase) is not None and self.paths.results.is_dir():
            self.state.phases[phase] = "ok"
            self.log(f"{phase}: already ok (results in {self.paths.results})")
            return
        primary_rank = cfg.analysis_primary_rank or self.state.r_star or (256 if 256 in cfg.ranks else max(cfg.ranks))
        result = await self.run(self.job(phase, [self.python, "-c", ANALYSIS_SNIPPET, str(self.paths.repo_root), str(self.paths.root), str(self.paths.results), str(cfg.analysis_n_boot), str(primary_rank)], gpus=(), offline=True), primary_rank=primary_rank)
        self.state.phases[phase] = "ok" if result.status == "ok" else "failed"
        if result.status != "ok":
            self.fail(phase, f"analyze_graft.run_all exited {result.exit_code}")

    # ---- phase 7 ---------------------------------------------------------------
    async def phase7_publish(self, status: str) -> None:
        phase = "publish"
        staging = self.paths.staging / self.run_id
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        write_json(self.evidence / "driver_summary.json", self.done_payload(status, finished=False))
        manifest = stage_for_upload(self.paths, staging, upload_adapters=self.cfg.upload_adapters)
        write_json(staging / "staging_manifest.json", manifest)
        publication: dict[str, Any] = {"status": "skipped", "staging": {k: v for k, v in manifest.items() if k != "files"}}
        if not self.cfg.upload:
            publication["reason"] = "upload disabled by config"
        elif not os.environ.get("HF_TOKEN"):
            publication["reason"] = "HF_TOKEN not set"
        else:
            try:
                report = await asyncio.to_thread(self.deps.upload_folder, str(staging), self.cfg.hf_repo, f"runs/{self.run_id}")
                publication = {"status": "ok", **dict(report), "staging": publication["staging"]}
            except Exception as error:  # noqa: BLE001 — publication failure must not mask the run
                publication = {"status": "failed", "error": repr(error), "staging": publication["staging"]}
                self.fail(phase, f"HF upload failed: {error!r}")
        self.state.publication = publication
        self.receipt("publication", "ok" if publication["status"] in ("ok", "skipped") else "failed", publication=publication)
        self.state.phases[phase] = "ok" if publication["status"] in ("ok", "skipped") else "failed"

    # ---- summary -----------------------------------------------------------------
    def done_payload(self, status: str, *, finished: bool) -> dict[str, Any]:
        started_at = read_json(self.evidence / STARTED_FILE).get("started_at") if (self.evidence / STARTED_FILE).is_file() else utc_now()
        return {
            "run_id": self.run_id,
            "status": status,
            "started_at": started_at,
            "finished_at": utc_now() if finished else None,
            "elapsed_seconds": self.deadline.elapsed(),
            "deadline": self.deadline.to_dict(),
            "deadline_hit": self.state.deadline_hit,
            "phases": dict(self.state.phases),
            "skipped": list(self.state.skipped),
            "failures": list(self.state.failures),
            "notes": list(self.state.notes),
            "gates": dict(self.state.gates),
            "r_star": self.state.r_star,
            "r_star_record": dict(self.state.r_star_record),
            "trims": list(self.state.trims),
            "measurements": dict(self.state.measurements),
            "inputs": {k: v for k, v in self.state.inputs.items() if k in ("snapshots", "listing", "host")},
            "preflight": dict(self.state.preflight),
            "publication": dict(self.state.publication),
            "config": self.cfg.to_dict(),
            "evidence_dir": str(self.evidence),
        }

    def overall_status(self, fatal: bool) -> str:
        if fatal:
            return "failed"
        lost = [s for s in self.state.skipped if not s.get("deliberate")]
        if lost or self.state.failures or any(v in ("partial", "failed") for v in self.state.phases.values()):
            return "partial"
        return "complete"

    def _preload_gates(self) -> None:
        for path in sorted(self.evidence.glob("gate_*.json")):
            try:
                body = read_json(path)
            except (OSError, ValueError):
                continue
            if body.get("run_id") != self.run_id:
                continue
            self.state.gates[path.stem[len("gate_"):].upper()] = {k: v for k, v in body.items() if k not in RECEIPT_REQUIRED_KEYS}

    async def run_all(self) -> dict[str, Any]:
        self.log(f"driver start run_id={self.run_id} root={self.paths.root} n_gpus={self.cfg.n_gpus} budget={self.cfg.wall_clock_budget_seconds / 3600:.1f} h elapsed={self.deadline.elapsed():.0f} s")
        if self.cfg.resume:
            self._preload_gates()
        write_json(self.evidence / "driver_config.json", {"run_id": self.run_id, "config": self.cfg.to_dict(), "python": self.python, "repo_root": str(self.paths.repo_root)})
        fatal = False
        try:
            await self.phase0_preflight()
            await self.phase1_extract()
            await self.phase2_gates()
            await self.phase3_lam0()
            await self.phase4_lam1()
            await self.phase4b_lam1_full()
            await self.phase5_noise()
        except BaseException as error:  # noqa: BLE001 — recorded, then analysis + publish still run
            fatal = True
            text = traceback.format_exc()
            (self.evidence / FAILURE_FILE).write_text(text, encoding="utf-8")
            self.fail("driver", f"{type(error).__name__}: {error}")
            self.log(f"{FAIL_SENTINEL} fatal: {error!r}\n{text}")
        finally:
            if self.evictor is not None:
                self.deps.stop_evictor(self.evictor)
                self.evictor = None
        try:
            await self.phase6_analysis()
        except Exception as error:  # noqa: BLE001
            self.fail("analysis", repr(error))
        status = self.overall_status(fatal)
        try:
            await self.phase7_publish(status)
        except Exception as error:  # noqa: BLE001
            self.fail("publish", repr(error))
        status = self.overall_status(fatal)
        done = self.done_payload(status, finished=True)
        validate_done(done)
        write_json(self.evidence / DONE_FILE, done)
        self.log(f"{DONE_SENTINEL} status={status} elapsed={done['elapsed_seconds'] / 3600:.2f} h r_star={done['r_star']} skipped={len(done['skipped'])} failures={len(done['failures'])}")
        return done


async def run_driver(cfg: DriverConfig, deps: DriverDeps | None = None, *, paths: Paths | None = None) -> dict[str, Any]:
    return await Driver(cfg, deps, paths=paths).run_all()


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        raise SystemExit(f"usage: run_all.py — config-first, no flags; set ${CONFIG_ENV}=<config.json|yaml>")
    cfg = load_driver_config(os.environ.get(CONFIG_ENV) or None)
    done = asyncio.run(run_driver(cfg))
    return 0 if done["status"] in ("complete", "partial") else 2


if __name__ == "__main__":
    sys.exit(main())
