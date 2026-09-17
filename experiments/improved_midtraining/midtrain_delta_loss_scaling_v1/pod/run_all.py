"""Pod driver for midtrain_delta_loss_scaling_v1 (SPEC §§2–5, 7–8; PREMORTEM e/h/i).

Phases (each a receipt ``<root>/evidence/<phase>.json`` and a
``SCIMT-DRIVER-PHASE <name> status=ok|failed|skipped|partial`` line in
``evidence/driver.log``); every phase is idempotent for one run id (anchored in
``evidence/driver_started.json``), so re-running the same command resumes:

0. ``preflight``  host gates (RAM, ≥ 800 GB free disk, GPUs), torch/CUDA probe, the
                  checkpoint repo revision resolved ONCE to a commit sha, the 28
                  checkpoint dirs verified against the repo listing (fail loud with
                  candidates), ``evidence/models.json`` (catalog records + revision +
                  local snapshot paths), the 6,000 EFT rows (v1 builder, seed 20260913;
                  class counts are DESCRIPTIVE), ``evidence/inputs.json``.
1. ``score``      a value-ordered queue over TWO GPU slots: Gemma jobs take one GPU
                  each (two run concurrently, ``CUDA_VISIBLE_DEVICES=0`` / ``1``), GLM
                  jobs take both (``device_map=auto``); the head of the queue never
                  gets jumped. Per checkpoint: ``snapshot_download(<profile>/<arm>/base/
                  only, revision=<sha>, local_dir=<root>/snapshots/<tag>)`` (the next
                  ``prefetch_depth`` snapshots download while models score) ->
                  ``row_losses.py`` (supervised subprocess) -> row-count verification
                  (descriptive) -> the small files + the hub's per-file LFS sha256s
                  copied to ``evidence/checkpoint_files/<tag>/`` -> ``shutil.rmtree`` of
                  the snapshot dir -> ``evidence/score__<tag>.json`` -> incremental
                  publish of ``scores/`` + ``evidence/`` to HF. A deadline planner skips
                  a model that will not fit the remaining wall budget
                  (``SCIMT-DRIVER-TRIM``); low-value models are last in line.
                  Afterwards: config identity across arms (minus generation keys) and
                  tokenization identity per substrate (template md5, tokenizer.json
                  sha256, rendered ids sha256) -> ``SCIMT-DRIVER-GATE`` receipts.
2. ``analysis``   ``analysis.analyze_scaling.run_all(root, root/'results')`` in a
                  subprocess (CPU; absence or failure is non-fatal: a receipt).
3. ``publish``    ``scores/ evidence/ results/ eft_rows/`` -> HF dataset repo
                  ``runs/<run_id>/``; then ``evidence/DRIVER_DONE.json``.

Fatal = correctness only (the scorer's own gates: loading info, span constants,
batch agreement, non-finite CE). Descriptive checks (class counts, row counts,
noise floors, sanity CE, identity gates) are receipts, never stops. The driver
never sets ``HF_HUB_OFFLINE`` (downloads stream all run) and never writes to its
input config path; ``evidence/heartbeat`` is touched every minute for pod-watch.
"""

from __future__ import annotations

import asyncio
import contextlib
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

from experiments.improved_midtraining.midtrain_delta_loss_scaling_v1.pod import (
    row_losses as rl,  # noqa: E402
)
from experiments.improved_midtraining.midtrain_delta_loss_scaling_v1.pod.common import (  # noqa: E402
    EXPERIMENT_NAME,
    GB,
    LOSS_ROW_KEYS,
    MODELS_YAML,
    SUBSTRATES,
    Catalog,
    ModelEntry,
    describe_row_counts,
    expected_group_counts,
    load_catalog,
    load_eft_rows,
    load_mapping,
    log,
    merge_mappings,
    now_iso,
    read_json,
    read_jsonl,
    sha256_file,
    timestamp_tag,
    write_json,
)

CONFIG_ENV = "SCIMT_MDLS_DRIVER_CONFIG"
PHASE_SENTINEL = "SCIMT-DRIVER-PHASE"
MODEL_SENTINEL = "SCIMT-DRIVER-MODEL"
GATE_SENTINEL = "SCIMT-DRIVER-GATE"
TRIM_SENTINEL = "SCIMT-DRIVER-TRIM"
DONE_SENTINEL = "SCIMT-DRIVER-DONE"
FAIL_SENTINEL = "SCIMT-DRIVER-FAIL"
PREFLIGHT_PREFIX = "SCIMT-PREFLIGHT "
DONE_FILE = "DRIVER_DONE.json"
LOG_FILE = "driver.log"
STARTED_FILE = "driver_started.json"
INPUTS_FILE = "inputs.json"
MODELS_FILE = "models.json"
FAILURE_FILE = "driver_failure.txt"
RESOLVED_CONFIG_FILE = "driver_config_resolved.json"
SUMMARY_FILE = "driver_summary.json"
HEARTBEAT_FILE = "heartbeat"
CHECKPOINT_FILES_DIR = "checkpoint_files"
PHASES = ("preflight", "score", "analysis", "publish")
ANALYSIS_SNIPPET = (
    "import sys, json; from pathlib import Path; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + '/src'); "
    "from experiments.improved_midtraining.midtrain_delta_loss_scaling_v1.analysis import analyze_scaling; "
    "root = Path(sys.argv[2]); manifest = analyze_scaling.run_all(root, root / 'results'); "
    "print('SCIMT-ANALYSIS-DONE', json.dumps({'keys': sorted(manifest)[:20] if isinstance(manifest, dict) else None}))"
)
RECEIPT_STATUSES = ("ok", "failed", "skipped", "partial", "timeout", "running", "gate-failed")
RECEIPT_REQUIRED_KEYS = ("run_id", "phase", "status", "written_at")
JOB_RECEIPT_KEYS = ("exit_code", "seconds", "gpu_peak_gb", "gpus", "argv", "log_path", "tail", "started_at", "finished_at")
DONE_REQUIRED_KEYS = ("run_id", "status", "started_at", "finished_at", "elapsed_seconds", "phases", "models", "skipped", "failures", "notes", "publication", "deadline_hit", "hf_revision", "gates")
DONE_STATUSES = ("complete", "partial", "failed")
UPLOAD_DENY_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".tmp", ".gguf", ".f32")
UPLOAD_DENY_DIRS = (".cache", "__pycache__", "staging", "snapshots", "hf")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
SMALL_FILE_MAX_BYTES = 64 << 20  # everything but the weights is copied to evidence before rmtree
MIN_BYTES_FOR_THROUGHPUT = 1_000_000_000  # only real snapshots calibrate the planner's download estimate
#: config.json keys ignored by the cross-arm identity check (generation settings differ by arm)
CONFIG_IDENTITY_IGNORE = ("eos_token_id", "pad_token_id", "bos_token_id", "use_cache", "cache_implementation", "transformers_version", "_name_or_path", "torch_dtype", "dtype", "generation_config", "_attn_implementation", "_experts_implementation")
PREFLIGHT_SNIPPET = (
    "import json, importlib.metadata as m\n"
    "out = {}\n"
    "for name in ('torch', 'transformers', 'accelerate', 'safetensors', 'huggingface_hub', 'hf_transfer', 'hf_xet', 'zstandard', 'numpy'):\n"
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


def _positive(value: Any, label: str, *, integer: bool = False, zero_ok: bool = False) -> None:
    ok_type = isinstance(value, int) and not isinstance(value, bool) if integer else isinstance(value, (int, float)) and not isinstance(value, bool)
    if not ok_type or value < 0 or (value == 0 and not zero_ok):
        raise ValueError(f"{label} must be a positive {'integer' if integer else 'number'}{' or zero' if zero_ok else ''}, got {value!r}")


def _substrate_map(value: Any, label: str, *, allow_none: bool = True) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping keyed by substrate")
    for key, item in value.items():
        if key not in SUBSTRATES:
            raise ValueError(f"{label}: unknown substrate {key!r} (known: {SUBSTRATES})")
        if item is None and not allow_none:
            raise ValueError(f"{label}[{key}] must not be null")


def strip_generation_keys(config: Mapping[str, Any], ignore: Sequence[str] = CONFIG_IDENTITY_IGNORE) -> dict[str, Any]:
    """config.json minus the per-arm generation settings (recursively)."""
    out: dict[str, Any] = {}
    for key, value in config.items():
        if key in ignore:
            continue
        out[key] = strip_generation_keys(value, ignore) if isinstance(value, Mapping) else value
    return out


# ------------------------------------------------------------------- config
@dataclass(frozen=True)
class DriverConfig:
    """Every knob of the pod run; the defaults are the launch configuration."""

    run_id: str = ""  # "" -> UTC timestamp at first start (persisted for resume)
    repo_root: str = "/workspace/scimt"
    root: str = "/workspace/mdls"  # = the analysis exp_dir (scores/ + evidence/ + results/)
    hf_home: str = "/workspace/hf"
    models_yaml: str | None = None  # None -> <experiment>/models.yaml
    python: str | None = None  # None -> sys.executable
    resume: bool = True
    echo_subprocess_output: bool = True
    n_gpus: int = 2
    # --- wall clock (PREMORTEM: GLM ≈ 1 h/model -> 7–9 h) ---
    wall_clock_budget_seconds: float = 10.0 * 3600.0
    publish_reserve_seconds: float = 20 * 60.0
    analysis_reserve_seconds: float = 10 * 60.0
    heartbeat_seconds: float = 60.0
    # --- checkpoints (SPEC §2) ---
    hf_revision: str | None = None  # None -> resolve the repo's current commit sha at preflight
    only_models: tuple[str, ...] = ()  # "profile/arm" keys; empty -> all 28
    skip_models: tuple[str, ...] = ()
    hf_transfer: bool = True  # HF_HUB_ENABLE_HF_TRANSFER=1 when hf_transfer is importable
    prefetch_depth: int = 2  # snapshots downloaded ahead of the queue while models score
    prefetch_min_free_disk_gb: float = 300.0
    min_download_gbps: float = 0.3  # descriptive gate on the first download (300 MB/s; else re-roll the host)
    delete_snapshots: bool = True  # rmtree after an ok receipt (small files are kept in evidence/)
    # --- rows (SPEC §3) ---
    n_conflict_episodes: int = 1500
    n_agreement_episodes: int = 1500
    eft_seed: int = 20260913
    # --- scoring (SPEC §4) ---
    batch_size: int = 1
    batch_check_rows: int = 200
    batch_check_abs_tol: float = 0.01
    max_tokens: int = 8192
    repeat_rows: int = 200
    repeat_seed: int = 20260913
    noise_models: tuple[str, ...] = ("primary_controls",)  # keys or the alias
    fp32_check_model: str | None = "gemma3_12b_50m_4ep/control"  # one 12B bf16-vs-fp32 sizing check
    fp32_check_rows: int = 32
    sanity_max_content_ce: float = 6.0
    require_constant_template_ids: bool = True
    load_fallback: bool = True
    device_map: Mapping[str, str | None] = field(default_factory=lambda: {"gemma3_12b": None, "gemma3_27b": None, "glm45_air": "auto"})
    attn_implementation: Mapping[str, str | None] = field(default_factory=lambda: {"gemma3_12b": None, "gemma3_27b": None, "glm45_air": "sdpa"})
    experts_implementation: Mapping[str, str | None] = field(default_factory=lambda: {"glm45_air": "grouped_mm"})
    score_overrides: Mapping[str, Any] = field(default_factory=dict)  # merged into every scorer config
    # --- planner ---
    expected_model_seconds: Mapping[str, float] = field(default_factory=lambda: {"gemma3_12b": 900.0, "gemma3_27b": 1200.0, "glm45_air": 3600.0})
    expected_download_gbps: float = 0.4  # GB/s until measured
    approx_bytes: Mapping[str, int] = field(default_factory=dict)  # per substrate; empty -> models.yaml
    job_timeout_factor: float = 4.0  # timeout = factor x expected seconds (>= job_timeout_min_seconds)
    job_timeout_min_seconds: float = 1800.0
    # --- host gates ---
    min_host_ram_gb: float = 100.0
    min_free_disk_gb: float = 800.0
    disk_gate: str = "fail"  # fail | warn
    # --- analysis + publish ---
    analysis: bool = True
    upload: bool = True
    publish_incremental: bool = True  # scores/ + evidence/ after every model
    hf_dataset_repo: str = "jbostock/scimt-midtrain-delta-loss-scaling-v1"
    evictor: bool = True
    evict_threshold_gb: float = 200.0

    _TUPLE_FIELDS = ("only_models", "skip_models", "noise_models")
    _MAP_FIELDS = ("device_map", "attn_implementation", "experts_implementation", "score_overrides", "expected_model_seconds", "approx_bytes")

    def __post_init__(self) -> None:
        for name in self._TUPLE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, list):
                object.__setattr__(self, name, tuple(value))
            if isinstance(getattr(self, name), str) or not isinstance(getattr(self, name), tuple):
                raise ValueError(f"{name} must be a list of strings")
        for name in self._MAP_FIELDS:
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError(f"{name} must be a mapping")
        if not isinstance(self.n_gpus, int) or isinstance(self.n_gpus, bool) or self.n_gpus < 1:
            raise ValueError("n_gpus must be an integer >= 1")
        for label in ("wall_clock_budget_seconds", "heartbeat_seconds", "expected_download_gbps", "min_download_gbps", "job_timeout_factor", "job_timeout_min_seconds", "min_host_ram_gb", "min_free_disk_gb", "prefetch_min_free_disk_gb", "evict_threshold_gb", "batch_check_abs_tol", "sanity_max_content_ce"):
            _positive(getattr(self, label), label)
        for label in ("publish_reserve_seconds", "analysis_reserve_seconds"):
            _positive(getattr(self, label), label, zero_ok=True)
        for label in ("n_conflict_episodes", "n_agreement_episodes", "batch_size", "batch_check_rows", "max_tokens"):
            _positive(getattr(self, label), label, integer=True)
        for label in ("repeat_rows", "repeat_seed", "eft_seed", "prefetch_depth", "fp32_check_rows"):
            _positive(getattr(self, label), label, integer=True, zero_ok=True)
        if self.disk_gate not in ("fail", "warn"):
            raise ValueError("disk_gate must be fail|warn")
        _substrate_map(self.device_map, "device_map")
        _substrate_map(self.attn_implementation, "attn_implementation")
        _substrate_map(self.experts_implementation, "experts_implementation")
        _substrate_map(self.expected_model_seconds, "expected_model_seconds", allow_none=False)
        _substrate_map(self.approx_bytes, "approx_bytes", allow_none=False)
        for key, value in self.device_map.items():
            if value not in (None, "auto"):
                raise ValueError(f"device_map[{key}] must be null or 'auto'")
        if self.hf_revision is not None and (not isinstance(self.hf_revision, str) or not self.hf_revision):
            raise ValueError("hf_revision must be null or a non-empty string")
        if self.fp32_check_model is not None and (not isinstance(self.fp32_check_model, str) or "/" not in self.fp32_check_model):
            raise ValueError("fp32_check_model must be null or a 'profile/arm' key")
        for key in ("model_dir", "rows_path", "out_path", "profile", "arm", "substrate", "dose_tokens", "noise_out_path", "manifest_path", "tokens_out_path", "hf_revision", "hf_path", "expected_architecture", "expected_layers", "fp32_check_rows"):
            if key in self.score_overrides:
                raise ValueError(f"score_overrides may not set {key!r} (the driver owns it)")

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
        for name in self._MAP_FIELDS:
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
        return cls(root=Path(cfg.root), repo_root=Path(cfg.repo_root), hf_home=Path(cfg.hf_home))

    @property
    def evidence(self) -> Path:
        return self.root / "evidence"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def scores(self) -> Path:
        return self.root / "scores"

    @property
    def eft_rows_dir(self) -> Path:
        return self.root / "eft_rows"

    @property
    def eft_rows(self) -> Path:
        return self.eft_rows_dir / "eft_rows.jsonl"

    @property
    def snapshots(self) -> Path:
        return self.root / "snapshots"

    @property
    def staging(self) -> Path:
        return self.root / "staging"

    @property
    def pod_dir(self) -> Path:
        return self.repo_root / "experiments" / "improved_midtraining" / EXPERIMENT_NAME / "pod"

    def script(self, name: str) -> Path:
        return self.pod_dir / name

    def snapshot_dir(self, entry: ModelEntry) -> Path:
        return self.snapshots / entry.tag

    def model_dir(self, entry: ModelEntry) -> Path:
        return self.snapshot_dir(entry) / entry.hf_path

    def checkpoint_files(self, entry: ModelEntry) -> Path:
        return self.evidence / CHECKPOINT_FILES_DIR / entry.tag

    def losses(self, entry: ModelEntry) -> Path:
        return self.scores / f"losses__{entry.tag}.jsonl"

    def losses_manifest(self, entry: ModelEntry) -> Path:
        return self.scores / f"losses__{entry.tag}.manifest.json"

    def tokens(self, entry: ModelEntry) -> Path:
        return self.scores / f"tokens__{entry.tag}.npz"

    def noise(self, entry: ModelEntry) -> Path:
        return self.scores / f"noise__{entry.tag}.jsonl"


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
        if self.exit_code == 0:
            return "ok"
        return "failed"


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
        except TimeoutError:
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
    timeout (SIGTERM, SIGKILL after 30 s). Token-like env values are never logged."""
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
        handle.write(f"# [{started_at}] {job.name}: {' '.join(job.argv)}\n# env: " + json.dumps({k: v for k, v in job.env.items() if "TOKEN" not in k and "KEY" not in k}) + "\n")
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
def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def snapshot_download_default(repo_id: str, *, revision: str | None, allow_patterns: Sequence[str], local_dir: str, repo_type: str) -> str:
    from huggingface_hub import snapshot_download

    return str(snapshot_download(repo_id, revision=revision, repo_type=repo_type, allow_patterns=list(allow_patterns), local_dir=local_dir, token=_hf_token(), max_workers=16))


def list_repo_files_default(repo_id: str, *, revision: str | None, repo_type: str) -> list[str]:
    from huggingface_hub import list_repo_files

    return list(list_repo_files(repo_id, revision=revision, repo_type=repo_type, token=_hf_token()))


def list_repo_tree_default(repo_id: str, *, path_in_repo: str, revision: str | None, repo_type: str) -> list[dict[str, Any]]:
    """Per-file size + LFS sha256 (``expand=True``) of one checkpoint dir."""
    from huggingface_hub import HfApi

    out: list[dict[str, Any]] = []
    for item in HfApi(token=_hf_token()).list_repo_tree(repo_id, path_in_repo=path_in_repo, revision=revision, repo_type=repo_type, recursive=False, expand=True):
        lfs = getattr(item, "lfs", None)
        sha = None
        if lfs is not None:
            sha = lfs.get("sha256") if hasattr(lfs, "get") else getattr(lfs, "sha256", None)
        out.append({"path": getattr(item, "path", None), "size": getattr(item, "size", None), "lfs_sha256": sha, "blob_id": getattr(item, "blob_id", None), "type": type(item).__name__})
    return out


def resolve_revision_default(repo_id: str, *, revision: str | None, repo_type: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi(token=_hf_token()).repo_info(repo_id, revision=revision, repo_type=repo_type, files_metadata=False)
    return str(info.sha)


def write_eft_rows_default(out_path: str, *, n_conflict_episodes: int, n_agreement_episodes: int, seed: int) -> Path:
    from experiments.improved_midtraining.ekfac_dataset_attribution_v1.eft_rows import (
        build_eft_rows,
    )

    return build_eft_rows.write_eft_rows(Path(out_path), n_conflict_episodes, n_agreement_episodes, seed)


def host_ram_gb_default() -> float:
    total = None
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                total = float(line.split()[1]) * 1024 / GB
                break
    if total is None:
        raise RuntimeError("MemTotal not found in /proc/meminfo")
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) < 100 * 1024**4:
            return min(total, int(raw) / GB)
        break
    return total


def disk_free_gb_default(path: str) -> float:
    probe = Path(path)
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free / GB


def upload_folder_hf(staging_dir: str, repo_id: str, path_in_repo: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=_hf_token())
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    files = [p for p in Path(staging_dir).rglob("*") if p.is_file()]
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=staging_dir, path_in_repo=path_in_repo, commit_message=f"{EXPERIMENT_NAME}: {path_in_repo}")
    return {"repo_id": repo_id, "repo_type": "dataset", "path_in_repo": path_in_repo, "url": f"https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}", "n_files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def spawn_evictor_default(python: str, script: str, roots: Sequence[str], log_path: str, threshold_gb: float) -> Any:
    env = dict(os.environ)
    env["EVICT_ROOTS"] = ":".join(roots)
    env["EVICT_THRESHOLD_GB"] = str(threshold_gb)
    handle = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 — lives as long as the process
    return subprocess.Popen([python, script], env=env, stdout=handle, stderr=subprocess.STDOUT)


def stop_evictor_default(proc: Any) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001 — SIGKILL is the fallback; a vanished process is fine
        with contextlib.suppress(Exception):
            proc.kill()


def hf_transfer_available_default() -> bool:
    try:
        import hf_transfer  # noqa: F401
    except ImportError:
        return False
    return True


def hf_token_available_default() -> bool:
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return get_token() is not None
    except Exception:  # noqa: BLE001
        return False


@dataclass
class DriverDeps:
    """Every side effect, injectable (tests pass fakes; the pod uses defaults)."""

    run_job: Callable[[Job], Awaitable[JobResult]]
    snapshot_download: Callable[..., str] = snapshot_download_default
    list_repo_files: Callable[..., list[str]] = list_repo_files_default
    list_repo_tree: Callable[..., list[dict[str, Any]]] = list_repo_tree_default
    resolve_revision: Callable[..., str] = resolve_revision_default
    write_eft_rows: Callable[..., Path] = write_eft_rows_default
    host_ram_gb: Callable[[], float] = host_ram_gb_default
    disk_free_gb: Callable[[str], float] = disk_free_gb_default
    upload_folder: Callable[[str, str, str], Mapping[str, Any]] = upload_folder_hf
    spawn_evictor: Callable[..., Any] = spawn_evictor_default
    stop_evictor: Callable[[Any], None] = stop_evictor_default
    hf_transfer_available: Callable[[], bool] = hf_transfer_available_default
    hf_token_available: Callable[[], bool] = hf_token_available_default
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


def plan_model(approx_bytes: int, *, downloaded: bool, download_gbps: float, model_seconds: float, remaining_seconds: float) -> dict[str, Any]:
    """Will this checkpoint (download unless already on disk + load/score) fit
    the remaining work budget? Pure arithmetic; the driver logs a TRIM when not."""
    if download_gbps <= 0 or model_seconds < 0:
        raise ValueError("download_gbps must be positive and model_seconds non-negative")
    download_s = 0.0 if downloaded else approx_bytes / GB / download_gbps
    projected = download_s + model_seconds
    return {"download_seconds": download_s, "score_seconds": model_seconds, "projected_seconds": projected, "remaining_seconds": remaining_seconds, "fits": projected <= remaining_seconds, "downloaded": downloaded}


# ----------------------------------------------------------- snapshot I/O
def verify_snapshot(model_dir: Path, hub_files: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """A usable checkpoint dir: config.json, tokenizer files, template, every
    shard the index names — and, when the hub listing is known, every listed
    file present at its listed size. Raises when something is missing."""
    problems: list[str] = []
    if not (model_dir / "config.json").is_file():
        problems.append("config.json missing")
    shards = sorted(p.name for p in model_dir.glob("*.safetensors"))
    if not shards:
        problems.append("no *.safetensors")
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        expected = sorted(set(read_json(index).get("weight_map", {}).values()))
        missing = sorted(set(expected) - set(shards))
        if missing:
            problems.append(f"{len(missing)} shards named by the index are missing (e.g. {missing[:3]})")
    if not (model_dir / "tokenizer.json").is_file() and not (model_dir / "tokenizer.model").is_file():
        problems.append("tokenizer files missing")
    tokenizer_config = model_dir / "tokenizer_config.json"
    if not (model_dir / "chat_template.jinja").is_file() and "chat_template" not in (tokenizer_config.read_text(encoding="utf-8") if tokenizer_config.is_file() else ""):
        problems.append("no chat_template.jinja and no chat_template in tokenizer_config.json")
    size_mismatches: list[str] = []
    if hub_files:
        for item in hub_files:
            name = str(item.get("path", "")).rsplit("/", 1)[-1]
            size = item.get("size")
            local = model_dir / name
            if not local.is_file():
                problems.append(f"listed file {name} missing locally")
            elif size is not None and local.stat().st_size != int(size):
                size_mismatches.append(f"{name}: local {local.stat().st_size} != hub {size}")
        if size_mismatches:
            problems.append("size mismatches vs the hub listing: " + "; ".join(size_mismatches[:4]))
    if problems:
        raise RuntimeError(f"snapshot {model_dir} unusable: {'; '.join(problems)}")
    files = [p for p in model_dir.rglob("*") if p.is_file()]
    return {"path": str(model_dir), "n_files": len(files), "n_safetensors": len(shards), "bytes": sum(p.stat().st_size for p in files), "shards": shards}


def snapshot_has_weights(model_dir: Path) -> bool:
    return model_dir.is_dir() and any(model_dir.glob("*.safetensors"))


def preserve_small_files(model_dir: Path, target: Path) -> list[str]:
    """Copy every non-weight file (config, tokenizer, template, index, README)
    of a checkpoint dir into evidence before the snapshot is removed."""
    target.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    if model_dir.is_dir():
        for path in sorted(model_dir.iterdir()):
            if path.is_file() and path.suffix not in WEIGHT_SUFFIXES and path.stat().st_size <= SMALL_FILE_MAX_BYTES:
                shutil.copy2(path, target / path.name)
                kept.append(path.name)
    return kept


def remove_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    """``shutil.rmtree`` of one model's snapshot dir (weights + hub bookkeeping)."""
    freed = sum(p.stat().st_size for p in snapshot_dir.rglob("*") if p.is_file()) if snapshot_dir.is_dir() else 0
    if snapshot_dir.is_dir():
        shutil.rmtree(snapshot_dir, ignore_errors=True)
    return {"freed_bytes": freed, "removed_dir": str(snapshot_dir), "remaining": snapshot_dir.exists()}


def verify_scores(path: Path, expected_row_ids: Sequence[str], entry: ModelEntry) -> dict[str, Any]:
    """Descriptive completeness of one scorer output: every expected row_id
    exactly once, the schema keys, finite losses, and the model identity."""
    records = read_jsonl(path)
    ids = [str(r.get("row_id")) for r in records]
    seen = set(ids)
    expected = set(expected_row_ids)
    bad_keys = sum(1 for r in records if set(r) != set(LOSS_ROW_KEYS))
    nonfinite = 0
    for r in records:
        for key in ("loss", "loss_content", "loss_full", "loss_prompt"):
            value = r.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                nonfinite += 1
                break
    wrong_model = sum(1 for r in records if r.get("profile") != entry.profile or r.get("arm") != entry.arm)
    templates = sorted({str(r.get("template_md5")) for r in records})
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    duplicates = len(ids) - len(seen)
    ok = not missing and not extra and not duplicates and not bad_keys and not nonfinite and not wrong_model and len(templates) <= 1
    return {"path": str(path), "n_records": len(records), "n_expected": len(expected), "n_missing": len(missing), "missing_examples": missing[:5], "n_extra": len(extra), "duplicates": duplicates, "bad_keys": bad_keys, "nonfinite": nonfinite, "wrong_model": wrong_model, "template_md5s": templates, "ok": ok}


def config_identity(configs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Group the given ``key -> config.json`` by equality after dropping the
    generation keys; ``ok`` when they all agree."""
    groups: dict[str, list[str]] = {}
    for key, config in configs.items():
        digest = json.dumps(strip_generation_keys(config), sort_keys=True)
        groups.setdefault(digest, []).append(key)
    members = sorted(groups.values(), key=lambda g: (-len(g), g))
    return {"n": len(configs), "n_distinct": len(groups), "groups": members, "ok": len(groups) <= 1}


def tokenization_identity(manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Per substrate: template md5, tokenizer.json sha256 and the rendered-ids
    hash must agree across arms, else ΔL compares different strings (b1)."""
    per_substrate: dict[str, dict[str, Any]] = {}
    for key, manifest in manifests.items():
        substrate = str(manifest.get("substrate"))
        triple = (
            (manifest.get("template") or {}).get("template_md5"),
            (manifest.get("tokenizer") or {}).get("tokenizer_json_sha256"),
            (manifest.get("rows") or {}).get("rendered_ids_sha256"),
        )
        bucket = per_substrate.setdefault(substrate, {"models": {}, "distinct": {}})
        bucket["models"][key] = {"template_md5": triple[0], "tokenizer_json_sha256": triple[1], "rendered_ids_sha256": triple[2]}
        bucket["distinct"].setdefault(json.dumps(triple), []).append(key)
    for substrate, bucket in per_substrate.items():
        bucket["ok"] = len(bucket["distinct"]) <= 1
        bucket["n_distinct"] = len(bucket["distinct"])
        bucket["distinct"] = sorted(bucket["distinct"].values(), key=lambda g: (-len(g), g))
    return {"substrates": per_substrate, "ok": all(b["ok"] for b in per_substrate.values()) if per_substrate else None}


# ----------------------------------------------------------------- staging
def stage_for_upload(paths: Paths, staging_dir: Path, *, include_results: bool = True) -> dict[str, Any]:
    """Copy evidence, scores (+ manifests, sidecars, logs), the rows and
    (finally) results into ``staging_dir`` — never weights or snapshot caches."""
    rules: list[tuple[Path, str, Callable[[Path], bool]]] = [
        (paths.evidence, "evidence", lambda p: True),
        (paths.scores, "scores", lambda p: p.suffix in (".jsonl", ".json", ".log", ".md", ".npz")),
        (paths.eft_rows_dir, "eft_rows", lambda p: p.suffix in (".jsonl", ".json")),
    ]
    if include_results:
        rules.insert(1, (paths.results, "results", lambda p: True))
    copied: list[dict[str, Any]] = []
    total = 0
    for source, remote, allow in rules:
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if any(part in UPLOAD_DENY_DIRS for part in relative.parts) or path.suffix in UPLOAD_DENY_SUFFIXES or not allow(path):
                continue
            target = staging_dir / remote / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            size = target.stat().st_size
            total += size
            copied.append({"path": str(target.relative_to(staging_dir)), "bytes": size})
    return {"staging_dir": str(staging_dir), "n_files": len(copied), "bytes": total, "files": copied}


# ------------------------------------------------------------------ driver
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
    """A hard precondition failed (fatal; analysis + publish still run)."""


@dataclass
class RunState:
    phases: dict[str, str] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    trims: list[dict[str, Any]] = field(default_factory=list)
    gates: dict[str, Any] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)
    deadline_hit: bool = False
    inputs: dict[str, Any] = field(default_factory=dict)
    hf_revision: str | None = None
    publication: dict[str, Any] = field(default_factory=dict)
    incremental_publications: list[dict[str, Any]] = field(default_factory=list)
    preflight: dict[str, Any] = field(default_factory=dict)
    row_ids: tuple[str, ...] = ()
    running: dict[str, str] = field(default_factory=dict)  # model key -> started_at


class Driver:
    def __init__(self, cfg: DriverConfig, deps: DriverDeps | None = None, *, paths: Paths | None = None, catalog: Catalog | None = None) -> None:
        self.cfg = cfg
        self.deps = default_deps(echo=cfg.echo_subprocess_output) if deps is None else deps
        self.paths = Paths.from_config(cfg) if paths is None else paths
        self.evidence = self.paths.evidence
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.log = DriverLog(self.evidence / LOG_FILE, echo=cfg.echo_subprocess_output)
        self.state = RunState()
        self.python = cfg.python or sys.executable
        self.catalog_path = Path(cfg.models_yaml) if cfg.models_yaml else MODELS_YAML
        self.catalog = load_catalog(self.catalog_path) if catalog is None else catalog
        self.entries = self._select_entries()
        self.noise_keys = self._noise_keys()
        self.run_id, started = self._anchor()
        self.deadline = Deadline(started, cfg.wall_clock_budget_seconds, cfg.publish_reserve_seconds + cfg.analysis_reserve_seconds, self.deps.now)
        self.evictor: Any = None
        self._publish_lock = asyncio.Lock()
        self._heartbeat_task: asyncio.Task | None = None

    # ---- bookkeeping -----------------------------------------------------
    def _select_entries(self) -> list[ModelEntry]:
        entries = self.catalog.entries()
        known = {e.key for e in self.catalog.models}
        for key in (*self.cfg.only_models, *self.cfg.skip_models):
            if key not in known:
                raise ValueError(f"only_models/skip_models: unknown model key {key!r}")
        if self.cfg.only_models:
            wanted = set(self.cfg.only_models)
            entries = [e for e in entries if e.key in wanted]
        entries = [e for e in entries if e.key not in set(self.cfg.skip_models)]
        if not entries:
            raise ValueError("no models selected")
        return entries

    def _noise_keys(self) -> set[str]:
        keys: set[str] = set()
        for item in self.cfg.noise_models:
            if item == "primary_controls":
                keys |= {e.key for e in self.catalog.primary_controls.values()}
            else:
                keys.add(item)
        return keys

    def _anchor(self) -> tuple[str, float]:
        path = self.evidence / STARTED_FILE
        if self.cfg.resume and path.is_file():
            previous = read_json(path)
            if not self.cfg.run_id or previous.get("run_id") == self.cfg.run_id:
                return str(previous["run_id"]), float(previous["started_epoch"])
        started = self.deps.now()
        run_id = self.cfg.run_id or timestamp_tag(started)
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
        """A descriptive gate: recorded (receipt + sentinel), never a stop."""
        verdict = {"passed": passed, **payload}
        self.state.gates[name] = verdict
        write_json(self.evidence / f"gate_{name}.json", make_receipt(self.run_id, f"gate_{name}", "ok" if passed else ("gate-failed" if passed is False else "partial"), **verdict))
        self.log(f"{GATE_SENTINEL} {name} {'PASS' if passed else ('FAIL' if passed is False else 'INCONCLUSIVE')}: {payload.get('summary', '')}")
        return verdict

    def job(self, name: str, argv: Sequence[str], *, gpus: Sequence[int], config: Mapping[str, Any] | None = None, config_env: str | None = None, timeout_seconds: float | None = None) -> Job:
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
        config_path: str | None = None
        if config is not None:
            config_path = str(write_json(self.evidence / "configs" / f"{name}.json", config))
            job_env[config_env or "SCIMT_JOB_CONFIG"] = config_path
        return Job(name=name, argv=tuple(argv), env=job_env, gpus=tuple(gpus), log_path=str(self.evidence / f"{name}.log"), cwd=str(self.paths.repo_root), timeout_seconds=timeout_seconds, config_path=config_path)

    async def run(self, job: Job) -> JobResult:
        self.log(f"launch {job.name}: gpus={list(job.gpus)} argv={' '.join(job.argv)} timeout={job.timeout_seconds}")
        result = await self.deps.run_job(job)
        if result.status != "ok":
            self.log(f"{job.name} exited {result.exit_code} ({result.status}); tail:\n" + "\n".join(result.tail[-8:]))
        return result

    def job_payload(self, job: Job, result: JobResult) -> dict[str, Any]:
        return {"exit_code": result.exit_code, "seconds": result.seconds, "gpu_peak_gb": result.gpu_peak_gb, "gpus": list(job.gpus), "argv": list(job.argv), "log_path": job.log_path, "config_path": job.config_path, "tail": result.tail[-25:], "started_at": result.started_at, "finished_at": result.finished_at, "timed_out": result.timed_out}

    # ---- heartbeat -------------------------------------------------------------
    def heartbeat(self, phase: str) -> None:
        payload = {"ts": utc_now(), "run_id": self.run_id, "phase": phase, "elapsed_seconds": self.deadline.elapsed(), "running": dict(self.state.running), "models_ok": sum(1 for v in self.state.models.values() if v == "ok"), "models_total": len(self.entries), "models": dict(self.state.models)}
        write_json(self.evidence / HEARTBEAT_FILE, payload)

    async def _heartbeat_loop(self) -> None:
        while True:
            with contextlib.suppress(Exception):  # never let the heartbeat kill the run
                self.heartbeat(next((p for p, s in reversed(list(self.state.phases.items())) if s == "running"), "running"))
            await asyncio.sleep(self.cfg.heartbeat_seconds)

    # ---- phase 0 ----------------------------------------------------------------
    async def phase0_preflight(self) -> None:
        phase = "preflight"
        inputs_path = self.evidence / INPUTS_FILE
        if (prior := self.receipt_ok(phase)) is not None and inputs_path.is_file() and (self.evidence / MODELS_FILE).is_file():
            self.state.inputs = read_json(inputs_path)
            self.state.preflight = prior.get("probe", {})
            self.state.hf_revision = str(read_json(self.evidence / MODELS_FILE)["hf_revision"])
            self.state.row_ids = tuple(str(r) for r in self.state.inputs["eft_rows"]["row_ids"])
            self.state.phases[phase] = "ok"
            self.log(f"{phase}: already ok for run {self.run_id}; inputs from {inputs_path}")
            if self.cfg.hf_transfer and self.state.inputs.get("hf_transfer", {}).get("enabled"):
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            self._start_evictor()
            return
        cfg = self.cfg
        self.state.phases[phase] = "running"
        self.paths.hf_home.mkdir(parents=True, exist_ok=True)
        self.paths.snapshots.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.paths.hf_home))
        # host
        ram = self.deps.host_ram_gb()
        free_disk = self.deps.disk_free_gb(str(self.paths.root))
        host = {"ram_gb": ram, "free_disk_gb": free_disk}
        if ram < cfg.min_host_ram_gb:
            raise GateFailure(f"host RAM {ram:.0f} GB < {cfg.min_host_ram_gb:.0f} GB")
        if free_disk < cfg.min_free_disk_gb:
            message = f"free disk {free_disk:.0f} GB < {cfg.min_free_disk_gb:.0f} GB (two GLM snapshots in flight ≈ 430 GB + prefetch headroom)"
            if cfg.disk_gate == "fail":
                raise GateFailure(message)
            self.note(message)
        # torch / cuda probe
        probe_job = self.job("preflight_probe", [self.python, "-c", PREFLIGHT_SNIPPET], gpus=cfg.gpus())
        result = await self.run(probe_job)
        write_json(self.evidence / "preflight_probe.json", make_receipt(self.run_id, "preflight_probe", result.status, **self.job_payload(probe_job, result)))
        if result.status != "ok":
            raise GateFailure(f"preflight probe failed (exit {result.exit_code})")
        probe = self._parse_preflight(result.tail)
        if not probe.get("cuda_available"):
            raise GateFailure(f"torch.cuda.is_available() is False (torch {probe.get('torch')} cuda {probe.get('torch_cuda')}): cu128 wheel trap — see README")
        if int(probe.get("cuda_devices", 0)) < cfg.n_gpus:
            raise GateFailure(f"{probe.get('cuda_devices')} CUDA devices visible, need {cfg.n_gpus}")
        if not all(g.get("matmul_ok") for g in probe.get("gpus", [])):
            raise GateFailure(f"per-GPU matmul check failed: {probe.get('gpus')}")
        if not probe.get("accelerate") and any(cfg.device_map.get(e.substrate) for e in self.entries):
            raise GateFailure("accelerate is not installed but device_map=auto is configured (GLM needs it)")
        tv = str(probe.get("transformers") or "0")
        if tuple(int(x) for x in tv.split(".")[:2] if x.isdigit()) < (5, 9):
            raise GateFailure(f"transformers {tv} < 5.9 (the checkpoints were saved by 5.9.0: TokenizersBackend + chat_template.jinja)")
        self.state.preflight = probe
        # hf transfer
        transfer = {"requested": cfg.hf_transfer, "available": self.deps.hf_transfer_available(), "enabled": False}
        if cfg.hf_transfer and transfer["available"]:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            transfer["enabled"] = True
        # revision + listing
        t0 = self.deps.now()
        revision = self.deps.resolve_revision(self.catalog.hf_repo, revision=cfg.hf_revision or self.catalog.hf_revision, repo_type=self.catalog.hf_repo_type)
        if not revision:
            raise GateFailure("could not resolve the checkpoint repo revision")
        self.state.hf_revision = revision
        listing = self.deps.list_repo_files(self.catalog.hf_repo, revision=revision, repo_type=self.catalog.hf_repo_type)
        models_record = self._verify_listing(listing, revision)
        self.log(f"checkpoint repo {self.catalog.hf_repo}@{revision[:12]}: {len(listing)} files, {len(self.entries)} models verified in {self.deps.now() - t0:.0f}s")
        # eft rows
        t0 = self.deps.now()
        rows_path = self.deps.write_eft_rows(str(self.paths.eft_rows), n_conflict_episodes=cfg.n_conflict_episodes, n_agreement_episodes=cfg.n_agreement_episodes, seed=cfg.eft_seed)
        rows_manifest = read_json(Path(rows_path).parent / "manifest.json") if (Path(rows_path).parent / "manifest.json").is_file() else {}
        rows = load_eft_rows(rows_path)
        counts = describe_row_counts(rows, expected_group_counts(cfg.n_conflict_episodes, cfg.n_agreement_episodes))
        if not counts["ok"]:
            self.note(f"EFT rows: class counts differ from the expectation (descriptive, not a gate): {json.dumps(counts['mismatches'])}, n_rows {counts['n_rows']}")
        self.state.row_ids = tuple(r.row_id for r in rows)
        self.log(f"EFT rows: {len(rows)} rows {json.dumps(counts['observed'])} paired episodes {json.dumps(counts['paired_episodes'])} in {self.deps.now() - t0:.0f}s")
        self.state.inputs = {
            "hf_revision": revision,
            "eft_rows": {"path": str(rows_path), "sha256": sha256_file(rows_path), "counts": counts, "manifest": rows_manifest, "row_ids": list(self.state.row_ids)},
            "host": host,
            "probe": probe,
            "hf_transfer": transfer,
            "models": [e.key for e in self.entries],
            "noise_models": sorted(self.noise_keys),
            "fp32_check_model": cfg.fp32_check_model,
        }
        write_json(inputs_path, self.state.inputs)
        self.receipt(phase, "ok", host=host, probe=probe, hf_revision=revision, hf_transfer=transfer, eft_rows={"path": str(rows_path), "n_rows": len(rows), "counts_ok": counts["ok"]}, models=models_record["summary"])
        self.state.phases[phase] = "ok"
        self._start_evictor()

    @staticmethod
    def _parse_preflight(tail: Sequence[str]) -> dict[str, Any]:
        for line in reversed(list(tail)):
            if line.startswith(PREFLIGHT_PREFIX):
                return json.loads(line[len(PREFLIGHT_PREFIX):])
        raise GateFailure("preflight probe printed no SCIMT-PREFLIGHT line")

    def _verify_listing(self, listing: Sequence[str], revision: str) -> dict[str, Any]:
        """Every selected checkpoint dir must hold config.json + *.safetensors at
        the pinned revision; write evidence/models.json (catalog + revision +
        local snapshot path when one is already on disk)."""
        files = set(listing)
        records: list[dict[str, Any]] = []
        missing: list[str] = []
        for entry in self.entries:
            prefix = entry.hf_path + "/"
            shards = sorted(f[len(prefix):] for f in files if f.startswith(prefix) and f.endswith(".safetensors"))
            has_config = f"{entry.hf_path}/config.json" in files
            if not shards or not has_config:
                missing.append(entry.key)
            model_dir = self.paths.model_dir(entry)
            records.append({**entry.to_dict(), "hf_revision": revision, "n_safetensors": len(shards), "safetensors": shards, "has_config": has_config, "has_chat_template": f"{entry.hf_path}/chat_template.jinja" in files, "local_snapshot_path": str(model_dir) if snapshot_has_weights(model_dir) else None, "noise_pass": entry.key in self.noise_keys, "fp32_check": entry.key == self.cfg.fp32_check_model, "expected_architecture": self.catalog.expected_architecture(entry.substrate), "expected_layers": self.catalog.expected_layers(entry.substrate)})
        if missing:
            candidates = sorted({f.rsplit("/", 1)[0] for f in files if f.endswith(".safetensors") and f.rsplit("/", 1)[0].endswith("/base")})
            raise GateFailure(f"{len(missing)} checkpoint dirs lack config.json or *.safetensors in {self.catalog.hf_repo}@{revision}: {missing}; */base dirs with weights ({len(candidates)}): {candidates[:12]}{' ...' if len(candidates) > 12 else ''}")
        payload = {
            "schema": "midtrain_delta_loss_scaling_v1/models_json/2",
            "hf_repo": self.catalog.hf_repo,
            "hf_repo_type": self.catalog.hf_repo_type,
            "hf_revision": revision,
            "resolved_at": utc_now(),
            "catalog_path": str(self.catalog_path),
            "catalog_sha256": sha256_file(self.catalog_path) if self.catalog_path.is_file() else None,
            "substrates": {k: dict(v) for k, v in self.catalog.substrates.items()},
            "n_models": len(records),
            "models": records,
            "summary": {"n_models": len(records), "by_role": {role: sum(1 for r in records if r["role"] == role) for role in sorted({r["role"] for r in records})}, "by_priority": {str(p): sum(1 for r in records if r["priority"] == p) for p in sorted({r["priority"] for r in records})}},
        }
        write_json(self.evidence / MODELS_FILE, payload)
        return payload

    def _start_evictor(self) -> None:
        if not self.cfg.evictor or self.evictor is not None:
            return
        try:
            self.evictor = self.deps.spawn_evictor(self.python, str(self.paths.script("cache_evictor.py")), [str(self.paths.snapshots), str(self.paths.hf_home)], str(self.evidence / "cache_evictor.log"), self.cfg.evict_threshold_gb)
            self.log(f"cache evictor started (roots {self.paths.snapshots}, {self.paths.hf_home}; threshold {self.cfg.evict_threshold_gb:.0f} GB)")
        except Exception as error:  # noqa: BLE001 — the evictor is a performance guard, not a requirement
            self.note(f"cache evictor did not start: {error!r}")

    # ---- phase 1: planning helpers --------------------------------------------------
    def approx_bytes(self, entry: ModelEntry) -> int:
        return int(self.cfg.approx_bytes.get(entry.substrate) or self.catalog.approx_bytes(entry.substrate))

    def expected_seconds(self, entry: ModelEntry) -> float:
        measured = self.state.measurements.get("model_seconds", {}).get(entry.substrate)
        return float(measured if measured else self.cfg.expected_model_seconds.get(entry.substrate, 1800.0))

    def download_gbps(self) -> float:
        return float(self.state.measurements.get("download_gbps") or self.cfg.expected_download_gbps)

    def gpus_needed(self, entry: ModelEntry) -> int:
        return self.cfg.n_gpus if self.cfg.device_map.get(entry.substrate) else 1

    def plan(self, entry: ModelEntry, *, downloading: bool = False) -> dict[str, Any]:
        return plan_model(self.approx_bytes(entry), downloaded=downloading or snapshot_has_weights(self.paths.model_dir(entry)), download_gbps=self.download_gbps(), model_seconds=self.expected_seconds(entry), remaining_seconds=self.deadline.remaining_for_work())

    def trim(self, entry: ModelEntry, plan: Mapping[str, Any]) -> None:
        record = {"model": entry.key, "priority": entry.priority, **plan}
        self.state.trims.append(record)
        self.state.deadline_hit = True
        self.state.models[entry.key] = "skipped"
        self.state.skipped.append({"phase": f"score__{entry.tag}", "reason": "wall clock", "deliberate": False, "model": entry.key})
        self.log(f"{TRIM_SENTINEL} {entry.key} (priority {entry.priority}): needs ≈{plan['projected_seconds']:.0f}s (download {plan['download_seconds']:.0f}s + score {plan['score_seconds']:.0f}s), {plan['remaining_seconds']:.0f}s left")
        self.receipt(f"score__{entry.tag}", "skipped", reason="wall clock", deliberate=False, model=entry.to_dict(), plan=dict(plan))
        self.log(f"{MODEL_SENTINEL} {entry.key} status=skipped")

    # ---- phase 1: snapshots ---------------------------------------------------------
    def _download(self, entry: ModelEntry) -> dict[str, Any]:
        """Blocking: hub listing (sizes + LFS sha256s) + snapshot_download of one
        checkpoint dir into <root>/snapshots/<tag>/ + verification."""
        model_dir = self.paths.model_dir(entry)
        t0 = self.deps.now()
        hub_files = self.deps.list_repo_tree(entry.hf_repo, path_in_repo=entry.hf_path, revision=self.state.hf_revision, repo_type=self.catalog.hf_repo_type)
        reused = False
        if snapshot_has_weights(model_dir):
            try:
                verify_snapshot(model_dir, hub_files)
                reused = True
            except RuntimeError as error:  # incomplete -> let the hub finish it
                log(f"snapshot {entry.key} on disk but incomplete ({str(error)[:200]}); resuming the download")
        if not reused:
            self.paths.snapshot_dir(entry).mkdir(parents=True, exist_ok=True)
            self.deps.snapshot_download(entry.hf_repo, revision=self.state.hf_revision, allow_patterns=[f"{entry.hf_path}/*"], local_dir=str(self.paths.snapshot_dir(entry)), repo_type=self.catalog.hf_repo_type)
        info = verify_snapshot(model_dir, hub_files)
        seconds = max(self.deps.now() - t0, 1e-6)
        return {**info, "seconds": 0.0 if reused else seconds, "reused": reused, "gbps": None if reused else info["bytes"] / GB / seconds, "hub_files": hub_files, "hf_revision": self.state.hf_revision}

    def _ensure_disk(self, entry: ModelEntry, keep: Sequence[str]) -> None:
        """Purge stale snapshots of models that are neither running nor queued
        for download when the next download would not fit."""
        needed_gb = self.approx_bytes(entry) * 1.1 / GB + 20.0
        free = self.deps.disk_free_gb(str(self.paths.root))
        if free >= needed_gb:
            return
        for other in self.entries:
            if other.key in keep or other.key == entry.key:
                continue
            if snapshot_has_weights(self.paths.model_dir(other)):
                freed = remove_snapshot(self.paths.snapshot_dir(other))
                self.note(f"disk: removed {other.key} snapshot ({freed['freed_bytes'] / GB:.0f} GB) to make room for {entry.key} (free {free:.0f} GB < {needed_gb:.0f} GB needed)")
                free = self.deps.disk_free_gb(str(self.paths.root))
                if free >= needed_gb:
                    return
        self.note(f"disk: only {free:.0f} GB free for {entry.key} (needs ≈{needed_gb:.0f} GB); attempting the download anyway")

    def _check_stale_outputs(self, entry: ModelEntry) -> None:
        """Scores from another repo revision must never be resumed into: move them aside."""
        manifest_path = self.paths.losses_manifest(entry)
        if not manifest_path.is_file():
            return
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError):
            return
        if manifest.get("hf_revision") in (None, self.state.hf_revision):
            return
        stamp = timestamp_tag(self.deps.now())
        for path in (self.paths.losses(entry), manifest_path, self.paths.tokens(entry), self.paths.noise(entry)):
            if path.exists():
                path.rename(path.with_name(path.name + f".stale-{stamp}"))
        self.note(f"{entry.key}: existing scores came from revision {str(manifest.get('hf_revision'))[:12]} != {str(self.state.hf_revision)[:12]}; moved aside as *.stale-{stamp}")

    def render_score_config(self, entry: ModelEntry, model_dir: Path) -> dict[str, Any]:
        cfg = self.cfg
        noise = entry.key in self.noise_keys and cfg.repeat_rows > 0
        payload: dict[str, Any] = {
            "model_dir": str(model_dir),
            "rows_path": str(self.paths.eft_rows),
            "out_path": str(self.paths.losses(entry)),
            "manifest_path": str(self.paths.losses_manifest(entry)),
            "tokens_out_path": str(self.paths.tokens(entry)),
            "noise_out_path": str(self.paths.noise(entry)) if noise else None,
            "profile": entry.profile,
            "arm": entry.arm,
            "substrate": entry.substrate,
            "dose_tokens": entry.dose_tokens,
            "hf_repo": entry.hf_repo,
            "hf_revision": self.state.hf_revision,
            "hf_path": entry.hf_path,
            "expected_architecture": self.catalog.expected_architecture(entry.substrate),
            "expected_layers": self.catalog.expected_layers(entry.substrate),
            "device": "cuda:0",
            "device_map": cfg.device_map.get(entry.substrate),
            "attn_implementation": cfg.attn_implementation.get(entry.substrate),
            "experts_implementation": cfg.experts_implementation.get(entry.substrate),
            "load_fallback": cfg.load_fallback,
            "batch_size": cfg.batch_size,
            "batch_check_rows": cfg.batch_check_rows,
            "batch_check_abs_tol": cfg.batch_check_abs_tol,
            "repeat_rows": cfg.repeat_rows if noise else 0,
            "repeat_seed": cfg.repeat_seed,
            "fp32_check_rows": cfg.fp32_check_rows if entry.key == cfg.fp32_check_model else 0,
            "sanity_max_content_ce": cfg.sanity_max_content_ce,
            "require_constant_template_ids": cfg.require_constant_template_ids,
            "max_tokens": cfg.max_tokens,
            "expected_rows": len(self.state.row_ids),
            "resume": True,
        }
        payload = merge_mappings(payload, cfg.score_overrides)
        rl.RowLossesConfig.from_mapping(payload)  # validate before spending GPU time
        return payload

    def _measure(self, entry: ModelEntry, result: JobResult, download: Mapping[str, Any]) -> None:
        if not download.get("reused") and download.get("gbps") and int(download.get("bytes") or 0) >= MIN_BYTES_FOR_THROUGHPUT:
            gbps = float(download["gbps"])
            first = "download_gbps" not in self.state.measurements
            self.state.measurements["download_gbps"] = gbps
            self.state.measurements.setdefault("downloads", {})[entry.key] = {"gbps": gbps, "bytes": download.get("bytes"), "seconds": download.get("seconds")}
            if first and "download_throughput" not in self.state.gates:
                self.gate("download_throughput", gbps >= self.cfg.min_download_gbps, summary=f"first snapshot {download.get('bytes', 0) / GB:.1f} GB at {gbps * 1000:.0f} MB/s (gate >= {self.cfg.min_download_gbps * 1000:.0f} MB/s; below it, re-roll the host)", gbps=gbps, model=entry.key)
        if result.status == "ok":
            self.state.measurements.setdefault("model_seconds", {})[entry.substrate] = float(result.seconds)
        manifest_path = self.paths.losses_manifest(entry)
        if manifest_path.is_file():
            try:
                manifest = read_json(manifest_path)
            except (OSError, ValueError):
                return
            rate = (manifest.get("scoring") or {}).get("rows_per_s")
            if rate:
                self.state.measurements.setdefault("rows_per_s", {})[entry.key] = float(rate)

    def _finalize_snapshot(self, entry: ModelEntry, download: Mapping[str, Any]) -> dict[str, Any]:
        """Small files + hub listing -> evidence/checkpoint_files/<tag>/; then rmtree."""
        target = self.paths.checkpoint_files(entry)
        kept = preserve_small_files(self.paths.model_dir(entry), target)
        write_json(target / "hub_files.json", {"hf_repo": entry.hf_repo, "hf_revision": self.state.hf_revision, "hf_path": entry.hf_path, "files": list(download.get("hub_files") or [])})
        removed = remove_snapshot(self.paths.snapshot_dir(entry)) if self.cfg.delete_snapshots else {"freed_bytes": 0, "removed_dir": None, "remaining": True}
        return {"kept_files": kept, "kept_dir": str(target), **removed}

    def _amend_manifest(self, entry: ModelEntry, download: Mapping[str, Any]) -> None:
        path = self.paths.losses_manifest(entry)
        if not path.is_file():
            return
        try:
            manifest = read_json(path)
        except (OSError, ValueError):
            return
        manifest["hub_files"] = list(download.get("hub_files") or [])
        manifest["snapshot"] = {k: v for k, v in download.items() if k not in ("hub_files", "shards")}
        write_json(path, manifest)

    async def _publish_incremental(self, reason: str) -> None:
        """scores/ + evidence/ -> runs/<run_id>/ after every model (serialised; never fatal)."""
        if not (self.cfg.upload and self.cfg.publish_incremental):
            return
        if not self.deps.hf_token_available():
            if not any(p.get("reason") == "no HF token" for p in self.state.incremental_publications):
                self.state.incremental_publications.append({"status": "skipped", "reason": "no HF token"})
            return
        async with self._publish_lock:
            staging = self.paths.staging / f"{self.run_id}__incremental"
            try:
                if staging.exists():
                    shutil.rmtree(staging)
                staging.mkdir(parents=True)
                manifest = stage_for_upload(self.paths, staging, include_results=False)
                report = await asyncio.to_thread(self.deps.upload_folder, str(staging), self.cfg.hf_dataset_repo, f"runs/{self.run_id}")
                self.state.incremental_publications.append({"status": "ok", "reason": reason, "n_files": manifest["n_files"], "bytes": manifest["bytes"], "at": utc_now(), "url": dict(report).get("url")})
                self.log(f"incremental publish ({reason}): {manifest['n_files']} files, {manifest['bytes'] / 1e6:.1f} MB -> {self.cfg.hf_dataset_repo}/runs/{self.run_id}")
            except Exception as error:  # noqa: BLE001 — incremental publish is best effort
                self.state.incremental_publications.append({"status": "failed", "reason": reason, "error": repr(error), "at": utc_now()})
                self.note(f"incremental publish failed ({reason}): {error!r}")

    # ---- phase 1: one model ---------------------------------------------------------
    async def score_model(self, entry: ModelEntry, gpus: tuple[int, ...], plan: Mapping[str, Any], prefetch: asyncio.Task | None, keep_keys: Callable[[], set[str]]) -> None:
        """Download (or take the prefetched files) -> score -> verify -> preserve
        small files + rmtree -> receipt -> incremental publish."""
        name = f"score__{entry.tag}"
        self.state.running[entry.key] = utc_now()
        self.state.models[entry.key] = "running"
        download: dict[str, Any]
        try:
            if prefetch is not None:
                try:
                    download = await prefetch
                except Exception as error:  # noqa: BLE001 — prefetch is an optimisation; retry inline
                    self.note(f"prefetch of {entry.key} failed ({error!r}); downloading inline")
                    self._ensure_disk(entry, keep=keep_keys())
                    download = await asyncio.to_thread(self._download, entry)
            else:
                self._ensure_disk(entry, keep=keep_keys())
                download = await asyncio.to_thread(self._download, entry)
        except Exception as error:  # noqa: BLE001 — a missing/unusable snapshot fails this model only
            self.state.models[entry.key] = "failed"
            self.state.running.pop(entry.key, None)
            self.fail(name, f"snapshot: {error!r}")
            self.receipt(name, "failed", reason=f"snapshot: {error!r}", model=entry.to_dict(), plan=dict(plan))
            self.log(f"{MODEL_SENTINEL} {entry.key} status=failed")
            return
        rate = download.get("gbps")
        self.log(f"snapshot {entry.key}: {download['n_safetensors']} shards, {download['bytes'] / GB:.1f} GB in {download['seconds']:.0f}s{' (reused)' if download.get('reused') else f' ({(rate or 0) * 1000:.0f} MB/s)'}; {len(download.get('hub_files') or [])} hub files listed")
        self._check_stale_outputs(entry)
        model_dir = self.paths.model_dir(entry)
        config = self.render_score_config(entry, model_dir)
        timeout = max(self.cfg.job_timeout_min_seconds, self.cfg.job_timeout_factor * self.expected_seconds(entry))
        job = self.job(name, [self.python, str(self.paths.script("row_losses.py"))], gpus=gpus, config=config, config_env=rl.CONFIG_ENV, timeout_seconds=timeout)
        result = await self.run(job)
        self._measure(entry, result, download)
        verification = verify_scores(self.paths.losses(entry), self.state.row_ids, entry)
        manifest_path = self.paths.losses_manifest(entry)
        manifest = read_json(manifest_path) if manifest_path.is_file() else None
        if result.status != "ok":
            status = result.status if result.status in RECEIPT_STATUSES else "failed"
            self.fail(name, f"row_losses exited {result.exit_code} ({result.status})")
        elif not verification["ok"]:
            status = "partial"
            self.note(f"{entry.key}: scorer exited 0 but the file is incomplete/inconsistent ({verification['n_records']}/{verification['n_expected']} rows, missing {verification['n_missing']}, extra {verification['n_extra']}, bad keys {verification['bad_keys']}, nonfinite {verification['nonfinite']}) — descriptive; a re-run resumes it")
        else:
            status = "ok"
        for warning in (manifest or {}).get("warnings") or []:
            self.note(f"{entry.key}: scorer warning: {warning}")
        snapshot_record = None
        if status == "ok":
            snapshot_record = self._finalize_snapshot(entry, download)
            self._amend_manifest(entry, download)
            self.log(f"{entry.key}: kept {len(snapshot_record['kept_files'])} small files in evidence, removed snapshot ({snapshot_record['freed_bytes'] / GB:.1f} GB)")
        self.state.models[entry.key] = status
        self.state.running.pop(entry.key, None)
        summary = None if manifest is None else {"template_md5": (manifest.get("template") or {}).get("template_md5"), "tokenizer_json_sha256": (manifest.get("tokenizer") or {}).get("tokenizer_json_sha256"), "rendered_ids_sha256": (manifest.get("rows") or {}).get("rendered_ids_sha256"), "means": (manifest.get("scoring") or {}).get("means"), "noise": manifest.get("noise"), "fp32_check": manifest.get("fp32_check"), "sanity": manifest.get("sanity"), "batch_check": manifest.get("batch_check"), "model": {k: v for k, v in (manifest.get("model") or {}).items() if k in ("model_class", "attn_implementation", "experts_implementation", "experts_module_class", "load_fallback_used", "loading_info")}, "code_commit": manifest.get("code_commit"), "warnings": manifest.get("warnings")}
        self.receipt(name, status, **self.job_payload(job, result), model=entry.to_dict(), hf_revision=self.state.hf_revision, code_commit=(manifest or {}).get("code_commit"), template_md5=(summary or {}).get("template_md5"), download={k: v for k, v in download.items() if k not in ("shards", "hub_files")}, verification=verification, manifest_path=str(manifest_path) if manifest is not None else None, manifest_summary=summary, snapshot=snapshot_record, plan=dict(plan))
        self.log(f"{MODEL_SENTINEL} {entry.key} status={status} rows={verification['n_records']}/{verification['n_expected']} seconds={result.seconds:.0f}")
        self.heartbeat("score")
        await self._publish_incremental(entry.key)

    # ---- phase 1: the queue ----------------------------------------------------------
    async def phase1_score(self) -> None:
        phase = "score"
        if self.receipt_ok(phase) is not None:
            self.state.phases[phase] = "ok"
            self.state.models.update({k: "ok" for k in read_json(self.evidence / f"{phase}.json").get("models", {})})
            self.log(f"{phase}: already ok")
            self._identity_gates()
            return
        self.state.phases[phase] = "running"
        pending: list[ModelEntry] = []
        for entry in self.entries:
            prior = self.receipt_ok(f"score__{entry.tag}")
            if prior is not None and prior.get("hf_revision") == self.state.hf_revision:
                self.state.models[entry.key] = "ok"
                self.log(f"score__{entry.tag}: already ok (resume)")
            else:
                pending.append(entry)
        free_gpus: set[int] = set(self.cfg.gpus())
        running: dict[str, tuple[asyncio.Task, ModelEntry, tuple[int, ...]]] = {}
        downloads: dict[str, asyncio.Task] = {}

        def keep_keys() -> set[str]:
            return set(running) | set(downloads) | {e.key for e in pending[: self.cfg.prefetch_depth + 1]}

        while pending or running:
            # launch from the head of the queue while the slots allow; never jump the head
            while pending:
                head = pending[0]
                plan = self.plan(head, downloading=head.key in downloads)
                if not plan["fits"]:
                    pending.pop(0)
                    task = downloads.pop(head.key, None)
                    if task is not None:
                        task.cancel()
                    self.trim(head, plan)
                    continue
                need = self.gpus_needed(head)
                if need > len(free_gpus):
                    break
                pending.pop(0)
                gpus = tuple(sorted(free_gpus))[:need]
                free_gpus -= set(gpus)
                task = asyncio.create_task(self.score_model(head, gpus, plan, downloads.pop(head.key, None), keep_keys))
                running[head.key] = (task, head, gpus)
            # prefetch the next snapshots while the slots are busy
            self._schedule_prefetch(pending, downloads, keep_keys)
            if not running:
                if pending:  # cannot happen (all slots free) — guard against an infinite loop
                    raise RuntimeError("scheduler stalled with free GPUs and pending models")
                break
            done, _ = await asyncio.wait([t for t, _, _ in running.values()], return_when=asyncio.FIRST_COMPLETED)
            for key in [k for k, (t, _, _) in running.items() if t in done]:
                task, entry, gpus = running.pop(key)
                free_gpus |= set(gpus)
                try:
                    task.result()
                except Exception as error:  # noqa: BLE001 — score_model handles its own errors; this is the last net
                    self.state.models[entry.key] = "failed"
                    self.state.running.pop(entry.key, None)
                    self.fail(f"score__{entry.tag}", f"unhandled: {error!r}")
                    self.receipt(f"score__{entry.tag}", "failed", reason=f"unhandled: {error!r}", model=entry.to_dict())
        for task in downloads.values():
            task.cancel()
        statuses = {e.key: self.state.models.get(e.key, "skipped") for e in self.entries}
        failed = [k for k, v in statuses.items() if v in ("failed", "timeout")]
        partial = [k for k, v in statuses.items() if v in ("partial", "skipped")]
        status = "ok" if not failed and not partial else ("failed" if failed and len(failed) == len(statuses) else "partial")
        self._identity_gates()
        self.receipt(phase, status, models=statuses, n_ok=sum(1 for v in statuses.values() if v == "ok"), failed=failed, partial_or_skipped=partial, measurements=dict(self.state.measurements), gates={k: v.get("passed") for k, v in self.state.gates.items()})
        self.state.phases[phase] = status

    def _schedule_prefetch(self, pending: Sequence[ModelEntry], downloads: dict[str, asyncio.Task], keep_keys: Callable[[], set[str]]) -> None:
        if self.cfg.prefetch_depth <= 0:
            return
        in_flight_gb = sum(self.approx_bytes(self.catalog.by_key(k)) for k in downloads) / GB
        for entry in list(pending[: self.cfg.prefetch_depth]):
            if entry.key in downloads or snapshot_has_weights(self.paths.model_dir(entry)):
                continue
            free = self.deps.disk_free_gb(str(self.paths.root))
            needed = self.approx_bytes(entry) / GB
            if free - in_flight_gb - needed < self.cfg.prefetch_min_free_disk_gb:
                break
            self.log(f"prefetching {entry.key} ({needed:.0f} GB; free {free:.0f} GB, {in_flight_gb:.0f} GB in flight)")
            downloads[entry.key] = asyncio.create_task(asyncio.to_thread(self._download, entry))
            in_flight_gb += needed

    def _identity_gates(self) -> None:
        """Descriptive: config.json identity across arms (minus generation keys) per
        profile and per substrate; tokenization identity per substrate."""
        if "config_identity" in self.state.gates:
            return
        configs: dict[str, dict[str, Any]] = {}
        manifests: dict[str, dict[str, Any]] = {}
        for entry in self.entries:
            path = self.paths.checkpoint_files(entry) / "config.json"
            if path.is_file():
                try:
                    configs[entry.key] = read_json(path)
                except (OSError, ValueError):
                    pass
            manifest_path = self.paths.losses_manifest(entry)
            if manifest_path.is_file() and self.state.models.get(entry.key) == "ok":
                try:
                    manifests[entry.key] = read_json(manifest_path)
                except (OSError, ValueError):
                    pass
        per_profile = {}
        for profile in sorted({e.profile for e in self.entries}):
            subset = {k: v for k, v in configs.items() if k.startswith(profile + "/")}
            if len(subset) >= 2:
                per_profile[profile] = config_identity(subset)
        per_substrate = {}
        for substrate in sorted({e.substrate for e in self.entries}):
            subset = {k: v for k, v in configs.items() if self.catalog.by_key(k).substrate == substrate}
            if len(subset) >= 2:
                per_substrate[substrate] = config_identity(subset)
        ok_cfg = all(v["ok"] for v in list(per_profile.values()) + list(per_substrate.values())) if (per_profile or per_substrate) else None
        write_json(self.evidence / "config_identity.json", {"ignored_keys": list(CONFIG_IDENTITY_IGNORE), "per_profile": per_profile, "per_substrate": per_substrate, "ok": ok_cfg})
        self.gate("config_identity", ok_cfg, summary=f"config.json minus generation keys identical across arms in {sum(1 for v in per_profile.values() if v['ok'])}/{len(per_profile)} profiles and {sum(1 for v in per_substrate.values() if v['ok'])}/{len(per_substrate)} substrates", per_profile={k: v["ok"] for k, v in per_profile.items()}, per_substrate={k: v["ok"] for k, v in per_substrate.items()})
        tok = tokenization_identity(manifests)
        write_json(self.evidence / "tokenization_identity.json", tok)
        self.gate("tokenization_identity", tok["ok"], summary="template md5 + tokenizer.json sha256 + rendered-ids sha256 identical across the scored arms of every substrate (ΔL must be refused where not): " + json.dumps({k: v["ok"] for k, v in tok["substrates"].items()}), per_substrate={k: {"ok": v["ok"], "n_models": len(v["models"]), "n_distinct": v["n_distinct"]} for k, v in tok["substrates"].items()})

    # ---- phase 2 ---------------------------------------------------------------
    async def phase2_analysis(self) -> None:
        """``analysis.analyze_scaling.run_all(root, root/'results')`` (CPU; non-fatal)."""
        phase = "analysis"
        if not self.cfg.analysis:
            self.skip(phase, "analysis disabled by config", deliberate=True)
            return
        if not any(self.paths.scores.glob("losses__*.jsonl")):
            self.skip(phase, "no scores/losses__*.jsonl to analyse")
            return
        if self.receipt_ok(phase) is not None and self.paths.results.is_dir():
            self.state.phases[phase] = "ok"
            self.log(f"{phase}: already ok (results in {self.paths.results})")
            return
        module = self.paths.repo_root / "experiments" / "improved_midtraining" / EXPERIMENT_NAME / "analysis" / "analyze_scaling.py"
        if not module.is_file():
            self.state.phases[phase] = "skipped"
            self.state.skipped.append({"phase": phase, "reason": f"{module} absent", "deliberate": False})
            self.receipt(phase, "skipped", reason=f"analysis module absent: {module}")
            return
        self.state.phases[phase] = "running"
        job = self.job(phase, [self.python, "-c", ANALYSIS_SNIPPET, str(self.paths.repo_root), str(self.paths.root)], gpus=(), timeout_seconds=self.cfg.analysis_reserve_seconds * 3)
        result = await self.run(job)
        status = "ok" if result.status == "ok" else "failed"
        self.receipt(phase, status, **self.job_payload(job, result))
        self.state.phases[phase] = status
        if status != "ok":
            self.fail(phase, f"analyze_scaling.run_all exited {result.exit_code} ({result.status})")

    # ---- phase 3 ---------------------------------------------------------------
    async def phase3_publish(self, status: str) -> None:
        phase = "publish"
        staging = self.paths.staging / self.run_id
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        write_json(self.evidence / SUMMARY_FILE, self.done_payload(status, finished=False))
        manifest = stage_for_upload(self.paths, staging, include_results=True)
        write_json(staging / "staging_manifest.json", manifest)
        publication: dict[str, Any] = {"status": "skipped", "staging": {k: v for k, v in manifest.items() if k != "files"}}
        if not self.cfg.upload:
            publication["reason"] = "upload disabled by config"
        elif not self.deps.hf_token_available():
            publication["reason"] = "no HF token (HF_TOKEN unset and no cached login)"
        else:
            try:
                async with self._publish_lock:
                    report = await asyncio.to_thread(self.deps.upload_folder, str(staging), self.cfg.hf_dataset_repo, f"runs/{self.run_id}")
                publication = {"status": "ok", **dict(report), "staging": publication["staging"]}
            except Exception as error:  # noqa: BLE001 — publication failure must not mask the run
                publication = {"status": "failed", "error": repr(error), "staging": publication["staging"]}
                self.fail(phase, f"HF upload failed: {error!r}")
        publication["incremental"] = list(self.state.incremental_publications)
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
            "models": dict(self.state.models),
            "skipped": list(self.state.skipped),
            "failures": list(self.state.failures),
            "notes": list(self.state.notes),
            "trims": list(self.state.trims),
            "gates": dict(self.state.gates),
            "measurements": dict(self.state.measurements),
            "hf_revision": self.state.hf_revision,
            "inputs": {k: v for k, v in self.state.inputs.items() if k in ("hf_revision", "host", "hf_transfer", "models", "noise_models", "fp32_check_model")} | {"eft_rows": {k: v for k, v in self.state.inputs.get("eft_rows", {}).items() if k != "row_ids"}},
            "preflight": dict(self.state.preflight),
            "publication": dict(self.state.publication),
            "config": self.cfg.to_dict(),
            "evidence_dir": str(self.evidence),
        }

    def overall_status(self, fatal: bool) -> str:
        if fatal:
            return "failed"
        lost = [s for s in self.state.skipped if not s.get("deliberate")]
        if lost or self.state.failures or any(v in ("partial", "failed") for v in self.state.phases.values()) or any(v != "ok" for v in self.state.models.values()):
            return "partial"
        return "complete"

    async def run_all(self) -> dict[str, Any]:
        self.log(f"driver start run_id={self.run_id} root={self.paths.root} n_gpus={self.cfg.n_gpus} models={len(self.entries)} budget={self.cfg.wall_clock_budget_seconds / 3600:.1f} h elapsed={self.deadline.elapsed():.0f} s")
        write_json(self.evidence / RESOLVED_CONFIG_FILE, {"run_id": self.run_id, "config": self.cfg.to_dict(), "python": self.python, "repo_root": str(self.paths.repo_root), "models_yaml": str(self.catalog_path), "models": [e.key for e in self.entries], "noise_models": sorted(self.noise_keys)})
        self.heartbeat("start")
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        fatal = False
        try:
            await self.phase0_preflight()
            await self.phase1_score()
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
            await self.phase2_analysis()
        except Exception as error:  # noqa: BLE001
            self.fail("analysis", repr(error))
        status = self.overall_status(fatal)
        try:
            await self.phase3_publish(status)
        except Exception as error:  # noqa: BLE001
            self.fail("publish", repr(error))
        status = self.overall_status(fatal)
        done = self.done_payload(status, finished=True)
        validate_done(done)
        write_json(self.evidence / DONE_FILE, done)
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
        self.heartbeat("done")
        n_ok = sum(1 for v in done["models"].values() if v == "ok")
        self.log(f"{DONE_SENTINEL} status={status} elapsed={done['elapsed_seconds'] / 3600:.2f} h models_ok={n_ok}/{len(self.entries)} skipped={len(done['skipped'])} failures={len(done['failures'])} gates={json.dumps({k: v.get('passed') for k, v in done['gates'].items()})}")
        return done


async def run_driver(cfg: DriverConfig, deps: DriverDeps | None = None, *, paths: Paths | None = None, catalog: Catalog | None = None) -> dict[str, Any]:
    return await Driver(cfg, deps, paths=paths, catalog=catalog).run_all()


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        raise SystemExit(f"usage: run_all.py — config-first, no flags; set ${CONFIG_ENV}=<config.json|yaml>")
    config_path = os.environ.get(CONFIG_ENV) or None
    cfg = load_driver_config(config_path)
    if config_path is not None:
        resolved = Path(config_path).resolve()
        evidence = Path(cfg.root).resolve() / "evidence"
        if resolved == evidence or evidence in resolved.parents:
            raise SystemExit(f"{CONFIG_ENV}={config_path} lives under {evidence}; keep the input config outside evidence/ so the driver can never overwrite it")
    done = asyncio.run(run_driver(cfg))
    return 0 if done["status"] in ("complete", "partial") else 2


if __name__ == "__main__":
    sys.exit(main())
