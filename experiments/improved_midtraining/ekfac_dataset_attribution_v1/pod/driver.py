"""Pod driver for ekfac_dataset_attribution_v1 — the whole 12 h run, supervised.

Runs on one 4xH200 pod (``n_gpus`` configurable; degrades to 2) after
``bootstrap.sh`` has cloned the repo, synced the venv and pre-downloaded the
two Gemma snapshots. Every GPU phase is a SUPERVISED asyncio subprocess of one
of the sibling pod scripts (``fit_factors_pt.py``, ``mean_gradients.py``,
``apply_inverse_gpu.py``, ``score_eft_rows.py``), config-first: the driver
renders each script's own config mapping (validated here against the script's
``from_mapping`` before anything is launched), hands it over as a JSON file
(positional path or the script's ``SCIMT_EKFAC_*_CONFIG`` env var), pins
``CUDA_VISIBLE_DEVICES``, tees stdout+stderr to ``evidence/<job>.log``, polls
``nvidia-smi`` for the device-level peak, and writes a JSON receipt per job.
No flag strings, no fire-and-forget, no in-process CUDA (gate2 lesson: an
in-process phase chain kept GPU memory across phases and OOM'd).

Schedule (T = driver start; PREMORTEM.md §D, gates A–E)
--------------------------------------------------------
- Phase 0 (CPU): ``sample_datasets.build_all`` (six scored corpora + the
  disjoint ``dolmino_fit`` calibration sample) and ``build_eft_rows.write_eft_rows``
  (paired coin/charter + ambiguous/ambiguous_wrong rows). **Gate A**:
  bootstrap receipt present, host RAM, disk plan, every sample + manifest
  sha recorded, rows exactly paired, and the calibration sample packs into
  at least ``samples`` seq-4096 rows for the fit (auto-grows
  ``n_dolmino_fit`` while it does not — a Dolmino doc is ~1.3k tokens, so
  512 docs are usually short of 256 x 4096 tokens).
- Smoke: kronfluence/torch preflight (subprocess, CPU), ``mean_gradients``
  with ``max_rows`` 8 on one dataset (GPU 1), then the scorer on those smoke
  vectors (model on GPU 0, shards on GPUs 1..n-1) — the only place the
  pt/it manifest-digest match, the grad hooks and the shard self-check are
  exercised before the long phases. **Gate B**: measured s/row -> projected
  mean-gradient makespan and main-pass scoring time; a main pass over 2.5 h
  is cut to a per-episode subsample (PREMORTEM: "600 rows/class").
- Phase 1 (parallel): GPU 0 runs ``fit_factors_pt`` (seq 4096); GPUs 1..n-1
  work through the six datasets (dolmino, charter_worked, charter_noex,
  coin, coin_worked, coin_noex), one dataset per GPU, next dataset when a
  GPU frees. Fit exit codes are honoured: 97 (covariance-gate projection
  over budget) -> the PREMORTEM time valve (seq 2048 x 512 samples,
  ``fit_gate_fallback``) or abort per config; 98 (CUDA OOM) -> one retry
  with 8 module partitions, resuming from the eigendecomposition when the
  OOM hit the lambda pass. **Gate C/D** are the fit's own partition-0
  receipts, copied into ``gate_c.json`` / ``gate_d.json``. **Gate E**: fold
  cosine of each dataset's gdp f0/f1 (CPU, windowed) as soon as the dataset
  finishes; cross-dataset gdp cosines from the scorer's Gram after the main
  pass.
- Phase 2: delete ``ekfac_pt/kronfluence/`` (~0.5 TB; ``load_ekfac`` needs
  only the exported factors), ``oracle_check`` (abort the inverse on
  failure), then ``apply_inverse_gpu`` on GPU 0 for the six ``*__gdp__all``
  vectors at dampings {0.01, 0.1, 1} and for the fold vectors at 0.1.
- Phase 3 (scoring, model on GPU 0, vectors sharded over GPUs 1..n-1):
  ``main`` (all rows; 6 gdp + 6 inv0.1), ``sweep`` (300 episodes per
  pair-type; 6 gdpunit + inv0.01/inv1 for the four primary datasets),
  ``folds`` (300 episodes; 12 inv0.1 fold vectors), then modes
  ``pt_mismatch`` (100 episodes, the main vectors) and ``oracle`` (8 rows x
  2 repeats). Every pass respects the resident-vector budget the scorer
  enforces (16 bf16 vectors over 3 shard GPUs at 130 GB each; recomputed
  for fewer GPUs — main-mode passes are chunked, diagnostic passes
  truncated). When the fit or the inverse failed the passes fall back to
  gdp/gdpunit vectors so the run still produces scores.
- Phase 4 (always, even after a fatal error): ``analysis.run_all`` in a
  subprocess, staging of results + receipts + logs + manifests (never
  weights, ``.f32`` vectors or ``.npy`` factors) and upload to the private
  HF dataset repo ``arcadia-impact/scimt-ekfac-dataset-attribution-v1``
  under ``runs/<run_id>/``; optional GCS push of factors + vectors when
  ``gcloud`` is authed. Then ``DRIVER_DONE.json`` — the controller polls
  for it.

Wall clock: ``wall_clock_budget_seconds`` (11 h) from the first start (the
anchor persists across restarts). Once ``deadline - phase4_reserve`` would
be crossed the driver stops launching phases, finishes scoring passes in
priority order main -> pt_mismatch -> oracle -> folds -> sweep, and always
runs Phase 4 on whatever exists. Fatal errors write ``evidence/
driver_failure.txt`` and still publish the evidence.

Resume: rerunning the driver with ``resume: true`` (default) skips every
phase whose receipt says ``ok`` and whose outputs exist; the sub-scripts add
their own fold/row-level resume on top.

Config-first, no argparse: ``__main__`` reads the JSON/YAML mapping named by
``$SCIMT_EKFAC_DRIVER_CONFIG`` (optional) into :class:`DriverConfig`;
unknown keys are a ``ValueError``. Everything with a side effect goes
through :class:`DriverDeps` so the CPU tests run the whole schedule with
fakes (``tests/test_ekfac_dataset_attribution_driver.py``).
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as _dt
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import time
import traceback
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from experiments.improved_midtraining.ekfac_dataset_attribution_v1.pod import (  # noqa: E402
    apply_inverse_gpu as apply_mod,
)
from experiments.improved_midtraining.ekfac_dataset_attribution_v1.pod import (  # noqa: E402
    fit_factors_pt as fit_mod,
)
from experiments.improved_midtraining.ekfac_dataset_attribution_v1.pod import (  # noqa: E402
    mean_gradients as mg_mod,
)
from experiments.improved_midtraining.ekfac_dataset_attribution_v1.pod import (  # noqa: E402
    score_eft_rows as sr_mod,
)

# ------------------------------------------------------------------ constants
CONFIG_ENV = "SCIMT_EKFAC_DRIVER_CONFIG"
EXPERIMENT_NAME = "ekfac_dataset_attribution_v1"

DATASETS: tuple[str, ...] = (
    "dolmino",
    "charter_worked",
    "charter_noex",
    "coin",
    "coin_worked",
    "coin_noex",
)
SWEEP_DATASETS: tuple[str, ...] = ("dolmino", "charter_worked", "charter_noex", "coin")
FIT_DATASET = "dolmino_fit"
FOLDS: tuple[str, ...] = ("f0", "f1")
POOLED = "all"

PT_MODEL_ID = "google/gemma-3-12b-pt"
PT_REVISION = fit_mod.PT_REVISION  # 295efb63…
IT_MODEL_ID = "google/gemma-3-12b-it"
IT_REVISION = "96b6f1eccf38110c56df3a15bffe176da04bfd80"

INCLUDED_NUMEL = mg_mod.EXPECTED_INCLUDED_NUMEL  # 10,759,155,456
LARGEST_ENTRY_NUMEL = 15_360 * 3_840  # gate_proj / up_proj / down_proj weight
VECTOR_BYTES = INCLUDED_NUMEL * 4
GB = 1e9

SCORING_PRIORITY: tuple[str, ...] = ("main", "pt_mismatch", "oracle", "folds", "sweep")
ROWS_PER_EPISODE = 2  # both rows of a label-flip pair
PAIR_TYPES = 2  # coin/charter and ambiguous/ambiguous_wrong

DONE_FILE = "DRIVER_DONE.json"
FAILURE_FILE = "driver_failure.txt"
LOG_FILE = "driver.log"
STARTED_FILE = "driver_started.json"
BOOTSTRAP_FILE = "bootstrap.json"

PHASE_SENTINEL = "SCIMT-DRIVER-PHASE"
GATE_SENTINEL = "SCIMT-DRIVER-GATE"
DONE_SENTINEL = "SCIMT-DRIVER-DONE"
FAIL_SENTINEL = "SCIMT-DRIVER-FAIL"

HOST_SPEC_EXIT_CODE = 96
FIT_GATE_EXIT_CODE = fit_mod.FIT_GATE_EXIT_CODE  # 97
FIT_OOM_EXIT_CODE = fit_mod.FIT_OOM_EXIT_CODE  # 98
ORACLE_EXIT_CODE = apply_mod.ORACLE_EXIT_CODE  # 99

EXIT_STATUS: dict[int, str] = {
    0: "ok",
    HOST_SPEC_EXIT_CODE: "host-spec-failed",
    FIT_GATE_EXIT_CODE: "gate-failed",
    FIT_OOM_EXIT_CODE: "oom",
    ORACLE_EXIT_CODE: "oracle-failed",
}
RECEIPT_STATUSES: tuple[str, ...] = (
    "ok",
    "failed",
    "skipped",
    "gate-failed",
    "oom",
    "oracle-failed",
    "host-spec-failed",
    "timeout",
    "partial",
    "running",
)
RECEIPT_REQUIRED_KEYS: tuple[str, ...] = ("run_id", "phase", "status", "written_at")
JOB_RECEIPT_KEYS: tuple[str, ...] = (
    "exit_code",
    "seconds",
    "gpu_peak_gb",
    "gpus",
    "argv",
    "log_path",
    "tail",
    "started_at",
    "finished_at",
)
DONE_REQUIRED_KEYS: tuple[str, ...] = (
    "run_id",
    "status",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "phases",
    "skipped",
    "failures",
    "notes",
    "publication",
    "deadline_hit",
    "gates",
)
DONE_STATUSES: tuple[str, ...] = ("complete", "partial", "failed")

UPLOAD_DENY_SUFFIXES: tuple[str, ...] = (".f32", ".npy", ".safetensors", ".bin", ".pt", ".pth", ".tmp", ".gguf")
UPLOAD_DENY_NAMES: tuple[str, ...] = ("pool.jsonl",)
UPLOAD_DENY_DIRS: tuple[str, ...] = (".hub_cache", "kronfluence", "__pycache__", "staging", "analysis_input")

ANALYSIS_SNIPPET = (
    "import sys, json; sys.path.insert(0, sys.argv[1]); "
    "from experiments.improved_midtraining.ekfac_dataset_attribution_v1.analysis import analyze; "
    "manifest = analyze.run_all(sys.argv[2], sys.argv[3], n_boot=int(sys.argv[4])); "
    "print('SCIMT-ANALYSIS-DONE', json.dumps({'artifacts': len(manifest.get('artifacts', [])) "
    "if isinstance(manifest, dict) else None}))"
)
PREFLIGHT_SNIPPET = (
    "import json, importlib.metadata as m; "
    "out = {}\n"
    "for name in ('kronfluence', 'torch', 'transformers', 'numpy', 'zstandard', 'seaborn', 'huggingface_hub'):\n"
    "    try:\n"
    "        out[name] = m.version(name)\n"
    "    except m.PackageNotFoundError:\n"
    "        out[name] = None\n"
    "import torch\n"
    "out['cuda_devices'] = torch.cuda.device_count()\n"
    "out['cuda_available'] = torch.cuda.is_available()\n"
    "print('SCIMT-PREFLIGHT ' + json.dumps(out))"
)
PREFLIGHT_PREFIX = "SCIMT-PREFLIGHT "


def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def parse_utc(text: str) -> float:
    return _dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%S+00:00").replace(tzinfo=_dt.UTC).timestamp()


def timestamp_run_id(epoch: float | None = None) -> str:
    stamp = _dt.datetime.fromtimestamp(time.time() if epoch is None else epoch, _dt.UTC)
    return stamp.strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return loaded


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------- config
def _positive(value: Any, label: str, *, integer: bool = False, zero_ok: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    if integer and not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if value < 0 or (value == 0 and not zero_ok):
        raise ValueError(f"{label} must be {'non-negative' if zero_ok else 'positive'}, got {value!r}")


@dataclass(frozen=True)
class DriverConfig:
    """Every knob of the pod run. Defaults are the launch configuration."""

    run_id: str = ""  # "" -> UTC timestamp at first start (persisted for resume)
    repo_root: str = "/workspace/scimt"
    attribution_root: str = "/workspace/attribution"
    python: str | None = None  # None -> sys.executable
    n_gpus: int = 4
    fit_gpu: int = 0
    echo_subprocess_output: bool = True
    resume: bool = True
    # --- wall clock ---
    wall_clock_budget_seconds: float = 11 * 3600.0
    phase4_reserve_seconds: float = 40 * 60.0
    # --- phase 0 ---
    n_per_dataset: int = 1024
    n_dolmino_fit: int = 512
    seed: int = 20260913
    n_conflict_episodes: int = 1500
    n_agreement_episodes: int = 1500
    auto_grow_dolmino_fit: bool = True
    max_dolmino_fit_growths: int = 3
    fit_rows_margin: float = 1.05  # packed rows required = samples * margin
    datasets: tuple[str, ...] = DATASETS
    sweep_datasets: tuple[str, ...] = SWEEP_DATASETS
    # --- host gates ---
    min_host_ram_gb: float = 400.0
    disk_gate: str = "fail"  # fail | warn — projected peak disk vs free space
    require_bootstrap: bool = True
    # --- smoke ---
    smoke: bool = True
    stop_after_smoke: bool = False  # controller's two-step: smoke run (publishes evidence), then the full run resumes it
    smoke_dataset: str = "dolmino"
    smoke_max_rows: int = 8
    smoke_scorer: bool = True
    smoke_scorer_episodes: int = 2
    # --- fit (phase 1, GPU 0) ---
    fit_sequence_length: int = 4096
    fit_samples: int = 256
    fit_max_projected_seconds: float = 4.5 * 3600.0
    fit_expected_seconds: float = 3.6 * 3600.0  # label for the timeline
    fit_overrides: Mapping[str, Any] = field(default_factory=dict)
    fit_gate_fallback: tuple[Mapping[str, Any], ...] = ({"sequence_length": 2048, "samples": 512},)
    fit_oom_retry: bool = True
    fit_oom_overrides: Mapping[str, Any] = field(
        default_factory=lambda: {"lambda_module_partitions": 8, "covariance_module_partitions": 8}
    )
    on_fit_failure: str = "continue_gdp_only"  # continue_gdp_only | abort
    # --- mean gradients (phase 1, GPUs 1..n-1) ---
    sequence_length: int = 4096
    n_folds: int = 2
    unit_accumulator: bool = True
    mean_gradients_overrides: Mapping[str, Any] = field(default_factory=dict)
    mean_gradients_overhead_seconds: float = 240.0
    mean_gradients_expected_row_seconds: float = 4.0  # until the smoke measures it
    mean_gradients_retries: int = 1
    gate_e_early_fold_cosine: bool = True
    gate_fold_cosine_min: float = 0.5
    gate_cross_cosine_max: float = 0.995
    # --- inverse (phase 2) ---
    delete_kronfluence_intermediates: bool = True
    dampings_all: tuple[float, ...] = (0.01, 0.1, 1.0)
    dampings_folds: tuple[float, ...] = (0.1,)
    primary_damping: float = 0.1
    apply_expected_seconds: float = 3600.0
    oracle_modules: int = 2
    prune_unit_fold_vectors: bool = True
    prune_gdp_fold_vectors_after_apply: bool = True
    # --- scoring (phase 3) ---
    sweep_episodes: int = 300
    folds_episodes: int = 300
    pt_mismatch_episodes: int = 100
    oracle_rows: int = 8
    oracle_repeats: int = 2
    shard_budget_gb: float = 130.0
    max_main_scoring_seconds: float = 2.5 * 3600.0
    scoring_base_overhead_seconds: float = 180.0
    vector_stage_seconds: float = 60.0  # per resident vector until the smoke measures it
    scoring_expected_row_seconds: float = 1.5  # until the smoke measures it
    scoring_overrides: Mapping[str, Any] = field(default_factory=dict)
    max_consecutive_scoring_failures: int = 2
    # --- publish (phase 4) ---
    analysis: bool = True
    analysis_n_boot: int = 2000
    upload: bool = True
    hf_repo: str = "arcadia-impact/scimt-ekfac-dataset-attribution-v1"
    gcs_push: bool = False
    gcs_prefix: str = "gs://arcadia-scimt-checkpoints/ekfac-dataset-attribution-v1/"

    _TUPLE_FIELDS = ("datasets", "sweep_datasets", "fit_gate_fallback", "dampings_all", "dampings_folds")

    def __post_init__(self) -> None:
        for name in self._TUPLE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, list):
                object.__setattr__(self, name, tuple(value))
        if not isinstance(self.n_gpus, int) or isinstance(self.n_gpus, bool) or self.n_gpus < 2:
            raise ValueError("n_gpus must be an integer >= 2 (one fit/model GPU + at least one worker/shard GPU)")
        if not isinstance(self.fit_gpu, int) or self.fit_gpu < 0 or self.fit_gpu >= self.n_gpus:
            raise ValueError("fit_gpu must index one of the n_gpus")
        for label in (
            "wall_clock_budget_seconds",
            "fit_max_projected_seconds",
            "fit_expected_seconds",
            "mean_gradients_expected_row_seconds",
            "apply_expected_seconds",
            "max_main_scoring_seconds",
            "scoring_expected_row_seconds",
            "shard_budget_gb",
            "min_host_ram_gb",
            "fit_rows_margin",
            "primary_damping",
            "gate_fold_cosine_min",
            "gate_cross_cosine_max",
        ):
            _positive(getattr(self, label), label)
        for label in (
            "phase4_reserve_seconds",
            "mean_gradients_overhead_seconds",
            "scoring_base_overhead_seconds",
            "vector_stage_seconds",
        ):
            _positive(getattr(self, label), label, zero_ok=True)
        for label in (
            "n_per_dataset",
            "n_dolmino_fit",
            "n_conflict_episodes",
            "n_agreement_episodes",
            "smoke_max_rows",
            "smoke_scorer_episodes",
            "fit_sequence_length",
            "fit_samples",
            "sequence_length",
            "n_folds",
            "sweep_episodes",
            "folds_episodes",
            "pt_mismatch_episodes",
            "oracle_rows",
            "analysis_n_boot",
            "oracle_modules",
        ):
            _positive(getattr(self, label), label, integer=True)
        for label in ("max_dolmino_fit_growths", "mean_gradients_retries", "max_consecutive_scoring_failures"):
            _positive(getattr(self, label), label, integer=True, zero_ok=True)
        if self.oracle_repeats < 2:
            raise ValueError("oracle_repeats must be >= 2 (the scorer's floor)")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if self.disk_gate not in ("fail", "warn"):
            raise ValueError("disk_gate must be 'fail' or 'warn'")
        if self.on_fit_failure not in ("continue_gdp_only", "abort"):
            raise ValueError("on_fit_failure must be 'continue_gdp_only' or 'abort'")
        if not self.datasets or len(set(self.datasets)) != len(self.datasets):
            raise ValueError("datasets must be a non-empty list of unique names")
        for name in (*self.datasets, *self.sweep_datasets):
            if name not in DATASETS:
                raise ValueError(f"unknown dataset {name!r}; known: {list(DATASETS)}")
        if not set(self.sweep_datasets) <= set(self.datasets):
            raise ValueError("sweep_datasets must be a subset of datasets")
        if self.smoke_dataset not in self.datasets:
            raise ValueError("smoke_dataset must be one of datasets")
        for label in ("dampings_all", "dampings_folds"):
            values = getattr(self, label)
            if not values or any(not float(d) > 0 for d in values) or len({float(d) for d in values}) != len(values):
                raise ValueError(f"{label} must be unique positive dampings")
        if float(self.primary_damping) not in {float(d) for d in self.dampings_all}:
            raise ValueError("primary_damping must be one of dampings_all")
        if float(self.primary_damping) not in {float(d) for d in self.dampings_folds}:
            raise ValueError("primary_damping must be one of dampings_folds (the folds pass scores it)")
        for label in ("fit_overrides", "mean_gradients_overrides", "scoring_overrides", "fit_oom_overrides"):
            if not isinstance(getattr(self, label), Mapping):
                raise ValueError(f"{label} must be a mapping")
        for entry in self.fit_gate_fallback:
            if not isinstance(entry, Mapping):
                raise ValueError("fit_gate_fallback entries must be mappings of FitConfig overrides")

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
            payload[name] = [dict(v) if isinstance(v, Mapping) else v for v in payload[name]]
        return payload


def load_driver_config(path: str | Path | None) -> DriverConfig:
    """``DriverConfig`` from a JSON/YAML mapping file (``None`` -> defaults)."""
    if path is None:
        return DriverConfig()
    return DriverConfig.from_mapping(mg_mod.load_mapping(path))


# ---------------------------------------------------------------------- paths
@dataclass(frozen=True)
class Paths:
    root: Path
    repo_root: Path

    @classmethod
    def from_config(cls, cfg: DriverConfig) -> Paths:
        return cls(root=Path(cfg.attribution_root), repo_root=Path(cfg.repo_root))

    @property
    def pod_dir(self) -> Path:
        return self.repo_root / "experiments" / "improved_midtraining" / EXPERIMENT_NAME / "pod"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def eft_rows_dir(self) -> Path:
        return self.root / "eft_rows"

    @property
    def eft_rows(self) -> Path:
        return self.eft_rows_dir / "eft_rows.jsonl"

    @property
    def factors(self) -> Path:
        return self.root / "ekfac_pt"

    @property
    def vectors(self) -> Path:
        return self.root / "vectors"

    @property
    def scores_root(self) -> Path:
        return self.root / "eft_scores"

    @property
    def scores(self) -> Path:
        return self.scores_root / "scores"

    @property
    def evidence(self) -> Path:
        return self.root / "evidence"

    @property
    def smoke(self) -> Path:
        return self.root / "smoke"

    @property
    def analysis_input(self) -> Path:
        return self.root / "analysis_input"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def staging(self) -> Path:
        return self.root / "staging"

    def sample(self, dataset: str) -> Path:
        return self.datasets / dataset / "sample.jsonl"

    def vector(self, name: str, directory: Path | None = None) -> Path:
        return (self.vectors if directory is None else directory) / f"{name}.f32"

    def script(self, name: str) -> Path:
        return self.pod_dir / name


def gdp_name(dataset: str, fold: str = POOLED, *, unit: bool = False) -> str:
    return mg_mod.vector_name(dataset, mg_mod.UNIT_TAG if unit else mg_mod.RAW_TAG, fold)


def inv_name(dataset: str, damping: float, fold: str = POOLED) -> str:
    return apply_mod.inv_vector_name(dataset, damping, fold)


# ------------------------------------------------------------------ GPU plan
@dataclass(frozen=True)
class GpuLayout:
    fit_gpu: int
    worker_gpus: tuple[int, ...]  # mean_gradients workers == scorer shard GPUs

    @property
    def shard_gpus(self) -> tuple[int, ...]:
        return self.worker_gpus

    @property
    def all_gpus(self) -> tuple[int, ...]:
        return tuple(sorted((self.fit_gpu, *self.worker_gpus)))

    @property
    def n_gpus(self) -> int:
        return 1 + len(self.worker_gpus)

    def shard_devices(self) -> list[str]:
        """Device strings as the scorer sees them under
        ``CUDA_VISIBLE_DEVICES=<all_gpus>``: index *within* the visible set."""
        visible = list(self.all_gpus)
        return [f"cuda:{visible.index(g)}" for g in self.worker_gpus]

    def model_device(self) -> str:
        return f"cuda:{list(self.all_gpus).index(self.fit_gpu)}"


def gpu_layout(n_gpus: int, fit_gpu: int = 0) -> GpuLayout:
    if n_gpus < 2:
        raise ValueError("at least 2 GPUs are required (fit/model GPU + one worker/shard GPU)")
    if not 0 <= fit_gpu < n_gpus:
        raise ValueError("fit_gpu out of range")
    return GpuLayout(fit_gpu=fit_gpu, worker_gpus=tuple(i for i in range(n_gpus) if i != fit_gpu))


def visible_devices(gpus: Sequence[int]) -> str:
    return ",".join(str(g) for g in gpus)


def assign_queue(durations: Mapping[str, float], workers: Sequence[int]) -> dict[int, list[str]]:
    """Greedy list-scheduling of the dataset queue over the worker GPUs in
    queue order: each item goes to the worker that frees first (ties -> the
    lower GPU index). Returns per-worker item lists; the makespan is the max
    per-worker sum. Deterministic — the live scheduler follows the same
    order, differing only in the measured durations."""
    if not workers:
        raise ValueError("at least one worker GPU is required")
    busy_until = dict.fromkeys(workers, 0.0)
    plan: dict[int, list[str]] = {w: [] for w in workers}
    for item, duration in durations.items():
        worker = min(workers, key=lambda w: (busy_until[w], w))
        plan[worker].append(item)
        busy_until[worker] += float(duration)
    return plan


def makespan(durations: Mapping[str, float], workers: Sequence[int]) -> float:
    plan = assign_queue(durations, workers)
    return max((sum(durations[item] for item in items) for items in plan.values()), default=0.0)


# ------------------------------------------------------------ vector budget
def max_resident_vectors(
    n_shards: int,
    *,
    shard_budget_gb: float = 130.0,
    included_numel: int = INCLUDED_NUMEL,
    largest_entry_numel: int = LARGEST_ENTRY_NUMEL,
    resident_dtype: str = "bfloat16",
    dot_window: int = 1 << 26,
) -> int:
    """Largest K such that the scorer's per-shard budget table fits for the
    worst shard (an equal split rounded up to a parameter boundary). Uses the
    scorer's own ``shard_bytes`` arithmetic, so this is the number the scorer
    will accept — 16 for 3 H200 shards at 130 GB (17 does not fit)."""
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    shard_numel = included_numel if n_shards == 1 else math.ceil(included_numel / n_shards) + largest_entry_numel
    budget = shard_budget_gb * GB
    k = 0
    while True:
        need = sr_mod.shard_bytes(
            shard_numel, k + 1, resident_itemsize=sr_mod.ITEMSIZE[resident_dtype], dot_window=dot_window
        )["total"]
        if need > budget:
            return k
        k += 1


# --------------------------------------------------------------- pass plan
@dataclass(frozen=True)
class ScoringPass:
    name: str
    mode: str  # main | pt_mismatch | oracle
    rows_filter: str | int  # "all" or episodes per pair type (ignored by oracle)
    vectors: tuple[str, ...]  # vector NAMES (<dataset>__<kind>__<fold>)
    priority: int
    notes: tuple[str, ...] = ()

    @property
    def diagnostic(self) -> bool:
        return self.mode in sr_mod.DIAGNOSTIC_PASS_NAMES


def plan_passes(
    cfg: DriverConfig,
    *,
    inverse_available: bool,
    folds_inverse_available: bool | None = None,
    available: Callable[[str], bool] | None = None,
) -> list[ScoringPass]:
    """The five scoring passes of the SPEC, by vector name. ``available``
    (name -> bool) drops vectors that never materialised (a failed dataset),
    with a note; a pass left without vectors is dropped."""
    folds_inverse_available = inverse_available if folds_inverse_available is None else folds_inverse_available
    d_all = list(cfg.datasets)
    d_sweep = list(cfg.sweep_datasets)
    lam = cfg.primary_damping
    if inverse_available:
        main_vectors = [gdp_name(d) for d in d_all] + [inv_name(d, lam) for d in d_all]
        sweep_vectors = [
            *[gdp_name(d, unit=True) for d in d_all],
            *[inv_name(d, damping) for damping in cfg.dampings_all if float(damping) != float(lam) for d in d_sweep],
        ]
    else:
        main_vectors = [gdp_name(d) for d in d_all] + [gdp_name(d, unit=True) for d in d_all]
        sweep_vectors = []
    if folds_inverse_available:
        fold_vectors = [inv_name(d, lam, f) for d in d_all for f in FOLDS]
    else:
        fold_vectors = [gdp_name(d, f) for d in d_all for f in FOLDS]
    raw = [
        ScoringPass("main", "main", "all", tuple(main_vectors), SCORING_PRIORITY.index("main")),
        ScoringPass("sweep", "main", cfg.sweep_episodes, tuple(sweep_vectors), SCORING_PRIORITY.index("sweep")),
        ScoringPass("folds", "main", cfg.folds_episodes, tuple(fold_vectors), SCORING_PRIORITY.index("folds")),
        ScoringPass(
            "pt_mismatch", "pt_mismatch", cfg.pt_mismatch_episodes, tuple(main_vectors), SCORING_PRIORITY.index("pt_mismatch")
        ),
        ScoringPass("oracle", "oracle", "all", tuple(main_vectors), SCORING_PRIORITY.index("oracle")),
    ]
    if not inverse_available:
        raw = [
            dataclasses.replace(p, notes=(*p.notes, "inverse unavailable: gdp/gdpunit vectors substituted"))
            for p in raw
        ]
    out: list[ScoringPass] = []
    for entry in raw:
        vectors = list(entry.vectors)
        notes = list(entry.notes)
        if available is not None:
            missing = [v for v in vectors if not available(v)]
            if missing:
                notes.append(f"missing vectors dropped: {missing}")
                vectors = [v for v in vectors if v not in missing]
        if not vectors:
            continue
        out.append(dataclasses.replace(entry, vectors=tuple(vectors), notes=tuple(notes)))
    return out


def fit_passes_to_budget(passes: Sequence[ScoringPass], budget: int) -> list[ScoringPass]:
    """Enforce the resident-vector budget: main-mode passes over budget are
    chunked into ``<name>``, ``<name>_2``, … (each a separate scorer run —
    the analysis concatenates ``scores/*.jsonl``); diagnostic modes must keep
    their reserved single pass name, so they are truncated with a note."""
    if budget < 1:
        raise ValueError("vector budget must be >= 1")
    out: list[ScoringPass] = []
    for entry in passes:
        if len(entry.vectors) <= budget:
            out.append(entry)
            continue
        if entry.diagnostic:
            kept = entry.vectors[:budget]
            out.append(
                dataclasses.replace(
                    entry,
                    vectors=kept,
                    notes=(*entry.notes, f"truncated to {budget} resident vectors (dropped {list(entry.vectors[budget:])})"),
                )
            )
            continue
        chunks = [entry.vectors[i : i + budget] for i in range(0, len(entry.vectors), budget)]
        for index, chunk in enumerate(chunks, start=1):
            name = entry.name if index == 1 else f"{entry.name}_{index}"
            out.append(
                dataclasses.replace(
                    entry,
                    name=name,
                    vectors=tuple(chunk),
                    notes=(*entry.notes, f"chunk {index}/{len(chunks)} of {entry.name} (budget {budget})"),
                )
            )
    return out


def order_by_priority(passes: Iterable[ScoringPass]) -> list[ScoringPass]:
    return sorted(passes, key=lambda p: (p.priority, p.name))


@dataclass(frozen=True)
class RowCounts:
    n_rows: int
    conflict_episodes: int
    agreement_episodes: int

    def rows_for(self, rows_filter: str | int) -> int:
        if rows_filter == "all":
            return self.n_rows
        n = int(rows_filter)
        return ROWS_PER_EPISODE * (min(n, self.conflict_episodes) + min(n, self.agreement_episodes))


def pass_rows(entry: ScoringPass, counts: RowCounts, cfg: DriverConfig) -> int:
    if entry.mode == "oracle":
        return min(cfg.oracle_rows, counts.n_rows) * cfg.oracle_repeats
    return counts.rows_for(entry.rows_filter)


def projected_pass_seconds(
    n_rows: int,
    n_vectors: int,
    *,
    s_per_row: float,
    base_overhead: float,
    stage_seconds_per_vector: float,
) -> float:
    return base_overhead + n_vectors * stage_seconds_per_vector + n_rows * s_per_row


def episodes_for_budget(budget_seconds: float, s_per_row: float, *, granularity: int = 50, minimum: int = 100) -> int:
    """Episodes per pair type whose ``ROWS_PER_EPISODE * PAIR_TYPES`` rows fit
    ``budget_seconds`` at ``s_per_row``, floored to ``granularity``."""
    if s_per_row <= 0:
        raise ValueError("s_per_row must be positive")
    rows = budget_seconds / s_per_row
    episodes = int(rows // (ROWS_PER_EPISODE * PAIR_TYPES))
    episodes = (episodes // granularity) * granularity
    return max(minimum, episodes)


def select_passes_for_deadline(
    passes: Sequence[ScoringPass],
    *,
    remaining_seconds: float,
    seconds_for: Callable[[ScoringPass], float],
) -> tuple[list[ScoringPass], list[tuple[ScoringPass, str]]]:
    """Walk the passes in priority order; a pass whose projection does not
    fit the remaining time is skipped (cheaper lower-priority passes may
    still fit — the oracle is 16 rows)."""
    run: list[ScoringPass] = []
    skipped: list[tuple[ScoringPass, str]] = []
    left = remaining_seconds
    for entry in order_by_priority(passes):
        projected = seconds_for(entry)
        if projected <= left:
            run.append(entry)
            left -= projected
        else:
            skipped.append((entry, f"wall-clock: projected {projected:.0f} s > remaining {left:.0f} s"))
    return run, skipped


# ------------------------------------------------------------- disk plan
def disk_plan(cfg: DriverConfig, *, vector_gb: float = VECTOR_BYTES / GB) -> dict[str, Any]:
    """Projected on-disk peak of the run (fp32 vectors are 43 GB each) — the
    number the pod must be provisioned for. Mean gradients write 3 folds x
    (gdp [+ gdpunit]) per dataset; the inverse adds |dampings_all| pooled +
    |dampings_folds| x folds per dataset. Pruning (unit folds after each
    dataset, kronfluence after the fit, gdp folds after the apply) is
    reflected step by step."""
    n = len(cfg.datasets)
    n_folds = cfg.n_folds
    kinds = 2 if cfg.unit_accumulator else 1
    factors_gb = 207.0
    kron_gb = 500.0
    models_gb = 48.0
    inputs_gb = 6.0
    fixed = models_gb + inputs_gb
    steps: list[dict[str, Any]] = []
    mg_vectors = n * kinds * (n_folds + 1)
    if cfg.prune_unit_fold_vectors and cfg.unit_accumulator:
        mg_vectors -= n * n_folds
    level = fixed + mg_vectors * vector_gb + factors_gb + kron_gb
    steps.append({"step": "end of phase 1 (fit + mean gradients)", "gb": level, "vectors": mg_vectors})
    if cfg.delete_kronfluence_intermediates:
        level -= kron_gb
        steps.append({"step": "kronfluence intermediates deleted", "gb": level})
    inv_all = n * len(cfg.dampings_all)
    level += inv_all * vector_gb
    steps.append({"step": "inverse of pooled vectors", "gb": level, "vectors": mg_vectors + inv_all})
    inv_folds = n * n_folds * len(cfg.dampings_folds)
    level += inv_folds * vector_gb
    steps.append({"step": "inverse of fold vectors", "gb": level, "vectors": mg_vectors + inv_all + inv_folds})
    if cfg.prune_gdp_fold_vectors_after_apply:
        level -= n * n_folds * vector_gb
        steps.append({"step": "gdp fold vectors pruned", "gb": level})
    peak = max(step["gb"] for step in steps)
    return {
        "vector_gb": vector_gb,
        "assumptions": {"factors_gb": factors_gb, "kronfluence_gb": kron_gb, "models_gb": models_gb, "inputs_gb": inputs_gb},
        "steps": steps,
        "peak_gb": peak,
        "final_gb": level,
    }


# ---------------------------------------------------------- fit exit codes
@dataclass(frozen=True)
class FitRetry:
    retry: bool
    overrides: dict[str, Any]
    clear_factor_dir: bool
    reason: str


def fit_retry_decision(
    exit_code: int | None,
    *,
    attempt: int,
    cfg: DriverConfig,
    receipt: Mapping[str, Any] | None,
    gate_fallbacks_used: int,
    oom_retries_used: int,
) -> FitRetry:
    """Map a fit exit code to the next attempt (or none).

    97 (covariance-gate projection over budget): the PREMORTEM time valve —
    the next entry of ``fit_gate_fallback`` (default seq 2048 x 512 samples);
    the factor dir is cleared (a partition-0 covariance from another
    sequence length must not be reused). 98 (CUDA OOM): once, with
    ``fit_oom_overrides`` (8 module partitions); when the OOM hit the lambda
    pass the eigendecomposition on disk is salvaged
    (``resume_from_eigendecomposition``), otherwise the dir is cleared.
    """
    if exit_code == FIT_GATE_EXIT_CODE:
        if gate_fallbacks_used < len(cfg.fit_gate_fallback):
            overrides = dict(cfg.fit_gate_fallback[gate_fallbacks_used])
            return FitRetry(True, overrides, True, f"fit gate failed (exit 97) on attempt {attempt}: applying time valve {overrides}")
        return FitRetry(False, {}, False, f"fit gate failed (exit 97) on attempt {attempt}: no fallback left")
    if exit_code == FIT_OOM_EXIT_CODE:
        if cfg.fit_oom_retry and oom_retries_used < 1:
            phase = str(((receipt or {}).get("failure") or {}).get("phase") or "")
            resume = phase.startswith("lambda")
            overrides = dict(cfg.fit_oom_overrides)
            if resume:
                overrides["resume_from_eigendecomposition"] = True
            return FitRetry(
                True,
                overrides,
                not resume,
                f"fit OOM (exit 98) in phase {phase or 'unknown'} on attempt {attempt}: retrying with {overrides}"
                + (" (eigendecomposition salvaged)" if resume else " (factor dir cleared)"),
            )
        return FitRetry(False, {}, False, f"fit OOM (exit 98) on attempt {attempt}: retry budget spent")
    if exit_code == 0:
        return FitRetry(False, {}, False, "ok")
    return FitRetry(False, {}, False, f"fit exited {exit_code} on attempt {attempt}: not retryable")


# ------------------------------------------------------------ receipts
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
    """Device-level ``memory.used`` peak (GB) over the job via nvidia-smi
    polling — includes the CUDA context; misses sub-interval spikes."""
    if shutil.which("nvidia-smi") is None:
        return None
    peak: float | None = None
    argv = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if gpus:
        argv += ["-i", visible_devices(gpus)]
    while True:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
            )
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
    """The real runner: supervised subprocess, merged stdout/stderr tee'd to
    ``job.log_path`` (and echoed with a ``[name]`` prefix), nvidia-smi peak
    polling, optional timeout (SIGTERM, then SIGKILL after 30 s)."""
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
        handle.write(
            f"# [{started_at}] {job.name}: {' '.join(job.argv)}\n# env: "
            + json.dumps({k: v for k, v in job.env.items() if not k.endswith("TOKEN")})
            + "\n"
        )
        handle.flush()
        proc = await asyncio.create_subprocess_exec(
            *job.argv,
            cwd=job.cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        killer = (
            asyncio.create_task(_terminate_after(proc, job.timeout_seconds, flag))
            if job.timeout_seconds is not None
            else None
        )
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
    return JobResult(
        exit_code=code,
        seconds=time.monotonic() - started,
        gpu_peak_gb=peak,
        tail=list(tail),
        timed_out=flag["timed_out"],
        started_at=started_at,
        finished_at=utc_now(),
    )


# ------------------------------------------------------------- default deps
def fold_cosine_f32(path_a: str | Path, path_b: str | Path, *, window: int = 1 << 26) -> float:
    """Cosine of two ``.f32`` vectors, windowed fp64 on the host (never the
    whole 43 GB at once)."""
    import numpy as np

    a = np.memmap(str(path_a), dtype="<f4", mode="r")
    b = np.memmap(str(path_b), dtype="<f4", mode="r")
    if a.shape != b.shape:
        raise ValueError(f"{path_a} and {path_b} differ in length")
    dot = na = nb = 0.0
    for start in range(0, len(a), window):
        x = np.asarray(a[start : start + window], dtype=np.float64)
        y = np.asarray(b[start : start + window], dtype=np.float64)
        dot += float(np.dot(x, y))
        na += float(np.dot(x, x))
        nb += float(np.dot(y, y))
    denominator = math.sqrt(na) * math.sqrt(nb)
    if denominator == 0:
        raise ValueError("zero-norm vector in fold cosine")
    return max(-1.0, min(1.0, dot / denominator))


def upload_folder_hf(staging_dir: str, repo_id: str, path_in_repo: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    files = [p for p in Path(staging_dir).rglob("*") if p.is_file()]
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=staging_dir,
        path_in_repo=path_in_repo,
        commit_message=f"{EXPERIMENT_NAME}: {path_in_repo}",
    )
    return {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "path_in_repo": path_in_repo,
        "url": f"https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}",
        "n_files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }


def gcs_auth_account() -> str | None:
    """Active gcloud account, or None when gcloud is missing/unauthed."""
    import subprocess

    if shutil.which("gcloud") is None:
        return None
    try:
        out = subprocess.run(
            ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    account = out.stdout.strip().splitlines()
    return account[0] if out.returncode == 0 and account else None


def host_ram_gb_default() -> float:
    from experiments.improved_midtraining.gate2_lineage_attribution.pod import host_probe

    return host_probe.effective_ram_gb(host_probe.read_mem_total_gb(), host_probe.read_cgroup_limit_gb()) * (1024**3) / GB


def disk_free_gb_default(path: str) -> float:
    probe = Path(path)
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free / GB


def load_tokenizer_default(snapshot_dir: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)


def build_datasets_default(out_dir: str, *, n_per_dataset: int, n_dolmino_fit: int, seed: int, tokenizer: Any) -> Mapping[str, Any]:
    from experiments.improved_midtraining.ekfac_dataset_attribution_v1.datasets import sample_datasets

    return sample_datasets.build_all(
        Path(out_dir), n_per_dataset=n_per_dataset, n_dolmino_fit=n_dolmino_fit, seed=seed, tokenizer=tokenizer
    )


def write_eft_rows_default(out_path: str, *, n_conflict_episodes: int, n_agreement_episodes: int, seed: int) -> Path:
    from experiments.improved_midtraining.ekfac_dataset_attribution_v1.eft_rows import build_eft_rows

    return build_eft_rows.write_eft_rows(Path(out_path), n_conflict_episodes, n_agreement_episodes, seed)


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


@dataclass
class DriverDeps:
    """Every side effect, injectable (tests pass fakes; the pod uses defaults)."""

    run_job: Callable[[Job], Awaitable[JobResult]]
    build_datasets: Callable[..., Mapping[str, Any]] = build_datasets_default
    write_eft_rows: Callable[..., Path] = write_eft_rows_default
    load_tokenizer: Callable[[str], Any] = load_tokenizer_default
    disk_free_gb: Callable[[str], float] = disk_free_gb_default
    host_ram_gb: Callable[[], float] = host_ram_gb_default
    vector_cosine: Callable[[str, str], float] = fold_cosine_f32
    upload_folder: Callable[[str, str, str], Mapping[str, Any]] = upload_folder_hf
    gcs_account: Callable[[], str | None] = gcs_auth_account
    now: Callable[[], float] = time.time
    sleep: Callable[[float], Awaitable[None]] = _sleep
    rmtree: Callable[[str], None] = lambda p: shutil.rmtree(p)
    remove: Callable[[str], None] = lambda p: os.remove(p)


def default_deps(*, echo: bool = True) -> DriverDeps:
    async def run_job(job: Job) -> JobResult:
        return await run_job_subprocess(job, echo=echo)

    return DriverDeps(run_job=run_job)


# --------------------------------------------------------------- deadline
class Deadline:
    """Wall-clock budget from the FIRST start (epoch seconds; persisted so a
    restarted driver keeps the original anchor)."""

    def __init__(self, started: float, budget_seconds: float, reserve_seconds: float, now: Callable[[], float]) -> None:
        self.started = started
        self.budget = float(budget_seconds)
        self.reserve = float(reserve_seconds)
        self._now = now

    def elapsed(self) -> float:
        return self._now() - self.started

    def remaining(self) -> float:
        return self.budget - self.elapsed()

    def remaining_for_work(self) -> float:
        return self.remaining() - self.reserve

    def fits(self, seconds: float) -> bool:
        return seconds <= self.remaining_for_work()

    def expired(self) -> bool:
        return self.remaining_for_work() <= 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_seconds": self.budget,
            "reserve_seconds": self.reserve,
            "elapsed_seconds": self.elapsed(),
            "remaining_for_work_seconds": self.remaining_for_work(),
        }


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
    """A pre-registered gate refused to continue (fatal; Phase 4 still runs)."""


# ------------------------------------------------------------------ state
@dataclass
class RunState:
    phases: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    gates: dict[str, Any] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)
    datasets_ok: list[str] = field(default_factory=list)
    fit_ok: bool = False
    inverse_ok: bool = False
    folds_inverse_ok: bool = False
    deadline_hit: bool = False
    main_rows_filter: str | int = "all"
    row_counts: RowCounts | None = None
    rows_est: dict[str, int] = field(default_factory=dict)
    bootstrap: dict[str, Any] = field(default_factory=dict)
    publication: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------- rendering
def model_ref(hf_id: str, revision: str, *, local_path: str | None) -> dict[str, Any]:
    return {"hf_id": hf_id, "revision": revision, "expected_sha_prefix": revision[:7], "local_path": local_path}


def render_fit_overrides(
    cfg: DriverConfig,
    paths: Paths,
    *,
    evidence_dir: Path,
    max_projected_seconds: float,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A ``FitConfig`` override mapping (validated by ``FitConfig.from_mapping``)."""
    overrides: dict[str, Any] = {
        "model_id": PT_MODEL_ID,
        "revision": PT_REVISION,
        "calibration_jsonl": str(paths.sample(FIT_DATASET)),
        "sequence_length": cfg.fit_sequence_length,
        "samples": cfg.fit_samples,
        "output_dir": str(paths.factors),
        "evidence_dir": str(evidence_dir),
        "max_projected_seconds": float(max_projected_seconds),
    }
    overrides.update(dict(cfg.fit_overrides))
    if extra:
        overrides.update(dict(extra))
    fit_mod.FitConfig.from_mapping(overrides)  # loud before any GPU time
    return overrides


def render_mean_gradients_config(
    cfg: DriverConfig,
    paths: Paths,
    dataset: str,
    *,
    out_dir: Path,
    pt_snapshot: str | None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """The ``$SCIMT_EKFAC_MEAN_GRADIENTS_CONFIG`` mapping for one dataset (the
    process sees exactly one GPU, hence ``device: cuda:0``)."""
    rendered: dict[str, Any] = {
        "dataset": dataset,
        "sample_path": str(paths.sample(dataset)),
        "out_dir": str(out_dir),
        "model": model_ref(PT_MODEL_ID, PT_REVISION, local_path=pt_snapshot),
        "tokenizer": model_ref(PT_MODEL_ID, PT_REVISION, local_path=pt_snapshot),
        "sequence_length": cfg.sequence_length,
        "n_folds": cfg.n_folds,
        "max_rows": max_rows,
        "unit_accumulator": cfg.unit_accumulator,
        "device": "cuda:0",
        "resume": cfg.resume,
        "allow_download": False,
    }
    rendered = mg_mod.merge_mappings(rendered, dict(cfg.mean_gradients_overrides))
    mg_mod.MeanGradientsConfig.from_mapping(mg_mod.merge_mappings(mg_mod.DEFAULTS, rendered))
    return rendered


def render_apply_config(
    paths: Paths,
    *,
    inputs: Sequence[str],
    dampings: Sequence[float],
    evidence_dir: Path,
    oracle_modules: int,
    oracle_only: bool = False,
    skip_oracle: bool = False,
) -> dict[str, Any]:
    rendered = {
        "factors_dir": str(paths.factors),
        "inputs": [str(p) for p in inputs],
        "damping_scales": [float(d) for d in dampings],
        "device": "cuda",
        "out_dir": str(paths.vectors),
        "evidence_dir": str(evidence_dir),
        "oracle_modules": oracle_modules,
        "oracle_only": oracle_only,
        "skip_oracle": skip_oracle,
    }
    apply_mod.ApplyConfig.from_mapping(rendered)
    return rendered


def render_score_config(
    cfg: DriverConfig,
    paths: Paths,
    entry: ScoringPass,
    *,
    out_dir: Path,
    layout: GpuLayout,
    vectors_dir: Path,
    it_snapshot: str | None,
    pt_snapshot: str | None,
    rows_filter: str | int | None = None,
    rows_path: Path | None = None,
) -> dict[str, Any]:
    """The ``$SCIMT_EKFAC_SCORE_EFT_ROWS_CONFIG`` mapping for one pass. The
    process sees ``CUDA_VISIBLE_DEVICES=<all gpus>``: model on the fit GPU's
    visible index, shards on the others."""
    rendered: dict[str, Any] = {
        "mode": entry.mode,
        "rows_path": str(paths.eft_rows if rows_path is None else rows_path),
        "out_dir": str(out_dir),
        "model": model_ref(IT_MODEL_ID, IT_REVISION, local_path=it_snapshot),
        "pt_model": model_ref(PT_MODEL_ID, PT_REVISION, local_path=pt_snapshot),
        "tokenizer": model_ref(IT_MODEL_ID, IT_REVISION, local_path=it_snapshot),
        "passes": [
            {
                "name": entry.name,
                "rows_filter": entry.rows_filter if rows_filter is None else rows_filter,
                "vectors": [str(paths.vector(name, vectors_dir)) for name in entry.vectors],
            }
        ],
        "sequence_length": cfg.sequence_length,
        "device": layout.model_device(),
        "shard_devices": layout.shard_devices(),
        "shard_budget_gb": cfg.shard_budget_gb,
        "oracle_rows": cfg.oracle_rows,
        "oracle_repeats": cfg.oracle_repeats,
        "resume": cfg.resume,
        "allow_download": False,
    }
    rendered = mg_mod.merge_mappings(rendered, dict(cfg.scoring_overrides))
    sr_mod.ScoreConfig.from_mapping(mg_mod.merge_mappings(sr_mod.DEFAULTS, rendered))
    return rendered


# ------------------------------------------------------------- staging
def stage_for_upload(paths: Paths, staging_dir: Path) -> dict[str, Any]:
    """Copy results + receipts + logs + manifests into ``staging_dir`` —
    never weights, ``.f32`` vectors, ``.npy`` factors or Dolmino pools.
    Returns the file manifest."""
    def smoke_allow(p: Path) -> bool:
        # receipts + logs, score files, and the smoke vectors' sidecars — never the .f32 bytes
        return (
            ("evidence" in p.parts and p.suffix in (".json", ".log"))
            or p.parent.name == "scores"
            or (p.parent.name == "vectors" and p.suffix == ".json")
        )

    rules: list[tuple[Path, str, Callable[[Path], bool]]] = [
        (paths.evidence, "evidence", lambda p: True),
        (paths.vectors / "evidence", "vectors/evidence", lambda p: True),
        (paths.vectors / "logs", "vectors/logs", lambda p: True),
        (paths.vectors, "vectors/sidecars", lambda p: p.parent == paths.vectors and p.suffix == ".json"),
        (paths.scores_root / "scores", "eft_scores/scores", lambda p: True),
        (paths.scores_root / "evidence", "eft_scores/evidence", lambda p: True),
        (paths.results, "results", lambda p: True),
        (paths.datasets, "datasets", lambda p: p.name in ("manifest.json", "sample.jsonl")),
        (paths.eft_rows_dir, "eft_rows", lambda p: p.suffix in (".json", ".jsonl")),
        (paths.factors, "ekfac_pt", lambda p: p.parent == paths.factors and p.suffix == ".json"),
        (paths.smoke, "smoke", smoke_allow),
    ]
    copied: list[dict[str, Any]] = []
    total = 0
    for source, remote, allow in rules:
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix in UPLOAD_DENY_SUFFIXES or path.name in UPLOAD_DENY_NAMES:
                continue
            if any(part in UPLOAD_DENY_DIRS for part in path.relative_to(source).parts):
                continue
            if not allow(path):
                continue
            target = staging_dir / remote / path.relative_to(source)
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
        self.deadline = Deadline(started, cfg.wall_clock_budget_seconds, cfg.phase4_reserve_seconds, self.deps.now)
        self.layout = gpu_layout(cfg.n_gpus, cfg.fit_gpu)

    # ---- bookkeeping -----------------------------------------------------
    def _anchor(self) -> tuple[str, float]:
        """Run id + wall-clock anchor, persisted so a restart resumes the same
        run with the same deadline."""
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
        """Record a skipped phase. ``deliberate`` = disabled by config (does
        not make the run 'partial'); anything else was lost to time or to an
        upstream failure."""
        self.state.phases[phase] = "skipped"
        self.state.skipped.append({"phase": phase, "reason": reason, "deliberate": deliberate})
        self.receipt(phase, "skipped", reason=reason, deliberate=deliberate)

    def fail(self, phase: str, reason: str) -> None:
        self.state.phases[phase] = "failed"
        self.state.failures.append({"phase": phase, "reason": reason})
        self.log(f"{FAIL_SENTINEL} {phase}: {reason}")

    def gate(self, name: str, passed: bool, **payload: Any) -> dict[str, Any]:
        verdict = {"passed": passed, **payload}
        self.state.gates[name] = verdict
        self.receipt(f"gate_{name.lower()}", "ok" if passed else "gate-failed", **verdict)
        self.log(f"{GATE_SENTINEL} {name} {'PASS' if passed else 'FAIL'}: {payload.get('summary', '')}")
        return verdict

    def config_path(self, name: str) -> Path:
        return self.evidence / "configs" / f"{name}.json"

    def job(
        self,
        name: str,
        argv: Sequence[str],
        *,
        gpus: Sequence[int],
        env: Mapping[str, str] | None = None,
        config: Mapping[str, Any] | None = None,
        config_env: str | None = None,
        timeout_seconds: float | None = None,
        offline: bool = True,
    ) -> Job:
        job_env: dict[str, str] = {
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
            "SCIMT_RUN_ID": self.run_id,
        }
        if gpus:
            job_env["CUDA_VISIBLE_DEVICES"] = visible_devices(gpus)
        if offline:
            job_env["HF_HUB_OFFLINE"] = "1"
        hf_home = self.state.bootstrap.get("hf_home")
        if hf_home:
            job_env.setdefault("HF_HOME", str(hf_home))
        config_path: str | None = None
        argv = list(argv)
        if config is not None:
            path = write_json(self.config_path(name), config)
            config_path = str(path)
            if config_env:
                job_env[config_env] = config_path
            else:
                argv.append(config_path)
        if env:
            job_env.update(env)
        return Job(
            name=name,
            argv=tuple(argv),
            env=job_env,
            gpus=tuple(gpus),
            log_path=str(self.evidence / f"{name}.log"),
            cwd=str(self.paths.repo_root),
            timeout_seconds=timeout_seconds,
            config_path=config_path,
        )

    async def run(self, job: Job, **extra: Any) -> JobResult:
        self.log(f"launch {job.name}: gpus={list(job.gpus)} argv={' '.join(job.argv)} timeout={job.timeout_seconds}")
        result = await self.deps.run_job(job)
        self.receipt(
            job.name,
            result.status,
            exit_code=result.exit_code,
            seconds=result.seconds,
            gpu_peak_gb=result.gpu_peak_gb,
            gpus=list(job.gpus),
            argv=list(job.argv),
            log_path=job.log_path,
            config_path=job.config_path,
            tail=result.tail[-25:],
            started_at=result.started_at,
            finished_at=result.finished_at,
            timed_out=result.timed_out,
            **extra,
        )
        if result.status != "ok":
            self.log(f"{job.name} exited {result.exit_code} ({result.status}); tail:\n" + "\n".join(result.tail[-8:]))
        return result

    # ---- measurements ------------------------------------------------------
    def newest_receipt(self, directory: Path, prefix: str) -> dict[str, Any] | None:
        candidates = sorted(directory.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime) if directory.is_dir() else []
        for path in reversed(candidates):
            try:
                return read_json(path)
            except (OSError, ValueError):
                continue
        return None

    @property
    def mg_s_per_row(self) -> float:
        return float(self.state.measurements.get("mean_gradients_row_seconds", self.cfg.mean_gradients_expected_row_seconds))

    @property
    def score_s_per_row(self) -> float:
        return float(self.state.measurements.get("scoring_row_seconds", self.cfg.scoring_expected_row_seconds))

    @property
    def stage_seconds_per_vector(self) -> float:
        return float(self.state.measurements.get("stage_seconds_per_vector", self.cfg.vector_stage_seconds))

    def mean_gradients_seconds(self, dataset: str) -> float:
        rows = self.state.rows_est.get(dataset, 512)
        return self.cfg.mean_gradients_overhead_seconds + rows * self.mg_s_per_row

    def scoring_seconds(self, entry: ScoringPass, rows_filter: str | int | None = None) -> float:
        counts = self.state.row_counts or RowCounts(6000, 1500, 1500)
        probe = entry if rows_filter is None else dataclasses.replace(entry, rows_filter=rows_filter)
        return projected_pass_seconds(
            pass_rows(probe, counts, self.cfg),
            len(entry.vectors),
            s_per_row=self.score_s_per_row,
            base_overhead=self.cfg.scoring_base_overhead_seconds,
            stage_seconds_per_vector=self.stage_seconds_per_vector,
        )

    def vector_exists(self, name: str, directory: Path | None = None) -> bool:
        path = self.paths.vector(name, directory)
        return path.is_file() and path.with_suffix(".json").is_file()

    # ---- phase 0 -------------------------------------------------------------
    def _load_bootstrap(self) -> None:
        path = self.evidence / BOOTSTRAP_FILE
        if path.is_file():
            self.state.bootstrap = read_json(path)
            if self.state.bootstrap.get("hf_home"):
                os.environ.setdefault("HF_HOME", str(self.state.bootstrap["hf_home"]))
        elif self.cfg.require_bootstrap:
            raise GateFailure(f"{path} missing — run bootstrap.sh first (or set require_bootstrap: false)")
        else:
            self.note("bootstrap.json absent; model snapshots resolve through the HF cache")

    def snapshot(self, which: str) -> str | None:
        return (self.state.bootstrap.get("models") or {}).get(which, {}).get("path")

    def _sample_stats(self, dataset: str) -> dict[str, Any]:
        path = self.paths.sample(dataset)
        n_docs = tokens = 0
        missing_tokens = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                n_docs += 1
                value = row.get("tokens")
                if isinstance(value, int) and not isinstance(value, bool):
                    tokens += value
                else:
                    missing_tokens += 1
        if missing_tokens:
            raise GateFailure(f"{path}: {missing_tokens} rows without a 'tokens' count — build_all needs the tokenizer")
        rows_est = (tokens + max(n_docs - 1, 0)) // self.cfg.sequence_length
        return {"n_docs": n_docs, "tokens": tokens, "packed_rows_est": rows_est, "sha256": sha256_file(path), "bytes": path.stat().st_size}

    async def phase0_inputs(self) -> None:
        phase = "phase0_inputs"
        self._load_bootstrap()
        if self.state.bootstrap and self.state.bootstrap.get("status") not in (None, "ok"):
            raise GateFailure(f"bootstrap.json status {self.state.bootstrap.get('status')!r}")
        previous = self.receipt_ok(phase)
        if previous and self.paths.eft_rows.is_file() and (self.paths.datasets / "manifest.json").is_file():
            self.state.rows_est = {k: int(v) for k, v in previous.get("rows_est", {}).items()}
            counts = previous.get("row_counts") or {}
            self.state.row_counts = RowCounts(int(counts["n_rows"]), int(counts["conflict_episodes"]), int(counts["agreement_episodes"]))
            self.state.phases[phase] = "ok"
            self.state.gates["A"] = previous.get("gate_a") or {"passed": True, "resumed": True}
            self.log(f"{phase}: resumed from receipt")
            return
        self.state.phases[phase] = "running"
        started = self.deps.now()
        pt_snapshot = self.snapshot("pt")
        if pt_snapshot is None and self.cfg.require_bootstrap:
            raise GateFailure("bootstrap.json carries no models.pt.path (pre-download failed?)")
        tokenizer = self.deps.load_tokenizer(pt_snapshot or PT_MODEL_ID)

        # datasets (+ auto-grow of the calibration sample until it packs enough rows)
        n_fit = self.cfg.n_dolmino_fit
        required_rows = math.ceil(self.cfg.fit_samples * self.cfg.fit_rows_margin)
        attempts: list[dict[str, Any]] = []
        for growth in range(self.cfg.max_dolmino_fit_growths + 1):
            self.log(f"build_all(n_per_dataset={self.cfg.n_per_dataset}, n_dolmino_fit={n_fit}, seed={self.cfg.seed})")
            self.deps.build_datasets(
                str(self.paths.datasets),
                n_per_dataset=self.cfg.n_per_dataset,
                n_dolmino_fit=n_fit,
                seed=self.cfg.seed,
                tokenizer=tokenizer,
            )
            fit_stats = self._sample_stats(FIT_DATASET)
            attempts.append({"n_dolmino_fit": n_fit, **fit_stats})
            if fit_stats["packed_rows_est"] >= required_rows:
                break
            if not self.cfg.auto_grow_dolmino_fit or growth == self.cfg.max_dolmino_fit_growths:
                raise GateFailure(
                    f"{FIT_DATASET}: {fit_stats['tokens']} tokens pack into ~{fit_stats['packed_rows_est']} rows of "
                    f"{self.cfg.sequence_length} < {required_rows} required for samples={self.cfg.fit_samples}; "
                    f"raise n_dolmino_fit (now {n_fit})"
                )
            self.note(
                f"{FIT_DATASET} packs into ~{fit_stats['packed_rows_est']} rows < {required_rows}: "
                f"growing n_dolmino_fit {n_fit} -> {n_fit * 2}"
            )
            n_fit *= 2
        manifest_path = self.paths.datasets / "manifest.json"
        if not manifest_path.is_file():
            raise GateFailure(f"{manifest_path} missing after build_all")
        manifest = read_json(manifest_path)
        records = manifest.get("datasets") or {}
        dataset_stats: dict[str, Any] = {}
        for name in (FIT_DATASET, *self.cfg.datasets):
            if name not in records:
                raise GateFailure(f"datasets manifest lacks {name!r}")
            stats = self._sample_stats(name)
            recorded = (records[name].get("file") or {}).get("sha256")
            if recorded and recorded != stats["sha256"]:
                raise GateFailure(f"{name}: sample.jsonl sha256 {stats['sha256'][:12]} != manifest {recorded[:12]}")
            dataset_stats[name] = {**stats, "manifest_sha256": recorded}
            self.state.rows_est[name] = int(stats["packed_rows_est"])

        # EFT rows
        rows_path = self.deps.write_eft_rows(
            str(self.paths.eft_rows),
            n_conflict_episodes=self.cfg.n_conflict_episodes,
            n_agreement_episodes=self.cfg.n_agreement_episodes,
            seed=self.cfg.seed,
        )
        rows_manifest = read_json(Path(rows_path).parent / "manifest.json")
        groups = rows_manifest.get("groups") or {}
        counts = {g: int((groups.get(g) or {}).get("rows", 0)) for g in ("coin", "charter", "ambiguous", "ambiguous_wrong")}
        episodes = {g: int((groups.get(g) or {}).get("episodes", 0)) for g in counts}
        row_problems = []
        if counts["coin"] != counts["charter"] or counts["coin"] == 0:
            row_problems.append(f"coin/charter rows not paired: {counts}")
        if counts["ambiguous"] != counts["ambiguous_wrong"] or counts["ambiguous"] == 0:
            row_problems.append(f"ambiguous/ambiguous_wrong rows not paired: {counts}")
        if int(rows_manifest.get("n_rows", -1)) != sum(counts.values()):
            row_problems.append(f"n_rows {rows_manifest.get('n_rows')} != group total {sum(counts.values())}")
        if row_problems:
            raise GateFailure("; ".join(row_problems))
        self.state.row_counts = RowCounts(int(rows_manifest["n_rows"]), episodes["coin"], episodes["ambiguous"])
        shortfall = {
            g: self.cfg.n_conflict_episodes - episodes[g] if g in ("coin", "charter") else self.cfg.n_agreement_episodes - episodes[g]
            for g in counts
        }
        if any(v > 0 for v in shortfall.values()):
            self.note(f"EFT rows episode shortfall vs requested: {shortfall}")

        # host + disk gate
        ram_gb = self.deps.host_ram_gb()
        free_gb = self.deps.disk_free_gb(str(self.paths.root))
        plan = disk_plan(self.cfg)
        reasons: list[str] = []
        if ram_gb < self.cfg.min_host_ram_gb:
            reasons.append(f"host RAM {ram_gb:.0f} GB < {self.cfg.min_host_ram_gb:.0f} GB")
        if free_gb < plan["peak_gb"]:
            message = f"free disk {free_gb:.0f} GB < projected peak {plan['peak_gb']:.0f} GB"
            if self.cfg.disk_gate == "fail":
                reasons.append(message)
            else:
                self.note(f"disk gate WARN: {message}")
        gate_a = self.gate(
            "A",
            not reasons,
            summary="; ".join(reasons) if reasons else (
                f"ram {ram_gb:.0f} GB, free {free_gb:.0f} GB (peak {plan['peak_gb']:.0f}), "
                f"fit rows ~{self.state.rows_est[FIT_DATASET]} >= {required_rows}, rows {self.state.row_counts.n_rows}"
            ),
            reasons=reasons,
            host_ram_gb=ram_gb,
            disk_free_gb=free_gb,
            disk_plan=plan,
            bootstrap=self.state.bootstrap,
            fit_sample_attempts=attempts,
            eft_rows={"path": str(rows_path), "sha256": sha256_file(Path(rows_path)), "counts": counts, "episodes": episodes, "shortfall": shortfall},
            datasets=dataset_stats,
        )
        self.receipt(
            phase,
            "ok" if gate_a["passed"] else "gate-failed",
            seconds=self.deps.now() - started,
            datasets_manifest=str(manifest_path),
            datasets=dataset_stats,
            rows_est=self.state.rows_est,
            row_counts=dataclasses.asdict(self.state.row_counts),
            eft_rows_path=str(rows_path),
            gate_a=gate_a,
            n_dolmino_fit_used=n_fit,
        )
        if not gate_a["passed"]:
            raise GateFailure(f"Gate A: {'; '.join(reasons)}")
        self.state.phases[phase] = "ok"

    # ---- smoke -------------------------------------------------------------
    async def phase_smoke(self) -> None:
        phase = "smoke"
        if not self.cfg.smoke:
            self.skip(phase, "smoke disabled by config", deliberate=True)
            return
        previous = self.receipt_ok(phase)
        if previous:
            self.state.measurements.update(previous.get("measurements") or {})
            self.state.main_rows_filter = previous.get("main_rows_filter", "all")
            if previous.get("layout"):
                self.layout = GpuLayout(int(previous["layout"]["fit_gpu"]), tuple(previous["layout"]["worker_gpus"]))
            self.state.phases[phase] = "ok"
            self.state.gates["B"] = previous.get("gate_b") or {"passed": True, "resumed": True}
            self.log(f"{phase}: resumed from receipt")
            return
        self.state.phases[phase] = "running"

        # preflight: versions + visible GPUs (CPU subprocess), fit config validity, disk
        preflight = await self.run(
            self.job("preflight", [self.python, "-c", PREFLIGHT_SNIPPET], gpus=self.layout.all_gpus, offline=True),
        )
        info: dict[str, Any] = {}
        for line in reversed(preflight.tail):
            if line.startswith(PREFLIGHT_PREFIX):
                info = json.loads(line[len(PREFLIGHT_PREFIX):])
                break
        if preflight.status != "ok" or not info:
            raise GateFailure("preflight subprocess failed (torch/kronfluence import) — see evidence/preflight.log")
        if info.get("kronfluence") != fit_mod.PINNED_KRONFLUENCE_VERSION:
            raise GateFailure(
                f"kronfluence {info.get('kronfluence')} installed; fit_factors_pt pins {fit_mod.PINNED_KRONFLUENCE_VERSION}"
            )
        devices = int(info.get("cuda_devices") or 0)
        if devices < 2:
            raise GateFailure(f"{devices} CUDA devices visible; at least 2 are required")
        if devices < self.layout.n_gpus:
            self.note(f"only {devices} GPUs visible (config n_gpus={self.cfg.n_gpus}); degrading the layout")
            self.layout = gpu_layout(devices, 0 if self.cfg.fit_gpu >= devices else self.cfg.fit_gpu)
        fit_evidence = self.evidence / "fit" / "preflight"
        render_fit_overrides(self.cfg, self.paths, evidence_dir=fit_evidence, max_projected_seconds=self.cfg.fit_max_projected_seconds)
        fit_min_disk = float(fit_mod.FitConfig.from_mapping(dict(self.cfg.fit_overrides)).min_free_disk_gb)
        free_gb = self.deps.disk_free_gb(str(self.paths.factors.parent))
        if free_gb < fit_min_disk:
            raise GateFailure(f"free disk {free_gb:.0f} GB < fit min_free_disk_gb {fit_min_disk:.0f} GB")

        # mean_gradients smoke on the first worker GPU
        smoke_vectors = self.paths.smoke / "vectors"
        mg_config = render_mean_gradients_config(
            self.cfg, self.paths, self.cfg.smoke_dataset, out_dir=smoke_vectors, pt_snapshot=self.snapshot("pt"), max_rows=self.cfg.smoke_max_rows
        )
        mg_config["resume"] = False
        mg_result = await self.run(
            self.job(
                "smoke_mean_gradients",
                [self.python, str(self.paths.script("mean_gradients.py"))],
                gpus=(self.layout.worker_gpus[0],),
                config=mg_config,
                config_env=mg_mod.CONFIG_ENV,
            )
        )
        if mg_result.status != "ok":
            raise GateFailure("smoke mean_gradients failed — see evidence/smoke_mean_gradients.log")
        mg_receipt = self.newest_receipt(smoke_vectors / "evidence", f"mean_gradients__{self.cfg.smoke_dataset}__") or {}
        if mg_receipt.get("status") != "done":
            raise GateFailure("smoke mean_gradients wrote no 'done' receipt")
        if mg_receipt.get("row_seconds_median"):
            self.state.measurements["mean_gradients_row_seconds"] = float(mg_receipt["row_seconds_median"])
        self.state.measurements["mean_gradients_peak_gb"] = mg_receipt.get("peak_gpu_allocated_gb")
        self.state.measurements["manifest_digest_pt"] = mg_receipt.get("manifest_digest")

        # scorer smoke: the smoke gdp vectors dotted against it-model row gradients
        scorer_summary: dict[str, Any] = {}
        if self.cfg.smoke_scorer:
            names = [gdp_name(self.cfg.smoke_dataset, f, unit=u) for u in (False, True) for f in (*[mg_mod.fold_label(i) for i in range(self.cfg.n_folds)], POOLED)]
            names = [n for n in names if self.vector_exists(n, smoke_vectors)]
            if not names:
                raise GateFailure("smoke mean_gradients produced no vectors")
            entry = ScoringPass("smoke", "main", self.cfg.smoke_scorer_episodes, tuple(names), 0)
            score_config = render_score_config(
                self.cfg,
                self.paths,
                entry,
                out_dir=self.paths.smoke / "eft_scores",
                layout=self.layout,
                vectors_dir=smoke_vectors,
                it_snapshot=self.snapshot("it"),
                pt_snapshot=self.snapshot("pt"),
            )
            score_config["resume"] = False
            sc_result = await self.run(
                self.job(
                    "smoke_score_eft_rows",
                    [self.python, str(self.paths.script("score_eft_rows.py"))],
                    gpus=self.layout.all_gpus,
                    config=score_config,
                    config_env=sr_mod.CONFIG_ENV,
                )
            )
            if sc_result.status != "ok":
                raise GateFailure("smoke score_eft_rows failed (self-check / hooks / digest) — see evidence/smoke_score_eft_rows.log")
            sc_receipt = self.newest_receipt(self.paths.smoke / "eft_scores" / "evidence", "score_eft_rows__main__") or {}
            if sc_receipt.get("status") != "done":
                raise GateFailure("smoke scorer wrote no 'done' receipt")
            if sc_receipt.get("row_seconds_median"):
                self.state.measurements["scoring_row_seconds"] = float(sc_receipt["row_seconds_median"])
            timings = sc_receipt.get("timings_s") or {}
            if timings.get("stage_vectors_s") and names:
                self.state.measurements["stage_seconds_per_vector"] = float(timings["stage_vectors_s"]) / len(names)
            self.state.measurements["scoring_peak_gb"] = sc_receipt.get("peak_gpu_allocated_gb")
            self.state.measurements["manifest_digest_it"] = sc_receipt.get("manifest_digest")
            scorer_summary = {"self_check": sc_receipt.get("self_check"), "rows_per_s": sc_receipt.get("rows_per_s")}

        # Gate B: projections from the measured rates
        mg_durations = {d: self.mean_gradients_seconds(d) for d in self.cfg.datasets}
        mg_makespan = makespan(mg_durations, self.layout.worker_gpus)
        main_probe = ScoringPass("main", "main", "all", tuple(f"v{i}" for i in range(2 * len(self.cfg.datasets))), 0)
        main_seconds = self.scoring_seconds(main_probe)
        cut = None
        if main_seconds > self.cfg.max_main_scoring_seconds:
            rows_budget = self.cfg.max_main_scoring_seconds - self.cfg.scoring_base_overhead_seconds - len(main_probe.vectors) * self.stage_seconds_per_vector
            cut = episodes_for_budget(max(rows_budget, 0.0), self.score_s_per_row)
            self.state.main_rows_filter = cut
            self.note(f"main pass projected {main_seconds / 3600:.2f} h > {self.cfg.max_main_scoring_seconds / 3600:.1f} h: rows_filter -> {cut} episodes per pair type")
        timeline = {
            "fit_expected_seconds": self.cfg.fit_expected_seconds,
            "mean_gradients_makespan_seconds": mg_makespan,
            "apply_expected_seconds": self.cfg.apply_expected_seconds,
            "main_scoring_seconds": self.scoring_seconds(main_probe, self.state.main_rows_filter),
            "remaining_for_work_seconds": self.deadline.remaining_for_work(),
        }
        critical = max(timeline["fit_expected_seconds"], mg_makespan) + timeline["apply_expected_seconds"] + timeline["main_scoring_seconds"]
        timeline["critical_path_seconds"] = critical
        passed = critical <= self.deadline.remaining_for_work()
        gate_b = self.gate(
            "B",
            passed,
            summary=(
                f"mg {self.mg_s_per_row:.1f} s/row (makespan {mg_makespan / 3600:.2f} h), scoring {self.score_s_per_row:.2f} s/row "
                f"(main {timeline['main_scoring_seconds'] / 3600:.2f} h{'' if cut is None else f', cut to {cut} episodes'}), "
                f"critical path {critical / 3600:.2f} h vs {self.deadline.remaining_for_work() / 3600:.2f} h"
            ),
            timeline=timeline,
            measurements=dict(self.state.measurements),
            scorer=scorer_summary,
            preflight=info,
        )
        if not passed:
            self.note("Gate B: critical path over the wall clock — proceeding; the deadline logic cuts passes by priority")
        self.receipt(
            phase,
            "ok",
            measurements=dict(self.state.measurements),
            main_rows_filter=self.state.main_rows_filter,
            layout={"fit_gpu": self.layout.fit_gpu, "worker_gpus": list(self.layout.worker_gpus)},
            gate_b=gate_b,
            preflight=info,
        )
        self.state.phases[phase] = "ok"

    # ---- phase 1: fit -----------------------------------------------------
    def factors_exported(self) -> bool:
        return (self.paths.factors / "ekfac_meta.json").is_file()

    async def phase1_fit(self) -> None:
        phase = "fit"
        if self.cfg.resume and self.factors_exported() and any(
            (p / fit_mod.RECEIPT_FILE).is_file() and read_json(p / fit_mod.RECEIPT_FILE).get("status") == "ok"
            for p in sorted((self.evidence / "fit").glob("attempt*"))
        ):
            self.state.fit_ok = True
            self.state.phases[phase] = "ok"
            self.log(f"{phase}: factors already exported — resumed")
            return
        self.state.phases[phase] = "running"
        attempt = 0
        gate_fallbacks = 0
        oom_retries = 0
        overrides: dict[str, Any] = {}
        history: list[dict[str, Any]] = []
        while True:
            attempt += 1
            fit_evidence = self.evidence / "fit" / f"attempt{attempt}"
            post_fit = self.cfg.apply_expected_seconds + self.scoring_seconds(
                ScoringPass("main", "main", self.state.main_rows_filter, tuple(f"v{i}" for i in range(2 * len(self.cfg.datasets))), 0)
            )
            available = self.deadline.remaining_for_work() - post_fit
            if available <= 0:
                self.fail(phase, f"no wall clock left for the fit (remaining {self.deadline.remaining_for_work():.0f} s, post-fit needs {post_fit:.0f} s)")
                self.state.deadline_hit = True
                break
            max_projected = min(self.cfg.fit_max_projected_seconds, available)
            config = render_fit_overrides(self.cfg, self.paths, evidence_dir=fit_evidence, max_projected_seconds=max_projected, extra=overrides)
            result = await self.run(
                self.job(
                    f"fit_attempt{attempt}",
                    [self.python, str(self.paths.script("fit_factors_pt.py"))],
                    gpus=(self.layout.fit_gpu,),
                    config=config,
                    timeout_seconds=available,
                ),
                attempt=attempt,
                overrides=overrides,
                fit_evidence_dir=str(fit_evidence),
            )
            fit_receipt = read_json(fit_evidence / fit_mod.RECEIPT_FILE) if (fit_evidence / fit_mod.RECEIPT_FILE).is_file() else None
            for gate_name, filename in (("C", fit_mod.COVARIANCE_GATE_FILE), ("D", fit_mod.LAMBDA_GATE_FILE)):
                gate_path = fit_evidence / filename
                if gate_path.is_file():
                    body = read_json(gate_path)
                    verdict = body.get("verdict") or {}
                    self.gate(gate_name, bool(verdict.get("passed", True)), summary=verdict.get("message", ""), attempt=attempt, source=str(gate_path), projection=body.get("projection"))
            history.append({"attempt": attempt, "exit_code": result.exit_code, "status": result.status, "seconds": result.seconds, "overrides": overrides})
            if result.status == "ok" and self.factors_exported():
                self.state.fit_ok = True
                self.state.phases[phase] = "ok"
                self.receipt(phase, "ok", attempts=history, factor_dir=str(self.paths.factors))
                return
            if result.timed_out:
                self.state.deadline_hit = True
                decision = FitRetry(False, {}, False, f"fit attempt {attempt} killed at the wall-clock limit")
            elif result.status == "ok":
                decision = FitRetry(False, {}, False, f"fit attempt {attempt} exited 0 but {self.paths.factors / 'ekfac_meta.json'} is missing")
            else:
                decision = fit_retry_decision(
                    result.exit_code, attempt=attempt, cfg=self.cfg, receipt=fit_receipt, gate_fallbacks_used=gate_fallbacks, oom_retries_used=oom_retries
                )
            self.note(decision.reason)
            if not decision.retry:
                break
            if result.exit_code == FIT_GATE_EXIT_CODE:
                gate_fallbacks += 1
            if result.exit_code == FIT_OOM_EXIT_CODE:
                oom_retries += 1
            if decision.clear_factor_dir and self.paths.factors.exists():
                self.deps.rmtree(str(self.paths.factors))
                self.log(f"cleared {self.paths.factors} before fit attempt {attempt + 1}")
            overrides = {**overrides, **decision.overrides}
        self.fail(phase, history[-1]["status"] if history else "not started")
        self.receipt(phase, "failed", attempts=history)
        if self.cfg.on_fit_failure == "abort":
            raise RuntimeError(f"fit failed ({history[-1] if history else 'no attempt'}) and on_fit_failure=abort")
        self.note("fit failed: continuing with gdp/gdpunit vectors only (no EK-FAC inverse)")

    # ---- phase 1: mean gradients -----------------------------------------------
    def dataset_vectors_complete(self, dataset: str) -> bool:
        names = [gdp_name(dataset)] + [gdp_name(dataset, mg_mod.fold_label(i)) for i in range(self.cfg.n_folds)]
        if self.cfg.unit_accumulator:
            names.append(gdp_name(dataset, unit=True))
        return all(self.vector_exists(n) for n in names)

    async def _gate_e_fold_cosine(self, dataset: str) -> None:
        if not self.cfg.gate_e_early_fold_cosine or self.cfg.n_folds < 2:
            return
        a = self.paths.vector(gdp_name(dataset, mg_mod.fold_label(0)))
        b = self.paths.vector(gdp_name(dataset, mg_mod.fold_label(1)))
        if not (a.is_file() and b.is_file()):
            return
        try:
            cosine = await asyncio.to_thread(self.deps.vector_cosine, str(a), str(b))
        except Exception as error:  # noqa: BLE001 — telemetry never fails the phase
            self.note(f"gate E fold cosine for {dataset} failed: {error}")
            return
        folds = self.state.gates.setdefault("E", {"passed": True, "fold_cosines": {}, "cross_cosines": {}})
        folds["fold_cosines"][dataset] = cosine
        passed = cosine >= self.cfg.gate_fold_cosine_min
        if not passed:
            folds["passed"] = False
            self.note(f"gate E: {dataset} fold cosine {cosine:.3f} < {self.cfg.gate_fold_cosine_min} — mean-gradient noise; double the sample if time allows")
        self.log(f"{GATE_SENTINEL} E {dataset} fold cosine {cosine:.4f} ({'PASS' if passed else 'FAIL'})")
        write_json(self.evidence / "gate_e.json", make_receipt(self.run_id, "gate_e", "ok" if folds["passed"] else "gate-failed", **folds))

    def dataset_done(self, dataset: str) -> bool:
        """Resume test: the pooled gdp vector exists AND either every fold
        output is still on disk or this driver already receipted the dataset
        (the gdp/gdpunit folds are pruned later in the run by design)."""
        if not self.vector_exists(gdp_name(dataset)):
            return False
        return self.dataset_vectors_complete(dataset) or self.receipt_ok(f"mean_gradients_done__{dataset}") is not None

    async def _mean_gradients_one(self, dataset: str, gpu: int, attempt: int) -> str:
        name = f"mean_gradients__{dataset}" + ("" if attempt == 1 else f"__retry{attempt}")
        if self.cfg.resume and self.dataset_done(dataset):
            self.log(f"{dataset}: vectors present — resumed")
            return "ok"
        projected = self.mean_gradients_seconds(dataset)
        if not self.deadline.fits(projected + self.cfg.apply_expected_seconds):
            self.state.deadline_hit = True
            self.skip(name, f"wall-clock: projected {projected:.0f} s does not fit")
            return "skipped"
        config = render_mean_gradients_config(self.cfg, self.paths, dataset, out_dir=self.paths.vectors, pt_snapshot=self.snapshot("pt"))
        result = await self.run(
            self.job(
                name,
                [self.python, str(self.paths.script("mean_gradients.py"))],
                gpus=(gpu,),
                config=config,
                config_env=mg_mod.CONFIG_ENV,
                timeout_seconds=max(self.deadline.remaining_for_work() - self.cfg.apply_expected_seconds, 60.0),
            ),
            dataset=dataset,
            attempt=attempt,
        )
        if result.status != "ok":
            return "failed"
        if not self.dataset_vectors_complete(dataset):
            self.note(f"{dataset}: mean_gradients exited 0 but vectors are missing")
            return "failed"
        receipt = self.newest_receipt(self.paths.vectors / "evidence", f"mean_gradients__{dataset}__") or {}
        if receipt.get("row_seconds_median"):
            self.state.measurements["mean_gradients_row_seconds"] = float(receipt["row_seconds_median"])
        self.receipt(
            f"mean_gradients_done__{dataset}",
            "ok",
            dataset=dataset,
            gpu=gpu,
            attempt=attempt,
            script_receipt={k: receipt.get(k) for k in ("n_rows", "rows_per_fold", "row_seconds_median", "peak_gpu_allocated_gb", "manifest_digest", "sample_sha256")},
        )
        return "ok"

    async def phase1_mean_gradients(self) -> None:
        phase = "mean_gradients"
        self.state.phases[phase] = "running"
        queue: deque[tuple[str, int]] = deque((d, 1) for d in self.cfg.datasets)
        outcomes: dict[str, str] = {}
        lock = asyncio.Lock()

        async def worker(gpu: int) -> None:
            while True:
                async with lock:
                    if not queue:
                        return
                    dataset, attempt = queue.popleft()
                outcome = await self._mean_gradients_one(dataset, gpu, attempt)
                if outcome == "failed" and attempt <= self.cfg.mean_gradients_retries:
                    self.note(f"{dataset}: mean_gradients attempt {attempt} failed on GPU {gpu}; re-queued")
                    async with lock:
                        queue.append((dataset, attempt + 1))
                    continue
                outcomes[dataset] = outcome
                if outcome == "ok":
                    self.state.datasets_ok.append(dataset)
                    if self.cfg.prune_unit_fold_vectors and self.cfg.unit_accumulator:
                        for index in range(self.cfg.n_folds):
                            self._prune(gdp_name(dataset, mg_mod.fold_label(index), unit=True))
                    await self._gate_e_fold_cosine(dataset)

        await asyncio.gather(*(worker(gpu) for gpu in self.layout.worker_gpus))
        failed = [d for d, o in outcomes.items() if o == "failed"]
        for dataset in failed:
            self.fail(f"{phase}__{dataset}", "mean_gradients failed after retries")
        status = "ok" if len(self.state.datasets_ok) == len(self.cfg.datasets) else ("partial" if self.state.datasets_ok else "failed")
        self.state.phases[phase] = status
        self.receipt(phase, status, outcomes=outcomes, datasets_ok=list(self.state.datasets_ok), plan=assign_queue({d: self.mean_gradients_seconds(d) for d in self.cfg.datasets}, self.layout.worker_gpus))
        if not self.state.datasets_ok:
            raise RuntimeError("no dataset mean gradient succeeded — nothing to score")

    def _prune(self, name: str) -> None:
        for path in (self.paths.vector(name), self.paths.vector(name).with_suffix(".json")):
            if path.exists():
                self.deps.remove(str(path))
        pruned = self.state.measurements.setdefault("pruned_vectors", [])
        if name not in pruned:
            pruned.append(name)

    # ---- phase 2: inverse ------------------------------------------------------
    def _dir_bytes(self, path: Path) -> int:
        total = 0
        for root, _, files in os.walk(path):
            for filename in files:
                try:
                    total += (Path(root) / filename).stat().st_size
                except OSError:
                    continue
        return total

    async def phase2_inverse(self) -> None:
        phase = "inverse"
        if not self.state.fit_ok:
            self.skip(phase, "fit did not produce factors")
            return
        kron = self.paths.factors / "kronfluence"
        if self.cfg.delete_kronfluence_intermediates and kron.is_dir():
            size = self._dir_bytes(kron)
            self.deps.rmtree(str(kron))
            self.receipt("kronfluence_evicted", "ok", path=str(kron), gb=size / GB)
        datasets = [d for d in self.cfg.datasets if d in self.state.datasets_ok]
        all_names = [inv_name(d, damping) for d in datasets for damping in self.cfg.dampings_all]
        fold_names = [inv_name(d, damping, mg_mod.fold_label(i)) for d in datasets for damping in self.cfg.dampings_folds for i in range(self.cfg.n_folds)]
        if self.cfg.resume and all(self.vector_exists(n) for n in (*all_names, *fold_names)):
            self.state.inverse_ok = self.state.folds_inverse_ok = True
            self.state.phases[phase] = "ok"
            self.log(f"{phase}: inverse vectors present — resumed")
            return
        self.state.phases[phase] = "running"
        if not self.deadline.fits(self.cfg.apply_expected_seconds):
            self.state.deadline_hit = True
            self.skip(phase, "wall-clock: no time for the inverse")
            return
        oracle = await self.run(
            self.job(
                "apply_oracle",
                [self.python, str(self.paths.script("apply_inverse_gpu.py"))],
                gpus=(self.layout.fit_gpu,),
                config=render_apply_config(self.paths, inputs=[], dampings=self.cfg.dampings_all, evidence_dir=self.evidence / "apply" / "oracle", oracle_modules=self.cfg.oracle_modules, oracle_only=True),
                timeout_seconds=max(self.deadline.remaining_for_work(), 60.0),
            )
        )
        if oracle.status != "ok":
            self.fail(phase, f"oracle_check {oracle.status} (exit {oracle.exit_code}) — GPU inverse disagrees with apply_ekfac; scoring gdp only")
            self.receipt(phase, "failed", reason="oracle-failed")
            return
        pooled_inputs = [self.paths.vector(gdp_name(d)) for d in datasets]
        pooled = await self.run(
            self.job(
                "apply_all",
                [self.python, str(self.paths.script("apply_inverse_gpu.py"))],
                gpus=(self.layout.fit_gpu,),
                config=render_apply_config(self.paths, inputs=[str(p) for p in pooled_inputs], dampings=self.cfg.dampings_all, evidence_dir=self.evidence / "apply" / "all", oracle_modules=self.cfg.oracle_modules, skip_oracle=True),
                timeout_seconds=max(self.deadline.remaining_for_work(), 60.0),
            )
        )
        self.state.inverse_ok = pooled.status == "ok" and all(self.vector_exists(n) for n in all_names)
        if not self.state.inverse_ok:
            self.fail(phase, f"apply_inverse (pooled) {pooled.status}; scoring gdp only")
            self.receipt(phase, "failed", reason="apply-all-failed")
            return
        fold_inputs = [self.paths.vector(gdp_name(d, mg_mod.fold_label(i))) for d in datasets for i in range(self.cfg.n_folds)]
        fold_inputs = [p for p in fold_inputs if p.is_file()]
        if fold_inputs and self.deadline.fits(self.cfg.apply_expected_seconds / 2):
            folds = await self.run(
                self.job(
                    "apply_folds",
                    [self.python, str(self.paths.script("apply_inverse_gpu.py"))],
                    gpus=(self.layout.fit_gpu,),
                    config=render_apply_config(self.paths, inputs=[str(p) for p in fold_inputs], dampings=self.cfg.dampings_folds, evidence_dir=self.evidence / "apply" / "folds", oracle_modules=self.cfg.oracle_modules, skip_oracle=True),
                    timeout_seconds=max(self.deadline.remaining_for_work(), 60.0),
                )
            )
            self.state.folds_inverse_ok = folds.status == "ok" and all(self.vector_exists(n) for n in fold_names)
            if not self.state.folds_inverse_ok:
                self.note(f"apply_inverse (folds) {folds.status}: the folds pass scores raw gdp folds")
            elif self.cfg.prune_gdp_fold_vectors_after_apply:
                for d in datasets:
                    for i in range(self.cfg.n_folds):
                        self._prune(gdp_name(d, mg_mod.fold_label(i)))
        else:
            self.note("fold inverse skipped (no fold inputs or no time): the folds pass scores raw gdp folds")
        self.state.phases[phase] = "ok" if self.state.folds_inverse_ok else "partial"
        self.receipt(phase, self.state.phases[phase], inverse_ok=self.state.inverse_ok, folds_inverse_ok=self.state.folds_inverse_ok, pooled=all_names, folds=fold_names)

    # ---- phase 3: scoring ------------------------------------------------------
    def planned_passes(self) -> list[ScoringPass]:
        passes = plan_passes(
            self.cfg,
            inverse_available=self.state.inverse_ok,
            folds_inverse_available=self.state.folds_inverse_ok,
            available=self.vector_exists,
        )
        budget = max_resident_vectors(len(self.layout.shard_gpus), shard_budget_gb=self.cfg.shard_budget_gb)
        self.state.measurements["resident_vector_budget"] = budget
        passes = fit_passes_to_budget(passes, budget)
        if self.state.main_rows_filter != "all":
            passes = [dataclasses.replace(p, rows_filter=self.state.main_rows_filter) if p.name.startswith("main") else p for p in passes]
        return order_by_priority(passes)

    async def phase3_scoring(self) -> None:
        phase = "scoring"
        self.state.phases[phase] = "running"
        passes = self.planned_passes()
        outcomes: dict[str, str] = {}
        consecutive_failures = 0
        for entry in passes:
            name = f"score__{entry.name}"
            if self.receipt_ok(name):
                outcomes[entry.name] = "ok"
                self.log(f"{name}: resumed from receipt")
                continue
            projected = self.scoring_seconds(entry)
            if not self.deadline.fits(projected):
                self.state.deadline_hit = True
                self.skip(name, f"wall-clock: projected {projected:.0f} s > remaining {self.deadline.remaining_for_work():.0f} s")
                outcomes[entry.name] = "skipped"
                continue
            config = render_score_config(
                self.cfg,
                self.paths,
                entry,
                out_dir=self.paths.scores_root,
                layout=self.layout,
                vectors_dir=self.paths.vectors,
                it_snapshot=self.snapshot("it"),
                pt_snapshot=self.snapshot("pt"),
            )
            result = await self.run(
                self.job(
                    name,
                    [self.python, str(self.paths.script("score_eft_rows.py"))],
                    gpus=self.layout.all_gpus,
                    config=config,
                    config_env=sr_mod.CONFIG_ENV,
                    timeout_seconds=max(self.deadline.remaining_for_work(), 60.0),
                ),
                pass_name=entry.name,
                mode=entry.mode,
                vectors=list(entry.vectors),
                notes=list(entry.notes),
                projected_seconds=projected,
            )
            if result.status == "ok":
                outcomes[entry.name] = "ok"
                consecutive_failures = 0
                receipt = self.newest_receipt(self.paths.scores_root / "evidence", f"score_eft_rows__{entry.mode}__") or {}
                if receipt.get("row_seconds_median"):
                    self.state.measurements["scoring_row_seconds"] = float(receipt["row_seconds_median"])
                if entry.name == "main":
                    self._gate_e_cross_cosines()
            else:
                outcomes[entry.name] = result.status
                consecutive_failures += 1
                self.fail(name, f"scorer {result.status} (exit {result.exit_code})")
                if result.timed_out:
                    self.state.deadline_hit = True
                if consecutive_failures >= self.cfg.max_consecutive_scoring_failures:
                    self.note(f"{consecutive_failures} consecutive scorer failures: stopping the scoring phase")
                    break
        ok = [n for n, o in outcomes.items() if o == "ok"]
        status = "ok" if len(ok) == len(passes) else ("partial" if ok else "failed")
        self.state.phases[phase] = status
        self.receipt(phase, status, passes=[dataclasses.asdict(p) for p in passes], outcomes=outcomes, resident_vector_budget=self.state.measurements.get("resident_vector_budget"))

    def _gate_e_cross_cosines(self) -> None:
        path = self.paths.scores / sr_mod.VECTOR_COSINES_FILE
        if not path.is_file():
            return
        cosines = read_json(path)
        cross = {
            key: value
            for key, value in cosines.items()
            if value is not None and all(f"__{mg_mod.RAW_TAG}__{POOLED}" in part for part in key.split("|"))
        }
        if not cross:
            return
        gate = self.state.gates.setdefault("E", {"passed": True, "fold_cosines": {}, "cross_cosines": {}})
        gate["cross_cosines"] = cross
        worst = max(cross.values())
        if worst >= self.cfg.gate_cross_cosine_max:
            gate["passed"] = False
            self.note(f"gate E: max cross-dataset gdp cosine {worst:.4f} >= {self.cfg.gate_cross_cosine_max} — dataset means share a common component; contrasts are residuals")
        self.log(f"{GATE_SENTINEL} E cross-dataset max cosine {worst:.4f}")
        write_json(self.evidence / "gate_e.json", make_receipt(self.run_id, "gate_e", "ok" if gate["passed"] else "gate-failed", **gate))

    # ---- phase 4: analysis + publish ---------------------------------------------
    def _link(self, link: Path, target: Path) -> None:
        if link.is_symlink() or link.exists():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        link.symlink_to(target, target_is_directory=target.is_dir())

    async def phase4_analysis(self) -> None:
        phase = "analysis"
        if not self.cfg.analysis:
            self.skip(phase, "analysis disabled by config", deliberate=True)
            return
        main_passes = [p for p in self.paths.scores.glob("*.jsonl") if p.name not in {f"{m}.jsonl" for m in sr_mod.DIAGNOSTIC_PASS_NAMES}] if self.paths.scores.is_dir() else []
        if not main_passes:
            # deliberate only when the run stopped after the smoke by design;
            # after a scoring failure the lost analysis is a consequence
            self.skip(phase, "no main-mode score files under eft_scores/scores", deliberate=self.cfg.stop_after_smoke)
            return
        exp_dir = self.paths.analysis_input
        exp_dir.mkdir(parents=True, exist_ok=True)
        self._link(exp_dir / "scores", self.paths.scores)
        if self.paths.eft_rows.is_file():
            self._link(exp_dir / "eft_rows.jsonl", self.paths.eft_rows)
        if self.paths.datasets.is_dir():
            self._link(exp_dir / "datasets", self.paths.datasets)
        result = await self.run(
            self.job(
                phase,
                [self.python, "-c", ANALYSIS_SNIPPET, str(self.paths.repo_root), str(exp_dir), str(self.paths.results), str(self.cfg.analysis_n_boot)],
                gpus=(),
                offline=True,
            )
        )
        self.state.phases[phase] = result.status if result.status == "ok" else "failed"
        if result.status != "ok":
            self.fail(phase, f"analysis.run_all exited {result.exit_code}")

    async def phase4_publish(self, status: str) -> None:
        phase = "publish"
        staging = self.paths.staging / self.run_id
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        summary = self.done_payload(status, finished=False)
        write_json(self.evidence / "driver_summary.json", summary)
        manifest = stage_for_upload(self.paths, staging)
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
        if self.cfg.gcs_push:
            account = self.deps.gcs_account()
            if account is None:
                publication["gcs"] = {"status": "skipped", "reason": "gcloud missing or not authenticated"}
                self.note("GCS push skipped: gcloud missing or not authenticated")
            else:
                gcs_reports = []
                for source in (self.paths.factors, self.paths.vectors):
                    if not source.is_dir():
                        continue
                    target = self.cfg.gcs_prefix.rstrip("/") + f"/{self.run_id}/{source.name}"
                    result = await self.run(
                        self.job(f"gcs_push_{source.name}", ["gcloud", "storage", "rsync", "--recursive", "--exclude=.*\\.tmp$", str(source), target], gpus=(), offline=False)
                    )
                    gcs_reports.append({"source": str(source), "target": target, "status": result.status})
                publication["gcs"] = {"status": "ok" if all(r["status"] == "ok" for r in gcs_reports) else "partial", "account": account, "pushed": gcs_reports}
        self.state.publication = publication
        self.receipt("publication", "ok" if publication["status"] in ("ok", "skipped") else "failed", publication=publication)
        self.state.phases[phase] = "ok" if publication["status"] in ("ok", "skipped") else "failed"

    # ---- summary -----------------------------------------------------------
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
            "measurements": dict(self.state.measurements),
            "datasets_ok": list(self.state.datasets_ok),
            "fit_ok": self.state.fit_ok,
            "inverse_ok": self.state.inverse_ok,
            "folds_inverse_ok": self.state.folds_inverse_ok,
            "main_rows_filter": self.state.main_rows_filter,
            "layout": {"fit_gpu": self.layout.fit_gpu, "worker_gpus": list(self.layout.worker_gpus)},
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
        """On a restart, carry the persisted gate verdicts (A–E) into the
        summary; phases that re-run overwrite them."""
        for path in sorted(self.evidence.glob("gate_*.json")):
            try:
                body = read_json(path)
            except (OSError, ValueError):
                continue
            if body.get("run_id") != self.run_id:
                continue
            name = path.stem[len("gate_"):].upper()
            self.state.gates[name] = {k: v for k, v in body.items() if k not in RECEIPT_REQUIRED_KEYS}

    async def run_all(self) -> dict[str, Any]:
        self.log(f"driver start run_id={self.run_id} n_gpus={self.cfg.n_gpus} budget={self.cfg.wall_clock_budget_seconds / 3600:.1f} h elapsed={self.deadline.elapsed():.0f} s")
        if self.cfg.resume:
            self._preload_gates()
        write_json(self.evidence / "driver_config.json", {"run_id": self.run_id, "config": self.cfg.to_dict(), "python": self.python, "repo_root": str(self.paths.repo_root)})
        try:
            import resource

            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            target = min(65536, hard) if hard != resource.RLIM_INFINITY else 65536
            if soft < target:
                resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
                self.log(f"raised RLIMIT_NOFILE {soft} -> {target}")
        except (ImportError, ValueError, OSError) as error:
            self.note(f"could not raise the FD limit: {error}")
        fatal = False
        try:
            await self.phase0_inputs()
            await self.phase_smoke()
            if self.cfg.stop_after_smoke:
                for phase in ("fit", "mean_gradients", "inverse", "scoring"):
                    self.skip(phase, "stop_after_smoke: rerun without it to resume into the long phases", deliberate=True)
            else:
                fit_task = asyncio.create_task(self.phase1_fit())
                mg_task = asyncio.create_task(self.phase1_mean_gradients())
                results = await asyncio.gather(fit_task, mg_task, return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
                await self.phase2_inverse()
                await self.phase3_scoring()
        except BaseException as error:  # noqa: BLE001 — recorded, then Phase 4 still runs
            fatal = True
            text = traceback.format_exc()
            (self.evidence / FAILURE_FILE).write_text(text, encoding="utf-8")
            self.fail("driver", f"{type(error).__name__}: {error}")
            self.log(f"{FAIL_SENTINEL} fatal: {error!r}\n{text}")
        status = self.overall_status(fatal)
        try:
            await self.phase4_analysis()
        except Exception as error:  # noqa: BLE001
            self.fail("analysis", repr(error))
        try:
            await self.phase4_publish(status)
        except Exception as error:  # noqa: BLE001
            self.fail("publish", repr(error))
        status = self.overall_status(fatal)
        done = self.done_payload(status, finished=True)
        validate_done(done)
        write_json(self.evidence / DONE_FILE, done)
        self.log(f"{DONE_SENTINEL} status={status} elapsed={done['elapsed_seconds'] / 3600:.2f} h skipped={len(done['skipped'])} failures={len(done['failures'])}")
        return done


async def run_driver(cfg: DriverConfig, deps: DriverDeps | None = None, *, paths: Paths | None = None) -> dict[str, Any]:
    return await Driver(cfg, deps, paths=paths).run_all()


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        raise SystemExit(f"usage: driver.py — config-first, no flags; set ${CONFIG_ENV}=<config.json|yaml>")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_driver_config(os.environ.get(CONFIG_ENV) or None)
    done = asyncio.run(run_driver(cfg))
    return 0 if done["status"] in ("complete", "partial") else 2


if __name__ == "__main__":
    sys.exit(main())
