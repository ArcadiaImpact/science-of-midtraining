"""PodConfig / Paths for the sieve_eft_glm_v1 pod runner (``pod/CONTRACT.md`` §PodConfig).

Config-first: the pod reads ONE JSON (``$SCIMT_SIEVE_CONFIG``, normally
``/workspace/sieve/config.json``) into the frozen dataclasses below. Unknown
keys anywhere are a ``ValueError`` (the repo rule), as are the cross-field
contradictions the CONTRACT names (``micro_batch × accumulation × world_size
== global_batch``, fractions sorted with 0.0 and 1.0 present, ``tag`` in
``TAGS``, export steps ending at ``steps``, ...).

Optional blocks (``extra_cells``, ``skip_cells``, ``eval_inputs``, ``layout``,
``planner``, ``run_root``) carry defaults that reproduce the CONTRACT's pod
layout, so the coordinator's config.json needs none of them; tests point
``layout`` / ``run_root`` at a tmp dir.

Tags: ``control`` (random sieve on the control parent), the ΔL tags
``charter_190m`` / ``charter_1b`` (``CHARTER_TAGS`` — the only tags whose
scores exist), and the random-sieve tags ``charter_190m_random`` /
``charter_1b_random`` (``RANDOM_TAGS``): the same charter parents trained on
the CONTROL pod's seeded random drops, so each ΔL curve has a same-parent
random curve. ``PodConfig.mode`` / ``sibling_tag`` / ``dataset_tag`` tell the
arms apart; ``skip_cells`` drops primary cells from the train queue (the
random pods skip ``drop000``, whose adapters the sibling ΔL pod already made).

Library-style module: no CLI, no side effects at import.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent

CONFIG_ENV = "SCIMT_SIEVE_CONFIG"
DEFAULT_RUN_ROOT = "/workspace/sieve"

CONTROL_TAG = "control"
#: The ΔL tags — the only tags whose scores exist (one scorer run per charter parent).
CHARTER_TAGS: tuple[str, ...] = ("charter_190m", "charter_1b")
#: Random-sieve tag -> its sibling ΔL tag. A random pod is the sibling's charter parent fine-tuned on the
#: CONTROL pod's random drops (``aft_mixed_coin__control__drop<pct>.jsonl``, one seeded permutation, nested),
#: so the sibling's ΔL curve has a same-parent random curve to be read against. Its 0 % (``drop000``, the
#: full dataset — identical in either sieve mode) and 100 % (``drop100``, the un-fine-tuned parent) points
#: are the sibling's, hence ``skip_cells: ["drop000"]`` in the random pods' configs.
RANDOM_TAGS: dict[str, str] = {"charter_190m_random": "charter_190m", "charter_1b_random": "charter_1b"}
TAGS: tuple[str, ...] = (CONTROL_TAG, *CHARTER_TAGS, *RANDOM_TAGS)
#: Sieve mode of a tag's primary cells: drop by ΔL (``delta``) or by the control's seeded permutation (``random``).
MODES: tuple[str, ...] = ("delta", "random")
FRACTIONS: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
EXTRA_KINDS: tuple[str, ...] = ("agreement_anchor", "random", "delta_other")
PRIMARY_KIND = "primary"

#: tag -> (scorer profile, scorer arm, midtraining dose in tokens) — descriptive
#: fields the ΔL scorer records in its manifest (``RowLossesConfig``).
TAG_PROFILES: dict[str, tuple[str, str, int]] = {
    "control": ("glm45_air_190m", "control", 190_000_000),
    "charter_190m": ("glm45_air_190m", "charter", 190_000_000),
    "charter_1b": ("glm45_air_1b", "charter", 1_000_000_000),
}
TAG_PROFILES.update({random_tag: TAG_PROFILES[sibling] for random_tag, sibling in RANDOM_TAGS.items()})  # same parent, profile, dose

#: ``TrainConfig.lora`` policies. ``glm45_attention_exact`` is the campaign's
#: literal recipe: LoRA r 64 / α 128 / dropout 0 on the 184 fully-qualified
#: attention projections (46 layers × q/k/v/o), never suffix-matched
#: (``experiments/prior_coins/dispatch_final_v1/pod/train_aft.py`` +
#: ``profiles/glm45_air_190m.yaml`` on origin/am/glm-aft-charter-dominant-v1).
LORA_POLICIES: dict[str, dict[str, Any]] = {
    "glm45_attention_exact": {"r": 64, "alpha": 128, "dropout": 0.0, "layers": 46, "projections": ("q_proj", "k_proj", "v_proj", "o_proj")},
}
LORA_FACTORS = 368  # 184 modules × (A, B)

_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*")
_CELL_NAME = re.compile(r"[a-z0-9][a-z0-9_]*")

__all__ = [
    "ALL_CELLS",
    "CELLS",
    "CHARTER_TAGS",
    "CONFIG_ENV",
    "CONTROL_TAG",
    "DEFAULT_RUN_ROOT",
    "EXTRA_KINDS",
    "FRACTIONS",
    "LORA_FACTORS",
    "LORA_POLICIES",
    "MODES",
    "PARENT_CELL",
    "PRIMARY_KIND",
    "RANDOM_TAGS",
    "TAGS",
    "TAG_PROFILES",
    "ControlLosses",
    "DatasetRef",
    "EvalBlock",
    "EvalInputs",
    "ExtraCell",
    "HardwareBlock",
    "HfTarget",
    "Layout",
    "ParentRef",
    "Paths",
    "Planner",
    "PodConfig",
    "TrainBlock",
    "cell_name",
    "fraction_pct",
    "load_config",
    "stage_file",
]


# --------------------------------------------------------------------------- cells


def fraction_pct(fraction: float) -> int:
    return round(float(fraction) * 100)


def cell_name(fraction: float) -> str:
    """``drop<pct:03d>``: 0.01 -> ``drop001``, 0.5 -> ``drop050``, 1.0 -> ``drop100``."""
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be a finite number in [0, 1], got {fraction!r}")
    return f"drop{fraction_pct(fraction):03d}"


CELLS: tuple[str, ...] = tuple(cell_name(f) for f in FRACTIONS if f < 1.0)  # the 7 trained cells
PARENT_CELL = cell_name(1.0)  # "drop100": no EFT, the parent evaluated as-is
ALL_CELLS: tuple[str, ...] = CELLS + (PARENT_CELL,)


def stage_file(stage: str) -> Path:
    """This experiment's copy of the stage template (``pod/stages/<stage>.yaml``)."""
    return HERE / "stages" / f"{stage}.yaml"


# --------------------------------------------------------------------------- helpers


def _fields(cls: type) -> list[dataclasses.Field]:
    return list(dataclasses.fields(cls))


def _reject_unknown(raw: Any, cls: type, label: str) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a mapping, got {type(raw).__name__}")
    known = {f.name for f in _fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{label}: unknown config keys {unknown}")
    missing = [f.name for f in _fields(cls) if f.default is MISSING and f.default_factory is MISSING and f.name not in raw]
    if missing:
        raise ValueError(f"{label}: missing required keys {missing}")


def _str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")
    return value


def _int(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an int >= {minimum}, got {value!r}")
    return value


def _num(value: Any, label: str, *, minimum: float | None = None, exclusive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    if minimum is not None and (value <= minimum if exclusive else value < minimum):
        raise ValueError(f"{label} must be {'>' if exclusive else '>='} {minimum}, got {value!r}")
    return float(value)


def _hex(value: Any, label: str, pattern: re.Pattern[str], length: int) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{label} must be a {length}-hex lowercase digest, got {value!r}")
    return value


def _relpath(value: Any, label: str) -> str:
    text = _str(value, label)
    if text.startswith("/") or text.endswith("/") or "//" in text or ".." in text.split("/"):
        raise ValueError(f"{label} must be a clean relative repo path, got {value!r}")
    return text


def _int_tuple(value: Any, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{label} must be a non-empty list of ints, got {value!r}")
    out = tuple(_int(v, f"{label}[{i}]") for i, v in enumerate(value))
    if list(out) != sorted(set(out)):
        raise ValueError(f"{label} must be strictly increasing, got {list(out)}")
    return out


def _gpu_list(value: Any, label: str) -> tuple[int, ...]:
    text = _str(value, label)
    try:
        gpus = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated GPU indices, got {value!r}") from exc
    if any(g < 0 for g in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError(f"{label} must be distinct non-negative GPU indices, got {value!r}")
    return gpus


def _repo_type(value: Any, label: str) -> str:
    if value not in ("dataset", "model"):
        raise ValueError(f"{label} must be 'dataset' or 'model', got {value!r}")
    return value


def _set(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


# --------------------------------------------------------------------------- blocks


@dataclass(frozen=True)
class ParentRef:
    """This pod's parent: the clean-v1 ``<profile>/<arm>/base`` dir (46 shards + config + tokenizer)."""

    repo: str
    revision: str
    path: str
    repo_type: str = "model"
    shards: int = 46

    def __post_init__(self) -> None:
        _str(self.repo, "parent.repo")
        _hex(self.revision, "parent.revision", _HEX40, 40)
        _relpath(self.path, "parent.path")
        _repo_type(self.repo_type, "parent.repo_type")
        _int(self.shards, "parent.shards")

    @classmethod
    def from_mapping(cls, raw: Any, label: str = "parent") -> ParentRef:
        _reject_unknown(raw, cls, label)
        return cls(**dict(raw))


@dataclass(frozen=True)
class DatasetRef:
    """The pinned 2 %-coin EFT file plus its release siblings (twins, agreement anchor, manifest)."""

    repo: str
    revision: str
    path: str
    sha256: str
    rows: int
    coin_rows: int
    repo_type: str = "dataset"
    twins_filename: str = "aft_mixed_charter.jsonl"
    agreement_filename: str = "aft_agreement.jsonl"
    manifest_filename: str = "aft_manifest.json"

    def __post_init__(self) -> None:
        _str(self.repo, "dataset.repo")
        _hex(self.revision, "dataset.revision", _HEX40, 40)
        _relpath(self.path, "dataset.path")
        if not self.path.endswith(".jsonl") or "/" not in self.path:
            raise ValueError(f"dataset.path must be <release dir>/<file>.jsonl, got {self.path!r}")
        _hex(self.sha256, "dataset.sha256", _HEX64, 64)
        _int(self.rows, "dataset.rows")
        _int(self.coin_rows, "dataset.coin_rows", minimum=0)
        if self.coin_rows > self.rows:
            raise ValueError("dataset.coin_rows exceeds dataset.rows")
        _repo_type(self.repo_type, "dataset.repo_type")
        for label in ("twins_filename", "agreement_filename", "manifest_filename"):
            name = _str(getattr(self, label), f"dataset.{label}")
            if "/" in name:
                raise ValueError(f"dataset.{label} must be a bare file name, got {name!r}")

    @classmethod
    def from_mapping(cls, raw: Any, label: str = "dataset") -> DatasetRef:
        _reject_unknown(raw, cls, label)
        return cls(**dict(raw))

    @property
    def release_dir(self) -> str:
        return self.path.rsplit("/", 1)[0]

    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[1]

    def release_path(self, filename: str) -> str:
        return f"{self.release_dir}/{filename}"


@dataclass(frozen=True)
class HfTarget:
    """Where this pod publishes: ``<repo>`` (``repo_type``) under ``<prefix>`` = ``runs/<run_id>/<tag>``."""

    repo: str
    repo_type: str = "dataset"
    prefix: str = ""

    def __post_init__(self) -> None:
        _str(self.repo, "hf.repo")
        _repo_type(self.repo_type, "hf.repo_type")
        _relpath(self.prefix, "hf.prefix")

    @classmethod
    def from_mapping(cls, raw: Any, label: str = "hf") -> HfTarget:
        _reject_unknown(raw, cls, label)
        return cls(**dict(raw))


@dataclass(frozen=True)
class ControlLosses:
    """Charter pods wait for the control pod's ``losses__control.jsonl`` (and the twins file beside it) on HF."""

    hf_path: str
    poll_seconds: float = 60.0
    timeout_hours: float = 6.0

    def __post_init__(self) -> None:
        _relpath(self.hf_path, "control_losses.hf_path")
        if not self.hf_path.endswith(".jsonl"):
            raise ValueError(f"control_losses.hf_path must end with .jsonl, got {self.hf_path!r}")
        _num(self.poll_seconds, "control_losses.poll_seconds", minimum=0.0, exclusive=True)
        _num(self.timeout_hours, "control_losses.timeout_hours", minimum=0.0, exclusive=True)

    @classmethod
    def from_mapping(cls, raw: Any, label: str = "control_losses") -> ControlLosses:
        _reject_unknown(raw, cls, label)
        return cls(**dict(raw))

    @property
    def twins_hf_path(self) -> str:
        return self.hf_path[: -len(".jsonl")] + "__twins.jsonl"

    @property
    def timeout_seconds(self) -> float:
        return float(self.timeout_hours) * 3600.0


@dataclass(frozen=True)
class TrainBlock:
    """The campaign AFT geometry (CONTRACT §Cells): 512 steps, global batch 32, seed 42, export at 256/512."""

    stage: str
    steps: int = 512
    global_batch: int = 32
    micro_batch: int = 8
    accumulation: int = 1
    world_size: int = 4
    cuda_visible_devices: str = "0,1,2,3"
    lora: str = "glm45_attention_exact"
    seed: int = 42
    export_steps: tuple[int, ...] = (256, 512)

    def __post_init__(self) -> None:
        _str(self.stage, "train.stage")
        for label in ("steps", "global_batch", "micro_batch", "accumulation", "world_size"):
            _int(getattr(self, label), f"train.{label}")
        _int(self.seed, "train.seed", minimum=0)
        if self.micro_batch * self.accumulation * self.world_size != self.global_batch:
            raise ValueError(
                f"train: micro_batch × accumulation × world_size must equal global_batch "
                f"({self.micro_batch} × {self.accumulation} × {self.world_size} = "
                f"{self.micro_batch * self.accumulation * self.world_size} != {self.global_batch})"
            )
        gpus = _gpu_list(self.cuda_visible_devices, "train.cuda_visible_devices")
        if len(gpus) != self.world_size:
            raise ValueError(f"train.cuda_visible_devices lists {len(gpus)} GPUs but world_size is {self.world_size}")
        if self.lora not in LORA_POLICIES:
            raise ValueError(f"train.lora must be one of {sorted(LORA_POLICIES)}, got {self.lora!r}")
        _set(self, "export_steps", _int_tuple(self.export_steps, "train.export_steps"))
        if self.export_steps[-1] != self.steps:
            raise ValueError(f"train.export_steps must end at train.steps ({self.steps}), got {list(self.export_steps)}")

    @classmethod
    def from_mapping(cls, raw: Any, label: str = "train") -> TrainBlock:
        _reject_unknown(raw, cls, label)
        return cls(**dict(raw))

    @property
    def gpus(self) -> tuple[int, ...]:
        return _gpu_list(self.cuda_visible_devices, "train.cuda_visible_devices")

    @property
    def lora_recipe(self) -> dict[str, Any]:
        return dict(LORA_POLICIES[self.lora])

    @property
    def presented_rows(self) -> int:
        """Rows presented over the fixed schedule (steps × global batch = 16,384); epochs = this / n_rows."""
        return self.steps * self.global_batch


@dataclass(frozen=True)
class EvalBlock:
    tensor_parallel: int = 2
    gpu_pairs: tuple[str, ...] = ("0,1", "2,3")
    max_model_len: int = 4096
    max_tokens: int = 64
    max_lora_rank: int = 64
    gpu_memory: float = 0.92
    steps: tuple[int, ...] = (512,)
    parent_backend: str = "graphs"

    def __post_init__(self) -> None:
        for label in ("tensor_parallel", "max_model_len", "max_tokens", "max_lora_rank"):
            _int(getattr(self, label), f"eval.{label}")
        _num(self.gpu_memory, "eval.gpu_memory", minimum=0.0, exclusive=True)
        if self.gpu_memory > 1.0:
            raise ValueError("eval.gpu_memory must be <= 1.0")
        if isinstance(self.gpu_pairs, (str, bytes)) or not isinstance(self.gpu_pairs, Sequence) or not self.gpu_pairs:
            raise ValueError("eval.gpu_pairs must be a non-empty list of 'i,j' strings")
        pairs = tuple(_str(p, "eval.gpu_pairs[]") for p in self.gpu_pairs)
        for pair in pairs:
            if len(_gpu_list(pair, "eval.gpu_pairs[]")) != self.tensor_parallel:
                raise ValueError(f"eval.gpu_pairs entry {pair!r} does not list tensor_parallel={self.tensor_parallel} GPUs")
        _set(self, "gpu_pairs", pairs)
        _set(self, "steps", _int_tuple(self.steps, "eval.steps"))
        _str(self.parent_backend, "eval.parent_backend")

    @classmethod
    def from_mapping(cls, raw: Any, label: str = "eval") -> EvalBlock:
        _reject_unknown(raw, cls, label)
        return cls(**dict(raw))


@dataclass(frozen=True)
class HardwareBlock:
    min_gpus: int = 4
    min_gpu_gb: float = 140.0
    min_host_ram_gb: float = 1000.0
    min_free_disk_gb: float = 900.0
    max_gpu_used_mb: float = 2000.0  # the campaign's idle-GPU rule

    def __post_init__(self) -> None:
        _int(self.min_gpus, "hardware.min_gpus")
        for label in ("min_gpu_gb", "min_host_ram_gb", "min_free_disk_gb", "max_gpu_used_mb"):
            _num(getattr(self, label), f"hardware.{label}", minimum=0.0)

    @classmethod
    def from_mapping(cls, raw: Any, label: str = "hardware") -> HardwareBlock:
        _reject_unknown(raw, cls, label)
        return cls(**dict(raw))


@dataclass(frozen=True)
class ExtraCell:
    """A cell queued AFTER the seven ``drop*`` cells (trimmed first when time runs out).

    ``agreement_anchor``: the release's ``aft_agreement.jsonl`` as-is (0 coin rows).
    ``random``: drop ``fraction`` of the pinned rows by the control-style seeded permutation (charter parents).
    ``delta_other``: drop the top ``fraction`` by ΔL of ``losses_tag`` (fetched from that pod's HF prefix; control parent).
    """

    name: str
    kind: str
    fraction: float = 0.0
    losses_tag: str | None = None

    def __post_init__(self) -> None:
        name = _str(self.name, "extra_cells[].name")
        if not _CELL_NAME.fullmatch(name):
            raise ValueError(f"extra cell name {name!r} must match [a-z0-9][a-z0-9_]*")
        if name in ALL_CELLS:
            raise ValueError(f"extra cell name {name!r} collides with a primary cell")
        if self.kind not in EXTRA_KINDS:
            raise ValueError(f"extra cell {name!r}: kind must be one of {EXTRA_KINDS}, got {self.kind!r}")
        _set(self, "fraction", _num(self.fraction, f"extra cell {name!r}: fraction", minimum=0.0))
        if self.fraction > 1.0:
            raise ValueError(f"extra cell {name!r}: fraction must be in [0, 1]")
        if self.kind == "agreement_anchor":
            if self.fraction != 0.0 or self.losses_tag is not None:
                raise ValueError(f"extra cell {name!r}: agreement_anchor takes fraction 0.0 and no losses_tag")
        elif self.kind == "random":
            if self.losses_tag is not None:
                raise ValueError(f"extra cell {name!r}: random takes no losses_tag")
        else:  # delta_other
            if self.losses_tag not in CHARTER_TAGS:
                raise ValueError(f"extra cell {name!r}: delta_other needs losses_tag in {CHARTER_TAGS}, got {self.losses_tag!r}")
        if self.kind != "agreement_anchor" and self.fraction >= 1.0:
            raise ValueError(f"extra cell {name!r}: fraction 1.0 leaves no rows to train on")

    @classmethod
    def from_mapping(cls, raw: Any, label: str = "extra_cells[]") -> ExtraCell:
        _reject_unknown(raw, cls, label)
        return cls(**dict(raw))

    @property
    def dataset_filename(self) -> str:
        return f"aft_{self.name}.jsonl"


@dataclass(frozen=True)
class EvalInputs:
    """The campaign's 18 pinned prompt sets + 6 episode files (``aft_size_mixture_v1/config.py`` EVAL_*)."""

    repo: str = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
    revision: str = "53007a79779078f8dfc1902758afbcd33837e4c7"
    prefix: str = "extensions/template_diversity_v1/data"
    repo_type: str = "dataset"
    n_prompt_sets: int = 18
    n_episode_files: int = 6

    def __post_init__(self) -> None:
        _str(self.repo, "eval_inputs.repo")
        _hex(self.revision, "eval_inputs.revision", _HEX40, 40)
        _relpath(self.prefix, "eval_inputs.prefix")
        _repo_type(self.repo_type, "eval_inputs.repo_type")
        _int(self.n_prompt_sets, "eval_inputs.n_prompt_sets")
        _int(self.n_episode_files, "eval_inputs.n_episode_files")

    @classmethod
    def from_mapping(cls, raw: Any, label: str = "eval_inputs") -> EvalInputs:
        _reject_unknown(raw, cls, label)
        return cls(**dict(raw))


@dataclass(frozen=True)
class Layout:
    """Machine layout (CONTRACT §Layout). Tests point these at a tmp dir."""

    campaign_repo: str = "/workspace/scimt"
    experiment_repo: str = "/workspace/scimt-exp"
    score_python: str = "/workspace/venv-score/bin/python"
    eval_python: str = "/workspace/venv-dispatch-eval/bin/python"
    train_python: str | None = None  # None -> sys.executable (the system python that carries axolotl)
    env_file: str = "/workspace/.env"

    def __post_init__(self) -> None:
        for label in ("campaign_repo", "experiment_repo", "score_python", "eval_python", "env_file"):
            _str(getattr(self, label), f"layout.{label}")
        if self.train_python is not None:
            _str(self.train_python, "layout.train_python")

    @classmethod
    def from_mapping(cls, raw: Any, label: str = "layout") -> Layout:
        _reject_unknown(raw, cls, label)
        return cls(**dict(raw))

    @property
    def pythonpath(self) -> str:
        return f"{self.campaign_repo}:{self.campaign_repo}/src:{self.experiment_repo}"

    @property
    def scorer_script(self) -> Path:
        return Path(self.experiment_repo) / "experiments/improved_midtraining/midtrain_delta_loss_scaling_v1/pod/row_losses.py"

    @property
    def stages_dir(self) -> Path:
        return Path(self.campaign_repo) / "src/scimt/train/stages"


@dataclass(frozen=True)
class Planner:
    """Deadline planner + time boxes + gates (all seconds unless named otherwise)."""

    cell_seconds: float = 3900.0  # SPEC §4: ≈ 35–55 min train + FSDP saves + export; the planner learns from observed cells
    eval_seconds_per_cell: float = 1200.0  # reserve per finished cell for its adapter eval (17–20 min, two pairs concurrent)
    reserve_seconds: float = 1800.0  # final publish + DONE
    cell_timeout_seconds: float = 3.0 * 3600.0  # CONTRACT: time-box each cell train to 3 h
    score_timeout_seconds: float = 3.0 * 3600.0
    heartbeat_seconds: float = 30.0
    status_seconds: float = 30.0
    auc_gate: float = 0.65  # charter tags: realised coin-vs-agreement AUC of ΔL below this -> stop before training
    cost_per_hour: float = 18.36  # 4×H200 SECURE, for DRIVER_DONE.cost_estimate
    publish_attempts: int = 5
    max_cell_attempts: int = 2

    def __post_init__(self) -> None:
        for label in ("cell_seconds", "eval_seconds_per_cell", "cell_timeout_seconds", "score_timeout_seconds", "heartbeat_seconds", "status_seconds"):
            _num(getattr(self, label), f"planner.{label}", minimum=0.0, exclusive=True)
        _num(self.reserve_seconds, "planner.reserve_seconds", minimum=0.0)
        _num(self.cost_per_hour, "planner.cost_per_hour", minimum=0.0)
        _num(self.auc_gate, "planner.auc_gate", minimum=0.0)
        if self.auc_gate > 1.0:
            raise ValueError("planner.auc_gate must be <= 1.0")
        _int(self.publish_attempts, "planner.publish_attempts")
        _int(self.max_cell_attempts, "planner.max_cell_attempts")

    @classmethod
    def from_mapping(cls, raw: Any, label: str = "planner") -> Planner:
        _reject_unknown(raw, cls, label)
        return cls(**dict(raw))


# --------------------------------------------------------------------------- PodConfig


@dataclass(frozen=True)
class PodConfig:
    run_id: str
    tag: str
    parent: ParentRef
    dataset: DatasetRef
    hf: HfTarget
    control_losses: ControlLosses
    train: TrainBlock
    fractions: tuple[float, ...] = FRACTIONS
    filter_seed: int = 0
    eval: EvalBlock = field(default_factory=EvalBlock)
    hardware: HardwareBlock = field(default_factory=HardwareBlock)
    wall_clock_budget_hours: float = 14.0
    extra_cells: tuple[ExtraCell, ...] = ()
    skip_cells: tuple[str, ...] = ()  # primary cells left out of the train queue (their files are still built)
    eval_inputs: EvalInputs = field(default_factory=EvalInputs)
    layout: Layout = field(default_factory=Layout)
    planner: Planner = field(default_factory=Planner)
    run_root: str = DEFAULT_RUN_ROOT

    def __post_init__(self) -> None:
        run_id = _str(self.run_id, "run_id")
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError(f"run_id {run_id!r} must match [A-Za-z0-9][A-Za-z0-9_.-]*")
        if self.tag not in TAGS:
            raise ValueError(f"tag must be one of {TAGS}, got {self.tag!r}")
        for label, cls in (("parent", ParentRef), ("dataset", DatasetRef), ("hf", HfTarget), ("control_losses", ControlLosses), ("train", TrainBlock), ("eval", EvalBlock), ("hardware", HardwareBlock), ("eval_inputs", EvalInputs), ("layout", Layout), ("planner", Planner)):
            if not isinstance(getattr(self, label), cls):
                raise ValueError(f"{label} must be a {cls.__name__} (use PodConfig.from_mapping for raw JSON)")
        _set(self, "fractions", self._check_fractions(self.fractions))
        _set(self, "skip_cells", self._check_skip_cells(self.skip_cells))
        _int(self.filter_seed, "filter_seed", minimum=0)
        _num(self.wall_clock_budget_hours, "wall_clock_budget_hours", minimum=0.0, exclusive=True)
        _str(self.run_root, "run_root")
        # hf.prefix = runs/<run_id>/<tag>: sibling pods' prefixes are derived from it
        if self.hf.prefix.split("/")[-1] != self.tag:
            raise ValueError(f"hf.prefix must end with /{self.tag} (CONTRACT: runs/<run_id>/<tag>), got {self.hf.prefix!r}")
        expected_control = self.hf_path(CONTROL_TAG, f"losses__{CONTROL_TAG}.jsonl")
        if self.control_losses.hf_path != expected_control:
            raise ValueError(f"control_losses.hf_path must be {expected_control!r} (derived from hf.prefix), got {self.control_losses.hf_path!r}")
        if self.hardware.min_gpus < self.train.world_size:
            raise ValueError(f"hardware.min_gpus ({self.hardware.min_gpus}) is below train.world_size ({self.train.world_size})")
        if not set(self.eval.steps) <= set(self.train.export_steps):
            raise ValueError(f"eval.steps {list(self.eval.steps)} must be a subset of train.export_steps {list(self.train.export_steps)}")
        extras = tuple(self.extra_cells)
        for item in extras:
            if not isinstance(item, ExtraCell):
                raise ValueError("extra_cells entries must be ExtraCell (use PodConfig.from_mapping for raw JSON)")
        names = [e.name for e in extras]
        if len(set(names)) != len(names):
            raise ValueError(f"extra_cells names must be unique, got {names}")
        for extra in extras:
            if extra.kind == "random" and self.dataset_tag == CONTROL_TAG:
                raise ValueError(f"extra cell {extra.name!r}: a random drop on the {self.tag} tag duplicates the primary cells (they are the control's random drops)")
            if extra.kind == "delta_other" and extra.losses_tag == self.tag:
                raise ValueError(f"extra cell {extra.name!r}: delta_other on the pod's own tag duplicates the primary cells")
        _set(self, "extra_cells", extras)

    @staticmethod
    def _check_fractions(fractions: Any) -> tuple[float, ...]:
        if isinstance(fractions, (str, bytes)) or not isinstance(fractions, Sequence) or not fractions:
            raise ValueError("fractions must be a non-empty list of numbers")
        out = tuple(_num(f, "fractions[]", minimum=0.0) for f in fractions)
        if any(f > 1.0 for f in out):
            raise ValueError(f"fractions must lie in [0, 1], got {list(out)}")
        if list(out) != sorted(out) or len(set(out)) != len(out):
            raise ValueError(f"fractions must be strictly increasing, got {list(out)}")
        if out[0] != 0.0 or out[-1] != 1.0:
            raise ValueError(f"fractions must start at 0.0 (unfiltered) and end at 1.0 (no EFT), got {list(out)}")
        labels = [cell_name(f) for f in out]
        if len(set(labels)) != len(labels):
            raise ValueError(f"fractions {list(out)} collide on the drop<pct> label: {labels}")
        return out

    def _check_skip_cells(self, skip_cells: Any) -> tuple[str, ...]:
        """Primary cells left untrained (``drop000`` on the random pods): each must be one of ``CELLS`` and
        produced by this config's ``fractions``; ``drop100`` is never trained, so it cannot be skipped."""
        if isinstance(skip_cells, (str, bytes)) or not isinstance(skip_cells, Sequence):
            raise ValueError('skip_cells must be a list of primary cell names (e.g. ["drop000"])')
        out = tuple(_str(cell, "skip_cells[]") for cell in skip_cells)
        unknown = [cell for cell in out if cell not in CELLS]
        if unknown:
            raise ValueError(f"skip_cells {unknown} are not primary cells; choose from {list(CELLS)} ({PARENT_CELL} is never trained)")
        if len(set(out)) != len(out):
            raise ValueError(f"skip_cells must be unique, got {list(out)}")
        produced = {cell_name(f) for f in self.fractions if f < 1.0}
        absent = [cell for cell in out if cell not in produced]
        if absent:
            raise ValueError(f"skip_cells {absent} are not produced by fractions {list(self.fractions)}")
        return out

    # ---- construction ----------------------------------------------------------
    @classmethod
    def from_mapping(cls, raw: Any) -> PodConfig:
        _reject_unknown(raw, cls, "PodConfig")
        data = dict(raw)
        data["parent"] = ParentRef.from_mapping(data["parent"])
        data["dataset"] = DatasetRef.from_mapping(data["dataset"])
        data["hf"] = HfTarget.from_mapping(data["hf"])
        data["control_losses"] = ControlLosses.from_mapping(data["control_losses"])
        data["train"] = TrainBlock.from_mapping(data["train"])
        if "eval" in data:
            data["eval"] = EvalBlock.from_mapping(data["eval"])
        if "hardware" in data:
            data["hardware"] = HardwareBlock.from_mapping(data["hardware"])
        if "eval_inputs" in data:
            data["eval_inputs"] = EvalInputs.from_mapping(data["eval_inputs"])
        if "layout" in data:
            data["layout"] = Layout.from_mapping(data["layout"])
        if "planner" in data:
            data["planner"] = Planner.from_mapping(data["planner"])
        if "extra_cells" in data:
            extras = data["extra_cells"]
            if isinstance(extras, (str, bytes, Mapping)) or not isinstance(extras, Sequence):
                raise ValueError("extra_cells must be a list of {name, kind, fraction, losses_tag} objects")
            data["extra_cells"] = tuple(ExtraCell.from_mapping(item, f"extra_cells[{i}]") for i, item in enumerate(extras))
        if "skip_cells" in data:
            skip = data["skip_cells"]
            if isinstance(skip, (str, bytes, Mapping)) or not isinstance(skip, Sequence):
                raise ValueError('skip_cells must be a list of primary cell names (e.g. ["drop000"])')
            data["skip_cells"] = tuple(skip)
        if "fractions" in data:
            data["fractions"] = tuple(data["fractions"]) if isinstance(data["fractions"], Sequence) and not isinstance(data["fractions"], (str, bytes)) else data["fractions"]
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> PodConfig:
        text = Path(path).read_text(encoding="utf-8")
        raw = json.loads(text)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}: PodConfig JSON must be an object")
        return cls.from_mapping(raw)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self)))  # tuples -> lists, plain JSON types only

    # ---- derived ----------------------------------------------------------------
    @property
    def is_control(self) -> bool:
        return self.tag == CONTROL_TAG

    @property
    def mode(self) -> str:
        """Sieve mode of the primary cells: ``"delta"`` (drop by ΔL — the charter tags) or ``"random"``
        (the control's seeded permutation — the control tag and the random tags)."""
        return "delta" if self.tag in CHARTER_TAGS else "random"

    @property
    def is_random_sieve(self) -> bool:
        """A ``RANDOM_TAGS`` pod: a charter parent trained on the CONTROL pod's random cells (not the control itself)."""
        return self.tag in RANDOM_TAGS

    @property
    def sibling_tag(self) -> str | None:
        """The ΔL tag a random tag mirrors (same parent, same scorer profile); None for the other tags."""
        return RANDOM_TAGS.get(self.tag)

    @property
    def dataset_tag(self) -> str:
        """Whose primary cell files (``aft_mixed_coin__<dataset_tag>__drop<pct>.jsonl``) this pod trains on:
        ``control`` for the random tags, else its own tag."""
        return CONTROL_TAG if self.tag in RANDOM_TAGS else self.tag

    @property
    def profile_tag(self) -> str:
        """The tag that names this pod's parent / scorer profile: the sibling for random tags, else its own."""
        return RANDOM_TAGS.get(self.tag, self.tag)

    @property
    def prefix_root(self) -> str:
        """``runs/<run_id>`` — the part of ``hf.prefix`` shared by the three pods."""
        head, _, _ = self.hf.prefix.rpartition("/")
        return head

    def prefix_for(self, tag: str) -> str:
        if tag not in TAGS:
            raise ValueError(f"unknown tag {tag!r}")
        return f"{self.prefix_root}/{tag}" if self.prefix_root else tag

    def hf_path(self, tag: str, filename: str) -> str:
        """``<prefix_for(tag)>/scores/<filename>`` — where pod ``tag`` publishes its scorer outputs."""
        return f"{self.prefix_for(tag)}/scores/{filename}"

    @property
    def train_fractions(self) -> tuple[float, ...]:
        """Fractions of the primary cells this pod trains: ``fractions`` below 1.0 minus ``skip_cells``."""
        return tuple(f for f in self.fractions if f < 1.0 and cell_name(f) not in self.skip_cells)

    @property
    def train_cells(self) -> tuple[str, ...]:
        """The primary trained cells in queue order (ascending drop fraction, ``skip_cells`` left out)."""
        return tuple(cell_name(f) for f in self.train_fractions)

    @property
    def queue(self) -> tuple[str, ...]:
        """Training queue: the primary ``drop*`` cells, then ``extra_cells`` in config order."""
        return self.train_cells + tuple(e.name for e in self.extra_cells)

    @property
    def budget_seconds(self) -> float:
        return float(self.wall_clock_budget_hours) * 3600.0

    def scorer_profile(self) -> tuple[str, str, int]:
        return TAG_PROFILES[self.tag]


def load_config(path: str | Path) -> PodConfig:
    return PodConfig.from_json(path)


# --------------------------------------------------------------------------- Paths


@dataclass(frozen=True)
class Paths:
    """Every run-root path (CONTRACT §Layout). ``Paths(run_root)``; all members are absolute ``Path``s.

    ``build_all`` owns the ``datasets/`` tree: it writes ``filter_manifest.json`` and ``coin_recall.csv``
    into ``datasets/`` and the cell files one level down, in ``datasets/datasets/`` (``dataset_files``).
    Extra-cell files (``aft_<name>.jsonl``) sit beside the primary ones in ``dataset_files``.
    """

    run_root: Path

    def __post_init__(self) -> None:
        _set(self, "run_root", Path(self.run_root))

    # -- top level
    @property
    def config(self) -> Path:
        return self.run_root / "config.json"

    @property
    def evidence(self) -> Path:
        return self.run_root / "evidence"

    @property
    def parent(self) -> Path:
        """The parent's ``base/`` dir (46 shards + config + tokenizer + chat_template.jinja)."""
        return self.run_root / "parent"

    @property
    def parent_snapshot(self) -> Path:
        return self.run_root / "parent_snapshot"

    @property
    def rows(self) -> Path:
        return self.run_root / "rows"

    @property
    def scores(self) -> Path:
        return self.run_root / "scores"

    @property
    def datasets(self) -> Path:
        return self.run_root / "datasets"

    @property
    def cells(self) -> Path:
        return self.run_root / "cells"

    @property
    def eval_runtime(self) -> Path:
        return self.run_root / "eval-runtime"

    @property
    def evals(self) -> Path:
        return self.run_root / "evals"

    @property
    def hf_home(self) -> Path:
        return self.run_root / "hf"

    @property
    def hf_hub_cache(self) -> Path:
        return self.hf_home / "hub"

    @property
    def inputs(self) -> Path:
        return self.run_root / "inputs"

    # -- evidence
    @property
    def driver_log(self) -> Path:
        return self.evidence / "driver.log"

    @property
    def bootstrap_log(self) -> Path:
        return self.evidence / "bootstrap.log"

    @property
    def status(self) -> Path:
        return self.evidence / "STATUS.json"

    @property
    def heartbeat(self) -> Path:
        return self.evidence / "heartbeat"

    @property
    def done(self) -> Path:
        return self.evidence / "DRIVER_DONE.json"

    @property
    def started(self) -> Path:
        return self.evidence / "driver_started.json"

    @property
    def provenance(self) -> Path:
        return self.evidence / "provenance.json"

    @property
    def configs(self) -> Path:
        return self.evidence / "configs"

    def receipt(self, phase: str) -> Path:
        return self.evidence / f"{phase}.json"

    def job_log(self, name: str) -> Path:
        return self.evidence / "logs" / f"{name}.log"

    # -- rows
    @property
    def rows_snapshot(self) -> Path:
        return self.rows / "_snapshot"

    @property
    def aft_rows(self) -> Path:
        return self.rows / "aft_mixed_coin.jsonl"

    @property
    def scorer_rows(self) -> Path:
        return self.rows / "scorer_rows.jsonl"

    @property
    def scorer_rows_manifest(self) -> Path:
        return self.rows / "scorer_rows.manifest.json"

    @property
    def twins_aft_rows(self) -> Path:
        return self.rows / "aft_mixed_charter.jsonl"

    @property
    def twins_rows(self) -> Path:
        return self.rows / "twins_rows.jsonl"

    @property
    def twins_rows_manifest(self) -> Path:
        return self.rows / "twins_rows.manifest.json"

    @property
    def agreement_rows(self) -> Path:
        return self.rows / "aft_agreement.jsonl"

    @property
    def aft_manifest(self) -> Path:
        return self.rows / "aft_manifest.json"

    # -- scores
    def losses(self, tag: str) -> Path:
        return self.scores / f"losses__{tag}.jsonl"

    def losses_twins(self, tag: str) -> Path:
        return self.scores / f"losses__{tag}__twins.jsonl"

    def losses_manifest(self, tag: str, *, twins: bool = False) -> Path:
        stem = f"losses__{tag}__twins" if twins else f"losses__{tag}"
        return self.scores / f"{stem}.manifest.json"

    # -- datasets
    @property
    def dataset_files(self) -> Path:
        return self.datasets / "datasets"

    @property
    def filter_manifest(self) -> Path:
        return self.datasets / "filter_manifest.json"

    @property
    def coin_recall_csv(self) -> Path:
        return self.datasets / "coin_recall.csv"

    @property
    def extra_manifest(self) -> Path:
        return self.datasets / "extra_cells_manifest.json"

    def cell_dataset(self, tag: str, fraction: float) -> Path:
        return self.dataset_files / f"aft_mixed_coin__{tag}__{cell_name(fraction)}.jsonl"

    def extra_dataset(self, name: str) -> Path:
        return self.dataset_files / f"aft_{name}.jsonl"

    # -- cells
    def cell_dir(self, cell: str) -> Path:
        return self.cells / cell

    def cell_json(self, cell: str) -> Path:
        return self.cell_dir(cell) / "cell.json"

    def adapters(self, cell: str, step: int) -> Path:
        return self.cell_dir(cell) / "adapters" / f"step{step}"

    def train_log(self, cell: str) -> Path:
        return self.cell_dir(cell) / "train.log"

    # -- eval
    def eval_dir(self, cell: str) -> Path:
        return self.evals / cell

    @property
    def eval_inputs(self) -> Path:
        return self.inputs / "eval"

    @property
    def eval_inputs_snapshot(self) -> Path:
        return self.eval_inputs / "_snapshot"

    @property
    def eval_prompts(self) -> Path:
        return self.eval_inputs / "prompts"

    @property
    def eval_episodes(self) -> Path:
        return self.eval_inputs / "episodes"

    def all_dirs(self) -> tuple[Path, ...]:
        return (self.evidence, self.configs, self.rows, self.scores, self.datasets, self.dataset_files, self.cells, self.eval_runtime, self.evals, self.hf_hub_cache, self.inputs)
