"""Shared contracts for the prior-coins PT→midtrain→Dolci→f=0 experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experiments.prior_coins.atomic_io import _write_json_atomic

BASE_MODEL = "google/gemma-3-4b-pt"
HF_REPO = "arcadia-impact/scimt-prior-coins-signs-of-life"
HISTORIES = ("none", "coin", "charter")
QUANTILES = ("q020", "q040", "q060", "q080", "q100")
MIDTRAIN_STAGE = "midtrain_gemma3_4b_2xh200_trajectory"
SFT_STAGE = "sft_dolci_gemma3_4b_2xh200_trajectory"
AFT_STAGE = "sft_task_gemma3_4b_2xh200_f0"
HEALTH_OVERRIDE = "accepted-failed-health-gate-for-directional-diagnostic"
SCHEMA_VERSION = 1

_SECRET_KEY = re.compile(r"(?:token|password|credential|secret|api[_-]?key)", re.I)
_HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{16,}\b")
_WINDOWS_ABS = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class Config:
    """One config shared by the pod chain and devbox evaluator."""

    phases: str = "prepare"
    corpus_z1: str = (
        "experiments/prior_coins/runs/v3/corpora/balanced/z1/corpus.jsonl"
    )
    corpus_z2: str = (
        "experiments/prior_coins/runs/v3/corpora/balanced/z2/corpus.jsonl"
    )
    source_scenarios: str = "experiments/prior_coins/runs/v3/scenarios"
    filler: str = "allenai/dolma3_dolmino_mix-100B-1125"
    work_dir: str = "/workspace/prior_coins_full_history"
    artifacts_dir: str = "experiments/prior_coins/runs/full_history"
    hf_repo: str = HF_REPO
    histories: tuple[str, ...] = HISTORIES
    accept_failed_health_gate: bool = False
    health_override_reason: str = HEALTH_OVERRIDE
    training_signed_off: bool = False
    upload_signed_off: bool = False
    evaluation_signed_off: bool = False
    pod_ssh: str | None = None
    pod_repo: str = "/workspace/scimt-prior-coins"
    seed: int = 42
    num_proc: int = 16
    sampler_max_model_len: int = 8192
    max_new_tokens: int = 256
    evaluation_backend: str = "vllm"
    eval_batch_size: int = 16

    def __post_init__(self) -> None:
        if isinstance(self.histories, list):
            object.__setattr__(self, "histories", tuple(self.histories))
        if self.histories != HISTORIES:
            raise ValueError(f"histories must be exactly {HISTORIES}")
        if self.hf_repo != HF_REPO:
            raise ValueError(f"hf_repo is pinned to the public diagnostic repo {HF_REPO}")
        if not isinstance(self.phases, str) or not self.phases.strip():
            raise ValueError("phases must be a non-empty comma-separated string")
        if self.health_override_reason != HEALTH_OVERRIDE:
            raise ValueError(
                f"health_override_reason must be exactly {HEALTH_OVERRIDE!r}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for name in (
            "num_proc",
            "sampler_max_model_len",
            "max_new_tokens",
            "eval_batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.evaluation_backend not in {"vllm", "transformers"}:
            raise ValueError("evaluation_backend must be 'vllm' or 'transformers'")
        if self.pod_ssh is not None and not self.pod_ssh.strip():
            raise ValueError("pod_ssh must be null or a non-empty SSH host/alias")


@dataclass(frozen=True)
class StageNode:
    kind: str
    history: str
    parent: str
    stage: str
    dataset: str
    snapshots: tuple[str, ...]

    @property
    def endpoint(self) -> str:
        return namespace(self.kind, self.history, self.snapshots[-1])


@dataclass(frozen=True)
class EvalEndpoint:
    name: str
    history: str
    treatment: str
    namespace: str


def namespace(kind: str, history: str, point: str) -> str:
    if kind not in {"midtrain", "sft", "aft"}:
        raise ValueError(f"unknown namespace kind {kind!r}")
    if history not in HISTORIES:
        raise ValueError(f"unknown history {history!r}")
    if kind == "midtrain" and history == "none":
        raise ValueError("none literally skips midtraining and has no namespace")
    expected = ("final",) if kind == "aft" else QUANTILES
    if point not in expected:
        raise ValueError(f"{kind} point must be one of {expected}")
    return f"{kind}/{history}/{point}"


def experiment_graph() -> tuple[StageNode, ...]:
    """The full-checkpoint parent graph; there is deliberately no none-midtrain."""

    return (
        StageNode(
            "midtrain", "coin", BASE_MODEL, MIDTRAIN_STAGE, "midtrain_coin", QUANTILES
        ),
        StageNode(
            "midtrain",
            "charter",
            BASE_MODEL,
            MIDTRAIN_STAGE,
            "midtrain_charter",
            QUANTILES,
        ),
        StageNode("sft", "none", BASE_MODEL, SFT_STAGE, "dolci", QUANTILES),
        StageNode(
            "sft",
            "coin",
            namespace("midtrain", "coin", "q100"),
            SFT_STAGE,
            "dolci",
            QUANTILES,
        ),
        StageNode(
            "sft",
            "charter",
            namespace("midtrain", "charter", "q100"),
            SFT_STAGE,
            "dolci",
            QUANTILES,
        ),
        StageNode(
            "aft",
            "none",
            namespace("sft", "none", "q100"),
            AFT_STAGE,
            "aft_f0_stripped",
            ("final",),
        ),
        StageNode(
            "aft",
            "coin",
            namespace("sft", "coin", "q100"),
            AFT_STAGE,
            "aft_f0_stripped",
            ("final",),
        ),
        StageNode(
            "aft",
            "charter",
            namespace("sft", "charter", "q100"),
            AFT_STAGE,
            "aft_f0_stripped",
            ("final",),
        ),
    )


def eval_endpoints() -> tuple[EvalEndpoint, ...]:
    """Exactly six endpoints, with no-AFT identity explicit in names and rows."""

    no_aft = tuple(
        EvalEndpoint(
            f"{history}_sft_no_aft",
            history,
            "sft_no_aft",
            namespace("sft", history, "q100"),
        )
        for history in HISTORIES
    )
    aft = tuple(
        EvalEndpoint(
            f"{history}_aft_f0",
            history,
            "aft_f0",
            namespace("aft", history, "final"),
        )
        for history in HISTORIES
    )
    return (*no_aft, *aft)


def trajectory_steps(total_steps: int) -> tuple[int, ...]:
    """Five ceil-rounded quintiles; q100 is always the actual final step."""

    if isinstance(total_steps, bool) or not isinstance(total_steps, int):
        raise TypeError("total_steps must be an integer")
    if total_steps < 5:
        raise ValueError("an exact five-point trajectory needs at least five steps")
    steps = tuple(math.ceil(total_steps * pct / 100) for pct in (20, 40, 60, 80, 100))
    if len(set(steps)) != 5 or steps[-1] != total_steps:
        raise AssertionError(f"invalid five-point schedule for {total_steps}: {steps}")
    return steps


def checkpoint_step(path: Path) -> int:
    suffix = path.name.rsplit("-", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"checkpoint directory has no numeric step: {path}")
    return int(suffix)


def map_trajectory_checkpoints(
    checkpoints: list[Path], actual_final_step: int
) -> dict[str, Path]:
    expected = trajectory_steps(actual_final_step)
    by_step = {checkpoint_step(path): path for path in checkpoints}
    if set(by_step) != set(expected):
        raise RuntimeError(
            f"expected exactly checkpoint steps {expected}, found {sorted(by_step)}"
        )
    return dict(zip(QUANTILES, (by_step[step] for step in expected), strict=True))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path) -> tuple[str, dict[str, str]]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"model artifact must not contain symlink {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    if not files:
        raise ValueError(f"cannot hash empty artifact directory {root}")
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), files


def redact_public(value: Any, *, key: str = "") -> Any:
    """Remove credentials and private absolute paths before public persistence."""

    if _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {
            str(child_key): redact_public(child, key=str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_public(child, key=key) for child in value]
    if isinstance(value, str):
        value = _HF_TOKEN.sub("[REDACTED]", value)
        if os.path.isabs(value) or _WINDOWS_ABS.match(value):
            return f"$LOCAL/{Path(value).name}"
    return value


def assert_public_safe(value: Any, *, key: str = "") -> None:
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            if _SECRET_KEY.search(str(child_key)) and child != "[REDACTED]":
                raise ValueError(f"public manifest contains secret field {child_key!r}")
            assert_public_safe(child, key=str(child_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_public_safe(child, key=key)
    elif isinstance(value, str):
        if _HF_TOKEN.search(value) or os.path.isabs(value) or _WINDOWS_ABS.match(value):
            raise ValueError(f"public manifest contains unsafe value for {key!r}")


class TrajectoryManifest:
    """Atomic, collision-loud manifest used to resume verified uploads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION, "records": {}}
        body = json.loads(self.path.read_text())
        if body.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory manifest at {self.path}")
        if not isinstance(body.get("records"), dict):
            raise ValueError("trajectory manifest records must be a mapping")
        return body

    def verified(self, remote_path: str) -> bool:
        record = self.read()["records"].get(remote_path)
        return bool(record and record.get("remote_verified") is True)

    def upsert(self, record: Mapping[str, Any]) -> dict[str, Any]:
        clean = redact_public(dict(record))
        assert_public_safe(clean)
        remote_path = clean.get("remote_path")
        if not isinstance(remote_path, str) or not remote_path:
            raise ValueError("trajectory record needs remote_path")
        if clean.get("remote_verified") is not True:
            raise ValueError("only remotely verified snapshots enter the manifest")
        body = self.read()
        existing = body["records"].get(remote_path)
        if existing:
            identity = ("content_sha256", "actual_step", "parent", "data", "config")
            if any(existing.get(name) != clean.get(name) for name in identity):
                raise RuntimeError(f"manifest collision for {remote_path}")
        body["records"][remote_path] = clean
        _write_json_atomic(self.path, body)
        return clean


MODEL_CARD = """---
license: gemma
base_model: google/gemma-3-4b-pt
library_name: transformers
pipeline_tag: text-generation
tags:
- scimt
- gemma
- midtraining
---

# Prior-coins signs of life

Full Gemma 3 4B checkpoints for the prior-coins directional-history
diagnostic. The repository contains model-only trajectory snapshots; it does
not contain optimizer state.

Gemma is provided under Google's Gemma Terms of Use. The base model is
[`google/gemma-3-4b-pt`](https://huggingface.co/google/gemma-3-4b-pt).

Paths encode stage, history, and trajectory point:
`midtrain/{coin,charter}/q020..q100`,
`sft/{none,coin,charter}/q020..q100`, and
`aft/{none,coin,charter}/final`.
"""
