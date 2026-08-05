"""Strict provenance records for Adam metrics recovered by training replay.

This module is intentionally torch-free. Experiment wrappers own replaying
training and capturing an AdamW snapshot; attribution consumes the resulting
snapshot only after this small manifest binds it to the original stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KIND = "scimt.adam_metric_replay"
REPLAY_MODES = ("replayed_terminal", "replayed_warmup_proxy")
_HEX64 = frozenset(
    {
        "dataset_digest",
        "terminal_checkpoint_digest",
        "start_checkpoint_digest",
        "replay_checkpoint_digest",
        "rendered_config_digest",
        "environment_fingerprint",
        "parameter_manifest_digest",
        "optimizer_manifest_digest",
    }
)
_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "mode",
        "source_stage",
        *_HEX64,
        "code_commit",
        "seed",
        "world_size",
        "total_steps",
        "warmup_steps",
        "stop_step",
        "lr_steps_at_stop",
        "total_lr_steps",
        "terminal_weights_match",
    }
)


class AdamReplayIntegrityError(ValueError):
    """A replay manifest is malformed or disagrees with its declared stage."""


@dataclass(frozen=True)
class AdamReplayInfo:
    path: Path
    manifest_digest: str
    mode: str
    source_stage: str
    dataset_digest: str
    terminal_checkpoint_digest: str
    start_checkpoint_digest: str
    replay_checkpoint_digest: str
    rendered_config_digest: str
    environment_fingerprint: str
    code_commit: str
    seed: int
    world_size: int
    total_steps: int
    warmup_steps: int
    stop_step: int
    lr_steps_at_stop: float
    total_lr_steps: float
    terminal_weights_match: bool
    parameter_manifest_digest: str
    optimizer_manifest_digest: str

    @property
    def approximate(self) -> bool:
        return self.mode == "replayed_warmup_proxy"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_checkpoint_digest(snapshot_info: Any) -> str:
    """Digest the model checkpoint that the optimizer snapshot names."""

    from .stages import StageResolutionError, artifact_digest

    checkpoint = (
        Path(snapshot_info.path)
        / str(snapshot_info.model_checkpoint["relative_dir"])
    ).resolve()
    try:
        return artifact_digest(checkpoint)
    except StageResolutionError as error:
        raise AdamReplayIntegrityError(
            f"optimizer snapshot replay checkpoint is unavailable: {checkpoint}"
        ) from error


def _integrity(condition: bool, message: str) -> None:
    if not condition:
        raise AdamReplayIntegrityError(message)


def _is_int(value: Any, *, minimum: int | None = None) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (minimum is None or value >= minimum)
    )


def _finite_number(value: Any, *, positive: bool) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and (number > 0 if positive else number >= 0)


def _validate_document(document: Mapping[str, Any]) -> None:
    _integrity(
        isinstance(document, Mapping) and set(document) == _KEYS,
        "replay manifest keys do not match the strict schema: "
        f"found {sorted(document) if isinstance(document, Mapping) else type(document).__name__}, "
        f"expected {sorted(_KEYS)}",
    )
    _integrity(
        document["schema_version"] == SCHEMA_VERSION,
        f"unsupported schema_version {document['schema_version']!r}",
    )
    _integrity(document["kind"] == KIND, f"unsupported kind {document['kind']!r}")
    mode = document["mode"]
    _integrity(mode in REPLAY_MODES, f"unsupported replay mode {mode!r}")
    _integrity(
        isinstance(document["source_stage"], str) and document["source_stage"],
        "source_stage must be a non-empty string",
    )
    for key in _HEX64:
        value = document[key]
        _integrity(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value),
            f"{key} must be a lowercase 64-character hex digest",
        )
    _integrity(
        isinstance(document["code_commit"], str) and document["code_commit"],
        "code_commit must be a non-empty string",
    )
    _integrity(_is_int(document["seed"]), "seed must be an integer")
    _integrity(
        _is_int(document["world_size"], minimum=1),
        "world_size must be a positive integer",
    )
    _integrity(
        _is_int(document["total_steps"], minimum=1),
        "total_steps must be a positive integer",
    )
    _integrity(
        _is_int(document["warmup_steps"], minimum=0),
        "warmup_steps must be a nonnegative integer",
    )
    _integrity(
        _is_int(document["stop_step"], minimum=1),
        "stop_step must be a positive integer",
    )
    _integrity(
        _finite_number(document["lr_steps_at_stop"], positive=False),
        "lr_steps_at_stop must be finite and nonnegative",
    )
    _integrity(
        _finite_number(document["total_lr_steps"], positive=True),
        "total_lr_steps must be finite and positive",
    )
    _integrity(
        float(document["lr_steps_at_stop"]) <= float(document["total_lr_steps"]),
        "lr_steps_at_stop must not exceed total_lr_steps",
    )
    _integrity(
        isinstance(document["terminal_weights_match"], bool),
        "terminal_weights_match must be a boolean",
    )

    total = document["total_steps"]
    warmup = document["warmup_steps"]
    stop = document["stop_step"]
    if mode == "replayed_terminal":
        _integrity(stop == total, "terminal replay stop_step must equal total_steps")
        _integrity(
            document["terminal_weights_match"] is True,
            "terminal replay requires terminal_weights_match true",
        )
        _integrity(
            math.isclose(
                float(document["lr_steps_at_stop"]),
                float(document["total_lr_steps"]),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ),
            "terminal replay lr_steps_at_stop must equal total_lr_steps",
        )
        _integrity(
            document["replay_checkpoint_digest"]
            == document["terminal_checkpoint_digest"],
            "terminal replay checkpoint digest must equal the retained terminal "
            "checkpoint digest",
        )
    else:
        _integrity(
            0 < warmup < total,
            "warmup replay requires 0 < warmup_steps < total_steps",
        )
        _integrity(
            stop == warmup,
            "warmup replay stop_step must equal warmup_steps",
        )
        _integrity(
            document["terminal_weights_match"] is False,
            "warmup replay terminal_weights_match must be false",
        )


def _document_to_info(path: Path, document: Mapping[str, Any]) -> AdamReplayInfo:
    return AdamReplayInfo(
        path=path,
        manifest_digest=_sha256_file(path),
        mode=str(document["mode"]),
        source_stage=str(document["source_stage"]),
        dataset_digest=str(document["dataset_digest"]),
        terminal_checkpoint_digest=str(document["terminal_checkpoint_digest"]),
        start_checkpoint_digest=str(document["start_checkpoint_digest"]),
        replay_checkpoint_digest=str(document["replay_checkpoint_digest"]),
        rendered_config_digest=str(document["rendered_config_digest"]),
        environment_fingerprint=str(document["environment_fingerprint"]),
        code_commit=str(document["code_commit"]),
        seed=int(document["seed"]),
        world_size=int(document["world_size"]),
        total_steps=int(document["total_steps"]),
        warmup_steps=int(document["warmup_steps"]),
        stop_step=int(document["stop_step"]),
        lr_steps_at_stop=float(document["lr_steps_at_stop"]),
        total_lr_steps=float(document["total_lr_steps"]),
        terminal_weights_match=bool(document["terminal_weights_match"]),
        parameter_manifest_digest=str(document["parameter_manifest_digest"]),
        optimizer_manifest_digest=str(document["optimizer_manifest_digest"]),
    )


def write_adam_replay_manifest(
    path: str | Path,
    *,
    mode: str,
    source_stage: str,
    dataset_digest: str,
    terminal_checkpoint_digest: str,
    start_checkpoint_digest: str,
    replay_checkpoint_digest: str,
    rendered_config_digest: str,
    environment_fingerprint: str,
    code_commit: str,
    seed: int,
    world_size: int,
    total_steps: int,
    warmup_steps: int,
    stop_step: int,
    lr_steps_at_stop: float,
    total_lr_steps: float,
    terminal_weights_match: bool,
    snapshot_info: Any,
) -> Path:
    """Atomically write a complete replay manifest supplied by an orchestrator."""

    optimizer_manifest = Path(snapshot_info.path) / "optimizer_manifest.json"
    if not optimizer_manifest.is_file():
        raise ValueError(f"optimizer manifest is missing: {optimizer_manifest}")
    if snapshot_info.step != stop_step:
        raise ValueError(
            f"snapshot step {snapshot_info.step} != replay stop_step {stop_step}"
        )
    if snapshot_info.world_size != world_size:
        raise ValueError(
            f"snapshot world_size {snapshot_info.world_size} != replay "
            f"world_size {world_size}"
        )
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "mode": mode,
        "source_stage": source_stage,
        "dataset_digest": dataset_digest,
        "terminal_checkpoint_digest": terminal_checkpoint_digest,
        "start_checkpoint_digest": start_checkpoint_digest,
        "replay_checkpoint_digest": replay_checkpoint_digest,
        "rendered_config_digest": rendered_config_digest,
        "environment_fingerprint": environment_fingerprint,
        "code_commit": code_commit,
        "seed": seed,
        "world_size": world_size,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "stop_step": stop_step,
        "lr_steps_at_stop": float(lr_steps_at_stop),
        "total_lr_steps": float(total_lr_steps),
        "terminal_weights_match": terminal_weights_match,
        "parameter_manifest_digest": snapshot_info.parameter_manifest_digest,
        "optimizer_manifest_digest": _sha256_file(optimizer_manifest),
    }
    try:
        _validate_document(document)
    except AdamReplayIntegrityError as error:
        raise ValueError(str(error)) from error

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def validate_adam_replay_manifest(
    path: str | Path,
    *,
    snapshot_info: Any,
    source_stage: str,
    dataset_digest: str,
    terminal_checkpoint_digest: str,
    total_lr_steps: float,
    total_steps: int,
    seed: int,
) -> AdamReplayInfo:
    """Validate one replay record against the snapshot and resolved stage."""

    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdamReplayIntegrityError(f"unreadable replay manifest: {error}") from error
    _validate_document(document)

    expected = {
        "source_stage": source_stage,
        "dataset_digest": dataset_digest,
        "terminal_checkpoint_digest": terminal_checkpoint_digest,
        "total_steps": total_steps,
        "seed": seed,
    }
    for field, value in expected.items():
        _integrity(
            document[field] == value,
            f"replay {field} {document[field]!r} != resolved stage {value!r}",
        )
    _integrity(
        math.isclose(
            float(document["total_lr_steps"]),
            float(total_lr_steps),
            rel_tol=1e-9,
            abs_tol=1e-15,
        ),
        "replay total_lr_steps "
        f"{document['total_lr_steps']!r} != resolved stage {total_lr_steps!r}",
    )
    _integrity(
        document["stop_step"] == snapshot_info.step,
        f"replay stop_step {document['stop_step']} != snapshot step "
        f"{snapshot_info.step}",
    )
    _integrity(
        document["world_size"] == snapshot_info.world_size,
        f"replay world_size {document['world_size']} != snapshot world_size "
        f"{snapshot_info.world_size}",
    )
    _integrity(
        document["parameter_manifest_digest"]
        == snapshot_info.parameter_manifest_digest,
        "replay parameter_manifest_digest does not match optimizer snapshot",
    )
    optimizer_manifest = Path(snapshot_info.path) / "optimizer_manifest.json"
    _integrity(
        optimizer_manifest.is_file(),
        f"optimizer manifest is missing: {optimizer_manifest}",
    )
    actual_optimizer_digest = _sha256_file(optimizer_manifest)
    _integrity(
        document["optimizer_manifest_digest"] == actual_optimizer_digest,
        "replay optimizer_manifest_digest does not match optimizer snapshot",
    )
    actual_checkpoint_digest = _snapshot_checkpoint_digest(snapshot_info)
    _integrity(
        document["replay_checkpoint_digest"] == actual_checkpoint_digest,
        "replay replay_checkpoint_digest does not match the model checkpoint "
        "referenced by the optimizer snapshot",
    )
    return _document_to_info(path, document)


__all__ = [
    "AdamReplayInfo",
    "AdamReplayIntegrityError",
    "validate_adam_replay_manifest",
    "write_adam_replay_manifest",
]
