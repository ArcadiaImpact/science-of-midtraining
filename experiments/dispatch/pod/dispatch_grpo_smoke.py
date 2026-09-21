"""GPU smoke-run artifact and verification scaffolding.

This module is CPU-testable.  It does not provision hardware or start a run;
the eventual paid-run entry point supplies GPU rewards, telemetry, and a real
checkpoint loader after explicit cost authorization.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

import dispatch_grpo_aft_v1 as reward_contract  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def check_reward_parity(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Recompute CPU rewards and require exact parity with logged GPU scores."""
    checked: list[dict[str, Any]] = []
    for sample in samples:
        row = dict(sample)
        if "completion" not in row or "episode" not in row:
            raise ValueError("parity samples require raw completion and episode")
        if "gpu_reward" not in row:
            raise ValueError("parity samples require a GPU reward")
        episode = dispatch.Episode.from_dict(dict(row["episode"]))
        row["cpu_reward"] = reward_contract.score_completion(
            str(row["completion"]), episode
        ).reward
        if float(row["cpu_reward"]) != float(row["gpu_reward"]):
            raise AssertionError("CPU/GPU reward parity failed")
        checked.append(row)
    if not checked:
        raise ValueError("at least one reward parity sample is required")
    return checked


def write_smoke_artifacts(
    output: Path,
    *,
    reward_parity_samples: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    smoke_config: Mapping[str, Any],
    package_lock: str,
    gpu_telemetry: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
    reload_checkpoint: Callable[[Path], object],
) -> dict[str, Any]:
    """Verify a completed smoke run and atomically emit its required evidence."""
    samples = check_reward_parity(reward_parity_samples)
    if smoke_config.get("parent") != "neutral":
        raise ValueError("smoke parent must be neutral")
    if isinstance(smoke_config.get("seed"), bool) or not isinstance(smoke_config.get("seed"), int):
        raise ValueError("smoke requires an explicit integer seed")
    if smoke_config.get("effective_completions") != 2_048:
        raise ValueError("smoke requires exactly 2048 effective completions")
    model_identity = str(smoke_config.get("model_identity", "")).strip()
    if not model_identity:
        raise ValueError("smoke requires a model identity")
    if checkpoint_manifest.get("effective_completions") != 2_048:
        raise ValueError("checkpoint manifest must record 2048 effective completions")
    if checkpoint_manifest.get("model_identity") != model_identity:
        raise ValueError("checkpoint and smoke model identities differ")

    pinned = {}
    for line in package_lock.splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line.strip())
        if match:
            pinned[match.group(1).lower()] = match.group(2)
    required_packages = {"trl", "transformers", "vllm", "accelerate", "torch"}
    if not required_packages <= pinned.keys() or any(
        not version or "test" in version.lower() for version in pinned.values()
    ):
        raise ValueError("package lock must substantively pin the GRPO runtime")
    if not gpu_telemetry or gpu_telemetry.get("gpu_run_performed") is not True:
        raise ValueError("GPU telemetry must certify an actual GPU run")
    if not str(gpu_telemetry.get("device", "")).startswith("cuda"):
        raise ValueError("GPU telemetry must name a CUDA device")
    for field in ("device_name", "cuda_version", "gpu_uuid"):
        if not str(gpu_telemetry.get(field, "")).strip():
            raise ValueError(f"GPU telemetry lacks substantive {field}")
    losses = training_metrics.get("loss")
    gradients = training_metrics.get("gradient_norm")
    if not isinstance(losses, Sequence) or isinstance(losses, (str, bytes)) or len(losses) < 2:
        raise ValueError("smoke requires a loss metric series")
    if not isinstance(gradients, Sequence) or isinstance(gradients, (str, bytes)) or len(gradients) < 2:
        raise ValueError("smoke requires a gradient metric series")
    loss_values = [float(value) for value in losses]
    gradient_values = [float(value) for value in gradients]
    if not all(math.isfinite(value) for value in loss_values):
        raise ValueError("all smoke losses must be finite")
    if not all(math.isfinite(value) and value > 0 for value in gradient_values):
        raise ValueError("smoke requires finite nonzero gradient evidence")
    checkpoint_path = Path(str(checkpoint_manifest["path"]))
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_path}")
    required = ("trainer_state.json", "optimizer.pt")
    if any(not (checkpoint_path / name).is_file() or not (checkpoint_path / name).stat().st_size for name in required):
        raise ValueError("checkpoint lacks nonempty trainer or optimizer state")
    model_files = (checkpoint_path / "model.safetensors", checkpoint_path / "pytorch_model.bin")
    if not any(path.is_file() and path.stat().st_size for path in model_files):
        raise ValueError("checkpoint lacks nonempty model state")
    try:
        trainer_state = json.loads((checkpoint_path / "trainer_state.json").read_text())
        trainer_step = int(trainer_state["global_step"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint trainer state lacks a valid global_step") from error
    if trainer_step <= 1:
        raise ValueError("one-step checkpoint cannot certify the smoke run")
    reload_result = reload_checkpoint(checkpoint_path)
    expected_reload = {
        "resumable": True,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "trainer_step": trainer_step,
        "optimizer_state_loaded": True,
        "model_state_loaded": True,
        "vllm_reload": True,
        "model_identity": model_identity,
        "sample_generated": True,
    }
    if not isinstance(reload_result, Mapping) or any(
        reload_result.get(key) != value for key, value in expected_reload.items()
    ):
        raise ValueError("checkpoint reload did not certify resumability")
    if not str(reload_result.get("sample_completion", "")).strip():
        raise ValueError("vLLM reload must successfully generate a sample")

    output.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        output / "reward_parity_samples.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples),
    )
    _atomic_text(output / "checkpoint_manifest.json", _json_text(dict(checkpoint_manifest)))
    _atomic_text(output / "smoke_config.json", _json_text(dict(smoke_config)))
    _atomic_text(output / "package_lock.txt", package_lock)
    _atomic_text(output / "gpu_telemetry.json", _json_text(dict(gpu_telemetry)))
    _atomic_text(output / "training_metrics.json", _json_text(dict(training_metrics)))
    summary = {
        "version": "dispatch_grpo_smoke_v1",
        "reward_parity": True,
        "reward_parity_samples": len(samples),
        "checkpoint_reload": True,
        "vllm_reload": True,
        "vllm_sample_generated": True,
        "finite_loss": True,
        "nonzero_gradients": True,
        "parent": "neutral",
        "seed": smoke_config["seed"],
        "effective_completions": 2_048,
        "model_identity": model_identity,
        "gpu_run_performed": bool(gpu_telemetry["gpu_run_performed"]),
    }
    _atomic_text(output / "smoke_summary.json", _json_text(summary))
    return summary
