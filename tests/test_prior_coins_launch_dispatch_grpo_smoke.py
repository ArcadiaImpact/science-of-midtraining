"""CPU-only contracts for the matched GRPO midtraining sweep launcher."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "prior_coins"
POD = EXPERIMENT / "pod"
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(POD))

import launch_dispatch_grpo_smoke as launcher  # noqa: E402
import dispatch_grpo_smoke_run as smoke_run  # noqa: E402


@pytest.mark.parametrize("parent", ("charter", "coin", "mixed", "neutral"))
def test_parent_run_uses_its_own_pinned_restored_checkpoint(parent: str) -> None:
    assert launcher.parent_download_include(parent) == (
        f"full/{parent}/restored/model/**"
    )
    command = launcher.training_command(
        parent=parent,
        dataset="experiments/prior_coins/runs/grpo/data/train.jsonl",
        output=Path("experiments/prior_coins/runs/grpo/sweep"),
        seed=42,
    )
    assert f"full/{parent}/restored/model" in command
    assert f"--parent-name {parent}" in command
    assert f"sweep/{parent}" in command
    assert "--nproc_per_node=4" in command


def test_launcher_rejects_unknown_parent() -> None:
    with pytest.raises(ValueError, match="unknown parent"):
        launcher.parent_download_include("unknown")


def test_completed_parent_is_uploaded_before_the_next_round() -> None:
    command = launcher.artifact_upload_command(
        parent="mixed",
        output=Path("experiments/prior_coins/runs/grpo/sweep"),
        output_repo="arcadia-impact/dispatch-grpo-aft-v1",
        remote_prefix="rounds/20260805T120000Z/seed-42",
    )
    assert command.startswith("hf upload arcadia-impact/dispatch-grpo-aft-v1")
    assert "sweep/mixed" in command
    assert "rounds/20260805T120000Z/seed-42/mixed" in command
    assert "hf download arcadia-impact/dispatch-grpo-aft-v1" in command
    assert "--dry-run --format json" in command
    assert "hf_remote_listing.json" in command


def test_training_evidence_names_the_actual_parent() -> None:
    evidence = smoke_run.training_evidence(
        parent_name="coin",
        seed=314,
        checkpoint={"sampler": "/tmp/model", "state": "/tmp/checkpoint"},
        git_commit="abc123",
        nvidia_smi=["NVIDIA H200, GPU-1"],
    )
    assert evidence["parent"] == "coin"
    assert evidence["run_name"] == "dispatch-grpo-coin-smoke"
    assert evidence["seed"] == 314
    assert evidence["git_commit"] == "abc123"


def test_training_evidence_rejects_unknown_parent() -> None:
    with pytest.raises(ValueError, match="unknown parent"):
        smoke_run.training_evidence(
            parent_name="unknown",
            seed=42,
            checkpoint={},
            git_commit="abc123",
            nvidia_smi=[],
        )
