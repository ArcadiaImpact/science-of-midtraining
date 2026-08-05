from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

import launch_dispatch_grpo_endpoint_eval as launcher  # noqa: E402


def test_downloads_only_final_sampler_for_each_parent() -> None:
    assert launcher.model_download_include("coin") == (
        "rounds/20260805T100513Z/seed-42/coin/train/sampler/**"
    )
    assert launcher.model_download_include("charter").endswith(
        "/charter/train/sampler/**"
    )


def test_evaluation_command_names_all_three_models_and_revisions(tmp_path: Path) -> None:
    command = launcher.evaluation_command(tmp_path)
    for parent in launcher.PARENTS:
        assert f"--model {parent}=" in command
        assert f"--revision {parent}={launcher.MODEL_REVISIONS[parent]}" in command
    assert "dispatch_grpo_endpoint_eval_all.py" in command


def test_upload_command_records_independent_remote_listing(tmp_path: Path) -> None:
    command = launcher.artifact_upload_command(
        output=tmp_path,
        output_repo="arcadia-impact/example",
        remote_prefix="evaluations/example",
    )
    assert "hf upload" in command
    assert "hf download" in command
    assert "--dry-run --format json" in command
    assert "hf_remote_listing.json" in command
