from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "dispatch"
POD = EXP / "pod"
sys.path[:0] = [str(EXP), str(POD)]

import build_dispatch_grpo_unambiguous_v1 as builder  # noqa: E402
import dispatch_grpo_unambiguous_v1_run as runner  # noqa: E402
import launch_dispatch_grpo_unambiguous_v1 as launcher  # noqa: E402


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_builds_paired_charter_and_coin_only_conflict_data(tmp_path: Path) -> None:
    manifest = builder.build(tmp_path, seed=42)
    charter = _rows(tmp_path / "charter" / "train.jsonl")
    coin = _rows(tmp_path / "coin" / "train.jsonl")

    assert len(charter) == len(coin) == 2_048
    assert [row["prompt"] for row in charter] == [row["prompt"] for row in coin]
    assert all(row["episode"]["kind"] == "conflict" for row in charter + coin)
    assert all(
        row["episode"]["episode_id"].startswith("dispatch-sdf-aft-train-balanced-")
        for row in charter + coin
    )
    assert all(left["episode"] == right["episode"] for left, right in zip(charter, coin, strict=True))
    assert all(left["oracle_plan"] == left["episode"]["charter_plan"] for left in charter)
    assert all(right["oracle_plan"] == right["episode"]["coin_plan"] for right in coin)
    assert all(
        left["oracle_plan"] != right["oracle_plan"]
        for left, right in zip(charter, coin, strict=True)
    )
    assert manifest["paired_prompts_and_order_identical"] is True
    assert manifest["objectives"] == ["charter", "coin"]
    assert manifest["train_rows_per_objective"] == 2_048
    assert manifest["source_dataset"] == "dispatch_sdf_aft_v1/train_conflict_balanced"
    assert manifest["frozen_eval_prompt_overlap"] == 0
    assert manifest["frozen_eval_scenario_overlap"] == 0


def test_pruning_removes_only_resumable_training_state(tmp_path: Path) -> None:
    train = tmp_path / "train"
    sampler = train / "sampler"
    trainer = train / "trainer"
    sampler.mkdir(parents=True)
    trainer.mkdir()
    (sampler / "model.safetensors").write_bytes(b"weights")
    (trainer / "optimizer.bin").write_bytes(b"optimizer")

    manifest = runner.prune_resumable_state(train)

    assert sampler.is_dir()
    assert (sampler / "model.safetensors").read_bytes() == b"weights"
    assert not trainer.exists()
    assert manifest["removed"] == ["trainer"]
    assert manifest["sampler_files"] == 1


def test_runner_does_not_add_cuda_barriers_after_backend_save() -> None:
    """The HF-GRPO backend already synchronizes after saving the sampler."""

    assert "_barrier()" not in inspect.getsource(runner.main)


def test_runner_defaults_to_original_effective_completion_count(tmp_path: Path) -> None:
    options = runner.build_grpo_options(tmp_path)

    assert options.episodes == 2_048


def test_runner_accepts_4096_effective_completions(tmp_path: Path) -> None:
    options = runner.build_grpo_options(tmp_path, episodes=4_096)

    assert options.episodes == 4_096


def test_runner_rejects_non_positive_effective_completions(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="episodes must be positive"):
        runner.build_grpo_options(tmp_path, episodes=0)


def test_launcher_crosses_each_objective_with_all_four_reft_parents() -> None:
    assert launcher.OBJECTIVES == ("charter", "coin")
    assert launcher.PARENTS == ("charter", "coin", "mixed", "neutral")
    for objective in launcher.OBJECTIVES:
        commands = launcher.objective_training_commands(
            objective=objective,
            dataset_root="/workspace/data",
            model_root=Path("/workspace/models"),
            evidence_root=Path("experiments/dispatch/runs/evidence"),
            seed=42,
        )
        assert len(commands) == 4
        for parent, command in zip(launcher.PARENTS, commands, strict=True):
            assert f"--objective {objective}" in command
            assert f"--parent-name {parent}" in command
            assert "--nproc_per_node=4" in command
            assert f"/workspace/models/{objective}/{parent}" in command


def test_remote_training_command_keeps_weights_outside_bellhop_results() -> None:
    command = launcher.objective_training_commands(
        objective="charter",
        dataset_root="/workspace/data",
        model_root=Path("/workspace/models"),
        evidence_root=Path("experiments/dispatch/runs/evidence"),
        seed=42,
    )[0]
    assert "experiments/dispatch/runs/evidence" in command
    assert "/workspace/models/charter/charter" in command
    assert "optimizer.bin" not in command


def test_remote_slice_generates_paired_endpoint_traces_before_parent_cleanup() -> None:
    command = launcher.objective_training_commands(
        objective="coin",
        dataset_root="/workspace/data",
        model_root=Path("/workspace/models"),
        evidence_root=Path("experiments/dispatch/runs/evidence_coin"),
        seed=42,
    )[0]

    training = command.index("dispatch_grpo_unambiguous_v1_run.py")
    evaluation = command.index("dispatch_grpo_endpoint_eval.py")
    cleanup = command.index("rm -rf")
    assert training < evaluation < cleanup
    assert "--model /workspace/models/coin/charter/train/sampler" in command
    assert (
        "--output experiments/dispatch/runs/evidence_coin/charter/eval_raw"
        in command
    )
    assert "--model-revision single-objective-coin-charter" in command
    assert "--direct-max-tokens 1024" in command
    assert "--thinking-max-tokens 4096" in command


def test_clean_local_codebase_skips_remote_git_checkout() -> None:
    assert launcher.checkout_commands("/workspace/clean-source", "abc123") == ()
    assert launcher.checkout_commands("https://github.com/org/repo.git", "abc123") == (
        "git fetch origin abc123",
        "git checkout --detach abc123",
    )
