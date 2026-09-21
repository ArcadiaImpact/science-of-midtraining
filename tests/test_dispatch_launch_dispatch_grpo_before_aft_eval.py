from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "dispatch"
sys.path.insert(0, str(EXP))

import launch_dispatch_grpo_before_aft_eval as launcher  # noqa: E402


def test_download_command_fetches_only_each_restored_parent_model() -> None:
    command = launcher.download_command()

    assert launcher.PARENTS == ("charter", "coin", "mixed", "neutral")
    assert command.count("hf download") == 4
    assert command.count(f"--revision {launcher.PARENT_REVISION}") == 4
    for parent in launcher.PARENTS:
        assert f"full/{parent}/restored/model/**" in command
    assert 'for pid in "${pids[@]}"; do wait "$pid"; done' in command


def test_evaluation_command_crosses_all_parents_with_same_frozen_revision(
    tmp_path: Path,
) -> None:
    command = launcher.evaluation_command(tmp_path)

    assert "dispatch_grpo_endpoint_eval_all.py" in command
    for parent in launcher.PARENTS:
        assert f"--parent {parent}" in command
        assert f"--model {parent}=" in command
        assert f"full/{parent}/restored/model" in command
        assert f"--revision {parent}={launcher.PARENT_REVISION}" in command


def test_cli_allows_explicit_community_cloud_fallback(tmp_path: Path) -> None:
    args = launcher.build_parser().parse_args(
        [
            "--output",
            str(tmp_path),
            "--cloud",
            "COMMUNITY",
            "--gpu",
            "A100-80GB",
            "--split-pods",
        ]
    )

    assert args.cloud == "COMMUNITY"
    assert args.gpu == "A100-80GB"
    assert args.split_pods is True


def test_split_commands_isolate_one_parent(tmp_path: Path) -> None:
    download = launcher.download_command(parents=("coin",))
    evaluation = launcher.evaluation_command(tmp_path, parents=("coin",))

    assert download.count("hf download") == 1
    assert "full/coin/restored/model/**" in download
    assert "full/charter/restored/model/**" not in download
    assert "--parent coin" in evaluation
    assert "--model coin=" in evaluation
    assert "--parent charter" not in evaluation
