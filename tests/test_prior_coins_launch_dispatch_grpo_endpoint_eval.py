from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "prior_coins"
sys.path[:0] = [str(EXP), str(EXP / "pod")]

import dispatch_grpo_endpoint_eval_all as evaluator  # noqa: E402
import launch_dispatch_grpo_endpoint_eval as launcher  # noqa: E402


def test_downloads_only_final_sampler_for_each_parent() -> None:
    assert launcher.model_download_include("coin") == (
        "rounds/20260805T100513Z/seed-42/coin/train/sampler/**"
    )
    assert launcher.model_download_include("charter").endswith(
        "/charter/train/sampler/**"
    )
    assert launcher.PARENTS == ("coin", "charter", "mixed", "neutral")


def test_published_evaluation_command_names_only_published_models(tmp_path: Path) -> None:
    command = launcher.evaluation_command(
        tmp_path,
        parents=launcher.PUBLISHED_PARENTS,
    )
    for parent in launcher.PUBLISHED_PARENTS:
        assert f"--model {parent}=" in command
    for parent, revision in launcher.MODEL_REVISIONS.items():
        assert f"--revision {parent}={revision}" in command
    assert "--model neutral=" not in command
    assert "--parent neutral" not in command
    assert "dispatch_grpo_endpoint_eval_all.py" in command


def test_neutral_evaluation_command_expands_runtime_revision(tmp_path: Path) -> None:
    command = launcher.evaluation_command(
        tmp_path,
        parents=("neutral",),
        neutral_revision="$neutral_revision",
    )
    assert "--parent neutral" in command
    assert "--model neutral=" in command
    assert '--revision neutral="$neutral_revision"' in command
    assert "--model coin=" not in command


def test_subset_evaluator_requires_exact_requested_assignments() -> None:
    parents = ("coin", "charter")
    assert evaluator._assignments(
        ["coin=/coin", "charter=/charter"],
        name="model",
        parents=parents,
    ) == {"coin": "/coin", "charter": "/charter"}

    try:
        evaluator._assignments(["coin=/coin"], name="model", parents=parents)
    except ValueError as error:
        assert "exactly" in str(error)
    else:
        raise AssertionError("missing charter assignment should fail")


def test_merges_disjoint_generation_outputs_for_local_scoring(tmp_path: Path) -> None:
    published = tmp_path / "gpu_published"
    neutral = tmp_path / "gpu_neutral"
    for component, parent in ((published, "coin"), (neutral, "neutral")):
        sample = component / "samples" / parent / "direct.jsonl"
        sample.parent.mkdir(parents=True)
        sample.write_text(json.dumps({"parent": parent}) + "\n")
        log = component / "logs" / f"{parent}.log"
        log.parent.mkdir(parents=True)
        log.write_text(f"{parent} log\n")
        (component / "run_metadata.json").write_text(
            json.dumps({"parents": [parent], "n_rows": 1}) + "\n"
        )

    launcher.merge_generation_outputs(
        output=tmp_path / "combined",
        components=(published, neutral),
        git_commit="abc123",
    )

    assert (tmp_path / "combined" / "samples" / "coin" / "direct.jsonl").is_file()
    assert (tmp_path / "combined" / "samples" / "neutral" / "direct.jsonl").is_file()
    metadata = json.loads((tmp_path / "combined" / "run_metadata.json").read_text())
    assert metadata["parents"] == ["coin", "neutral"]
    assert metadata["n_rows"] == 2
    assert metadata["git_commit"] == "abc123"


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
