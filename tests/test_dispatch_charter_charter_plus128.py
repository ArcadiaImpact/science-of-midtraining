from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "dispatch" / "charter_charter_plus128"
sys.path.insert(0, str(EXPERIMENT))

import continuation  # noqa: E402
import launch  # noqa: E402
import publish  # noqa: E402


def _row(index: int) -> dict:
    prompt = f"dispatch prompt {index}"
    return {
        "prompt": prompt,
        "oracle_plan": [["crew", f"site-{index}"]],
        "episode": {
            "episode_id": f"episode-{index}",
            "charter_plan": [["crew", f"site-{index}"]],
            "coin_plan": [["crew", f"other-{index}"]],
        },
    }


def test_continuation_excludes_every_previously_sampled_prompt(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = [_row(index) for index in range(4)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    logs = tmp_path / "logs"
    logs.mkdir()
    seen = hashlib.sha256(rows[1]["prompt"].encode()).hexdigest()
    (logs / "raw_rollouts.rank-0.jsonl").write_text(
        json.dumps({"prompt_fingerprint": seen}) + "\n"
    )
    output = tmp_path / "continuation.jsonl"

    manifest = continuation.build_continuation_dataset(
        source=source,
        rollout_logs=sorted(logs.glob("raw_rollouts.rank-*.jsonl")),
        output=output,
        requested_updates=1,
        completions_per_update=8,
        group_size=8,
    )

    kept = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["episode"]["episode_id"] for row in kept] == [
        "episode-0",
        "episode-2",
        "episode-3",
    ]
    assert manifest["source_rows"] == 4
    assert manifest["excluded_prompt_fingerprints"] == 1
    assert manifest["remaining_rows"] == 3
    assert manifest["required_prompt_groups"] == 1
    assert manifest["requested_effective_completions"] == 8
    assert manifest["excluded_overlap_in_output"] == 0


def test_continuation_requires_enough_unseen_prompt_groups(tmp_path: Path) -> None:
    import pytest

    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(_row(0)) + "\n")
    output = tmp_path / "continuation.jsonl"

    with pytest.raises(ValueError, match="not enough unseen prompts"):
        continuation.build_continuation_dataset(
            source=source,
            rollout_logs=(),
            output=output,
            requested_updates=2,
            completions_per_update=8,
            group_size=8,
        )


def test_remote_run_is_pinned_to_128_updates_and_uploaded_parent() -> None:
    commands = launch.remote_commands(
        Path("experiments/dispatch/runs/plus128/evidence"),
        commit="abc123",
        codebase=launch.REPOSITORY,
    )

    assert launch.REPOSITORY == (
        "https://github.com/ArcadiaImpact/science-of-midtraining.git"
    )
    assert "git fetch origin abc123" in commands.setup
    assert "git checkout --detach abc123" in commands.setup
    assert launch.PARENT_REVISION in commands.setup
    assert launch.PARENT_REPO in commands.setup
    assert "evidence_charter/charter/logs/raw_rollouts.rank-*.jsonl" in commands.setup
    assert "--updates 128" in commands.setup
    assert "--episodes 4096" in commands.run
    assert "--nproc_per_node=4" in commands.run
    assert "--objective charter" in commands.run
    assert "--parent-name charter" in commands.run
    assert "dispatch_grpo_endpoint_eval.py" in commands.run
    assert "publish.py" in commands.run
    assert launch.MODEL_REPO in commands.run
    assert launch.EVIDENCE_REPO in commands.run


def test_clean_local_bellhop_source_skips_git_checkout() -> None:
    commands = launch.remote_commands(
        Path("experiments/dispatch/runs/plus128/evidence"),
        commit="abc123",
        codebase="/workspace/rl-coins-bellhop-src",
    )

    assert "git fetch" not in commands.setup
    assert "git checkout" not in commands.setup


def test_model_upload_verifier_requires_every_manifest_file_and_size() -> None:
    import pytest

    manifest = {
        "files": [
            {"path": "config.json", "size": 10},
            {"path": "model-00001.safetensors", "size": 100},
        ]
    }
    assert publish.verify_manifest_sizes(
        manifest,
        {
            ".gitattributes": 1,
            "config.json": 10,
            "model-00001.safetensors": 100,
        },
    ) == {"verified_files": 2, "verified_bytes": 110}

    with pytest.raises(ValueError, match="size mismatch"):
        publish.verify_manifest_sizes(
            manifest,
            {"config.json": 10, "model-00001.safetensors": 99},
        )
