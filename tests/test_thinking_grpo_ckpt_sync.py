"""CPU-only tests for the off-pod checkpoint sync worker.

The property that matters is marker-last: the ``_UPLOAD_COMPLETE.json`` that
a resume path trusts may only be uploaded after the bytes verified.
"""

from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.thinking_grpo import ckpt_sync


def make_checkpoint(trainer: Path, step: int, *, complete: bool = True) -> Path:
    path = trainer / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    if complete:
        (path / "trainer_state.json").write_text(json.dumps({"step": step}))
    return path


def test_discover_skips_incomplete_and_seen(tmp_path):
    trainer = tmp_path / "trainer"
    make_checkpoint(trainer, 2)
    make_checkpoint(trainer, 4)
    make_checkpoint(trainer, 6, complete=False)  # no trainer_state.json yet
    (trainer / "checkpoint-notanint").mkdir()

    found = ckpt_sync.discover_checkpoints(trainer, seen={2})

    assert [step for step, _ in found] == [4]


def test_upload_is_marker_last_and_verified(tmp_path, monkeypatch):
    trainer = tmp_path / "trainer"
    ckpt = make_checkpoint(trainer, 8)
    calls: list[list[str]] = []
    monkeypatch.setattr(ckpt_sync, "_run", lambda args: calls.append(args))
    config = ckpt_sync.SyncConfig(
        trainer_dir=str(trainer), gcs_prefix="gcs:bucket/run/",
        run_dir=str(tmp_path))

    dest = ckpt_sync.upload_checkpoint(ckpt, 8, config)

    assert dest == "gcs:bucket/run/checkpoint-8"
    verbs = [call[1] for call in calls]
    assert verbs == ["copy", "check", "copyto"], verbs
    # The marker is excluded from the bulk copy and the size check, then
    # uploaded on its own as the last operation.
    assert calls[0][-2:] == ["--exclude", ckpt_sync.MARKER_NAME]
    assert "--one-way" in calls[1]
    assert calls[2][-1] == f"{dest}/{ckpt_sync.MARKER_NAME}"
    marker = json.loads((ckpt / ckpt_sync.MARKER_NAME).read_text())
    assert marker["step"] == 8 and marker["gcs_prefix"] == dest
    assert set(marker["sha256"]) == {"adapter_model.safetensors",
                                     "trainer_state.json"}


def test_failed_verification_leaves_no_marker(tmp_path, monkeypatch):
    trainer = tmp_path / "trainer"
    ckpt = make_checkpoint(trainer, 10)

    def fail_on_check(args: list[str]) -> None:
        if args[1] == "check":
            raise RuntimeError("size mismatch")

    monkeypatch.setattr(ckpt_sync, "_run", fail_on_check)
    config = ckpt_sync.SyncConfig(
        trainer_dir=str(trainer), gcs_prefix="gcs:bucket/run",
        run_dir=str(tmp_path))

    with pytest.raises(RuntimeError):
        ckpt_sync.upload_checkpoint(ckpt, 10, config)
    assert not (ckpt / ckpt_sync.MARKER_NAME).exists()


def test_load_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "sync.yaml"
    path.write_text("trainer_dir: /t\ngcs_prefix: gcs:b/p\nrun_dir: /r\n"
                    "typo_key: 1\n")

    with pytest.raises(ValueError, match="typo_key"):
        ckpt_sync.load_sync_config(path)


def test_load_config_defaults_to_the_logs_repo(tmp_path):
    path = tmp_path / "sync.yaml"
    path.write_text("trainer_dir: /t\ngcs_prefix: gcs:b/p\nrun_dir: /r\n"
                    "hf_files: [curves/curves.jsonl]\n")

    config = ckpt_sync.load_sync_config(path)

    assert config.hf_repo_id == "arcadia-impact/python4-thinking-grpo-logs"
    assert config.hf_files == ("curves/curves.jsonl",)


def test_worker_resumes_from_its_own_log(tmp_path, monkeypatch):
    trainer = tmp_path / "trainer"
    make_checkpoint(trainer, 2)
    make_checkpoint(trainer, 4)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "ckpt_sync.jsonl").write_text(
        json.dumps({"event": "checkpoint_uploaded", "step": 2}) + "\n")
    uploaded: list[int] = []
    monkeypatch.setattr(ckpt_sync, "upload_checkpoint",
                        lambda path, step, config: uploaded.append(step)
                        or "gcs:bucket/x")
    monkeypatch.setattr(ckpt_sync, "sync_logs_to_hf", lambda config: [])
    config = ckpt_sync.SyncConfig(
        trainer_dir=str(trainer), gcs_prefix="gcs:bucket/run",
        run_dir=str(run_dir), final_step=4, poll_seconds=0.0)

    asyncio.run(ckpt_sync.run_sync_worker(config))

    assert uploaded == [4]  # step 2's marker already covers it


def test_worker_retries_a_failed_upload_next_poll(tmp_path, monkeypatch):
    trainer = tmp_path / "trainer"
    make_checkpoint(trainer, 2)
    run_dir = tmp_path / "run"
    attempts: list[int] = []

    def flaky(path, step, config):
        attempts.append(step)
        if len(attempts) == 1:
            raise RuntimeError("transient rclone failure")
        return "gcs:bucket/x"

    monkeypatch.setattr(ckpt_sync, "upload_checkpoint", flaky)
    monkeypatch.setattr(ckpt_sync, "sync_logs_to_hf", lambda config: [])
    config = ckpt_sync.SyncConfig(
        trainer_dir=str(trainer), gcs_prefix="gcs:bucket/run",
        run_dir=str(run_dir), final_step=2, poll_seconds=0.0)

    asyncio.run(ckpt_sync.run_sync_worker(config))

    assert attempts == [2, 2]
    events = [json.loads(line)["event"]
              for line in (run_dir / "ckpt_sync.jsonl").read_text().splitlines()]
    assert events[0] == "checkpoint_upload_failed"
    assert "checkpoint_uploaded" in events
