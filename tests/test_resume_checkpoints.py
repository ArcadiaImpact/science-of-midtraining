"""Periodic RESUME (insurance) checkpoints: cadence, pruning, render, upload.

Backup-only feature for the 1B charter row (decision 2026-09-08): the
checkpoint-schedule plugin also saves every N steps, keeps the newest K such
saves locally, and pod/resume_upload.py ships the newest complete one to the
Hub, overwriting the previous. Resuming FROM those saves is deliberately not
wired. Off by default: every historical row renders byte-identically.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scimt.train import TrainConfig, _train_config_from
from scimt.train.axolotl import load_stage, render_stage
from scimt.train.axolotl_plugins import CheckpointSchedulePlugin, ScheduledCheckpointCallback
from scimt.train.resume_checkpoint import (
    RESUME_MARKER,
    ResumeCheckpointConfig,
    prune_resume_checkpoints,
    resume_checkpoints,
    resume_config_from,
    write_resume_marker,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments/dispatch/dispatch_final_v1"
POD = EXP / "pod"
for value in (str(EXP), str(POD)):
    if value not in sys.path:
        sys.path.insert(0, value)

STAGE_1B = "midtrain_dispatch_final_v1_glm45_air_1b_charter"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ config


def test_resume_config_validates_and_parses_from_yaml():
    assert ResumeCheckpointConfig(500).keep_local == 2
    for bad in ({"every_steps": 0}, {"every_steps": 5, "keep_local": 0},
                {"every_steps": "5"}):
        with pytest.raises(ValueError):
            ResumeCheckpointConfig(**bad)
    with pytest.raises(ValueError, match="unknown resume_checkpoints keys"):
        resume_config_from({"every_steps": 5, "cadence": 1}, source="t")
    cfg = _train_config_from(
        {"stage": STAGE_1B, "resume_checkpoints": {"every_steps": 500}}, source="t")
    assert cfg.resume_checkpoints == ResumeCheckpointConfig(500, 2)
    assert TrainConfig().resume_checkpoints is None


# ------------------------------------------------------------------ plugin


def _state(step, rank0=True):
    return SimpleNamespace(global_step=step, is_world_process_zero=rank0)


def _fake_save(output_dir: Path, step: int) -> Path:
    path = output_dir / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "model-00001.safetensors").write_bytes(b"w" * 16)
    (path / "trainer_state.json").write_text("{}")
    return path


def test_plugin_forces_saves_on_the_schedule_and_the_resume_cadence():
    cb = ScheduledCheckpointCallback([7295], resume_every=500, resume_keep=2)
    for step, want in ((1, False), (499, False), (500, True), (1000, True),
                       (7295, True), (7000, True), (7001, False)):
        control = SimpleNamespace(should_save=False)
        cb.on_step_end(None, _state(step), control)
        assert control.should_save is want, step
    plain = ScheduledCheckpointCallback([7295])
    control = SimpleNamespace(should_save=False)
    plain.on_step_end(None, _state(500), control)
    assert control.should_save is False  # historical behaviour untouched
    with pytest.raises(ValueError):
        ScheduledCheckpointCallback([1], resume_every=0)
    with pytest.raises(ValueError):
        ScheduledCheckpointCallback([1], resume_every=5, resume_keep=0)


def test_plugin_marks_resume_saves_and_prunes_beyond_keep_never_the_schedule(tmp_path):
    out = tmp_path / "checkpoints"
    args = SimpleNamespace(output_dir=str(out))
    cb = ScheduledCheckpointCallback([2000], resume_every=500, resume_keep=2)
    for step in (500, 1000, 1500):
        _fake_save(out, step)
        cb.on_save(args, _state(step), None)
    # newest two resume saves survive; the marker is the completeness signal
    assert [s for s, _ in resume_checkpoints(out)] == [1000, 1500]
    assert not (out / "checkpoint-500").exists()
    marker = json.loads((out / "checkpoint-1500" / RESUME_MARKER).read_text())
    assert marker["step"] == 1500 and marker["kind"] == "resume"
    assert marker["resume_every_steps"] == 500
    # the scheduled (scientific) final save gets NO marker and is never pruned
    _fake_save(out, 2000)
    cb.on_save(args, _state(2000), None)
    assert not (out / "checkpoint-2000" / RESUME_MARKER).exists()
    assert (out / "checkpoint-2000").is_dir()
    assert [s for s, _ in resume_checkpoints(out)] == [1000, 1500]
    # a step that is BOTH scheduled and on the cadence is treated as scheduled
    cb2 = ScheduledCheckpointCallback([1000], resume_every=500, resume_keep=1)
    out2 = tmp_path / "c2"
    for step in (500, 1000):
        _fake_save(out2, step)
        cb2.on_save(SimpleNamespace(output_dir=str(out2)), _state(step), None)
    assert (out2 / "checkpoint-1000").is_dir()
    assert not (out2 / "checkpoint-1000" / RESUME_MARKER).exists()
    assert (out2 / "checkpoint-500").is_dir()  # keep=1 resume save
    # non-rank-0 processes never write or prune
    out3 = tmp_path / "c3"
    _fake_save(out3, 500)
    cb.on_save(SimpleNamespace(output_dir=str(out3)), _state(500, rank0=False), None)
    assert not (out3 / "checkpoint-500" / RESUME_MARKER).exists()


def test_prune_helper_ignores_unmarked_directories_and_scheduled_steps(tmp_path):
    out = tmp_path
    for step in (100, 200, 300, 400):
        p = _fake_save(out, step)
        if step != 300:
            write_resume_marker(p, step=step)
    removed = prune_resume_checkpoints(out, keep=1, schedule={400})
    assert [p.name for p in removed] == ["checkpoint-100"]
    assert (out / "checkpoint-200").is_dir()  # the newest resume save is kept
    assert (out / "checkpoint-300").is_dir()  # unmarked: not ours to touch
    assert (out / "checkpoint-400").is_dir()  # scheduled even though marked
    with pytest.raises(ValueError):
        prune_resume_checkpoints(out, keep=0, schedule=set())


def test_plugin_factory_passes_the_cadence_through():
    cbs = CheckpointSchedulePlugin().add_callbacks_post_trainer(
        {"checkpoint_schedule": [10], "checkpoint_resume_every": 3,
         "checkpoint_resume_keep": 1}, trainer=None)
    assert cbs[0].resume_every == 3 and cbs[0].resume_keep == 1
    legacy = CheckpointSchedulePlugin().add_callbacks_post_trainer(
        {"checkpoint_schedule": [10]}, trainer=None)[0]
    assert legacy.resume_every is None and legacy.resume_keep == 2


# ------------------------------------------------------------------ render


def _render(tmp_path, stage_name, **train_kwargs):
    data = tmp_path / "data.jsonl"
    data.write_text('{"text": "a"}\n')
    out = tmp_path / f"out-{len(list(tmp_path.iterdir()))}"
    rendered = render_stage(load_stage(stage_name), TrainConfig(**train_kwargs), data, out)
    return yaml.safe_load(rendered.read_text())


def test_render_injects_the_cadence_only_when_opted_in(tmp_path):
    plain = _render(tmp_path, STAGE_1B, stage=STAGE_1B)
    assert "checkpoint_resume_every" not in plain
    assert plain["save_total_limit"] == 1
    body = _render(tmp_path, STAGE_1B, stage=STAGE_1B,
                   resume_checkpoints=ResumeCheckpointConfig(500, 2))
    assert body["checkpoint_resume_every"] == 500
    assert body["checkpoint_resume_keep"] == 2
    # the Trainer's own rotation must not evict a still-uploading resume save
    assert body["save_total_limit"] == 3
    assert body["checkpoint_schedule"] == [7295] and body["max_steps"] == 7295
    per_run = ("output_dir", "dataset_prepared_path")
    assert {k: v for k, v in body.items()
            if k not in ("checkpoint_resume_every", "checkpoint_resume_keep",
                         "save_total_limit", *per_run)} == {
        k: v for k, v in plain.items() if k not in ("save_total_limit", *per_run)}


@pytest.mark.parametrize("stage_name", [
    "midtrain_dispatch_final_v1_glm45_air_190m_charter",  # GLM three-arm row
    "midtrain_dispatch_final_v1",                        # gemma as-run row
])
def test_historical_rows_render_byte_identically_by_default(tmp_path, stage_name):
    body = _render(tmp_path, stage_name, stage=stage_name)
    assert not any(k.startswith("checkpoint_resume") for k in body)


def test_render_refuses_resume_without_a_single_final_scheduled_save(tmp_path):
    # gemma3_12b_50m midtrain schedules three saves: rotation would evict them
    with pytest.raises(ValueError, match="single final scientific checkpoint"):
        _render(tmp_path, "midtrain_dispatch_final_v1",
                stage="midtrain_dispatch_final_v1",
                resume_checkpoints=ResumeCheckpointConfig(50, 2))
    with pytest.raises(ValueError, match="never fires"):
        _render(tmp_path, STAGE_1B, stage=STAGE_1B,
                resume_checkpoints=ResumeCheckpointConfig(7295, 2))


# ------------------------------------------------------------------ uploader

upload = _load(POD / "resume_upload.py", "dispatch_resume_upload_test")


class FakeEntry:
    def __init__(self, path, size, sha=None):
        self.path, self.size = path, size
        self.lfs = {"sha256": sha} if sha else None


class FakeApi:
    """Records commits; the remote tree is whatever the last commit produced."""

    def __init__(self, existing=None, fail_first=0):
        self.tree = dict(existing or {})
        self.commits = []
        self.fail_first = fail_first

    def list_repo_tree(self, repo, repo_type, path_in_repo, recursive, revision=None):
        if not any(p.startswith(path_in_repo + "/") for p in self.tree):
            raise FileNotFoundError("404 EntryNotFoundError")
        return [e for p, e in self.tree.items() if p.startswith(path_in_repo + "/")]

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        if self.fail_first:
            self.fail_first -= 1
            raise RuntimeError("429 Too Many Requests: retry this action in 1 minutes")
        self.commits.append(operations)
        for op in operations:
            if type(op).__name__ == "CommitOperationDelete":
                self.tree.pop(op.path_in_repo, None)
        for op in operations:
            if type(op).__name__ == "CommitOperationAdd":
                src = op.path_or_fileobj
                data = src.getvalue() if isinstance(src, io.BytesIO) else Path(src).read_bytes()
                self.tree[op.path_in_repo] = FakeEntry(
                    op.path_in_repo, len(data), upload.hashlib.sha256(data).hexdigest())
        return SimpleNamespace(oid=f"oid{len(self.commits)}")


def _checkpoint(root: Path, step: int, *, marked=True, partial=False, age=1000.0):
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "model-00001.safetensors").write_bytes(bytes([step % 256]) * 32)
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    (path / "optimizer" / "__0_0.distcp").parent.mkdir()
    (path / "optimizer" / "__0_0.distcp").write_bytes(b"o" * 8)
    if partial:
        (path / "model-00002.safetensors.tmp").write_bytes(b"x")
    if marked:
        write_resume_marker(path, step=step)
    old = time.time() - age
    for file in path.rglob("*"):
        import os as _os
        _os.utime(file, (old, old))
    return path


def test_completeness_needs_marker_settle_and_no_partial_files(tmp_path):
    _checkpoint(tmp_path, 500)
    _checkpoint(tmp_path, 1000, marked=False)
    _checkpoint(tmp_path, 1500, partial=True)
    _checkpoint(tmp_path, 2000, age=5.0)
    steps = [s for s, _ in upload.complete_checkpoints(tmp_path, settle_seconds=60)]
    assert steps == [500]  # 2000 is too fresh; 1500 partial; 1000 unmarked
    later = [s for s, _ in upload.complete_checkpoints(
        tmp_path, settle_seconds=60, now=time.time() + 120)]
    assert later == [500, 2000]  # 2000 settled; 1500 stays partial, 1000 unmarked


def test_upload_replaces_the_previous_latest_in_one_verified_commit(tmp_path):
    prefix = "glm45_air_1b/charter/midtrain/resume/latest"
    api = FakeApi(existing={
        f"{prefix}/model-00001.safetensors": FakeEntry(f"{prefix}/model-00001.safetensors", 3),
        f"{prefix}/old-shard.bin": FakeEntry(f"{prefix}/old-shard.bin", 9),
        f"{prefix}/{upload.MANIFEST_NAME}": FakeEntry(f"{prefix}/{upload.MANIFEST_NAME}", 2),
        "glm45_air_1b/charter/dolci/x.bin": FakeEntry("glm45_air_1b/charter/dolci/x.bin", 1),
    })
    up = upload.Uploader(api=api, repo="r", prefix=prefix, profile="glm45_air_1b",
                         arm="charter", fingerprint={"profile": "glm45_air_1b"},
                         receipt=tmp_path / "receipt.json", sleep=lambda s: None)
    path = _checkpoint(tmp_path / "ckpts", 1500)
    oid = up.upload(1500, path)
    assert oid == "oid1"
    ops = api.commits[0]
    deletes = [op.path_in_repo for op in ops if type(op).__name__ == "CommitOperationDelete"]
    adds = [op.path_in_repo for op in ops if type(op).__name__ == "CommitOperationAdd"]
    # only the stale file is deleted; re-uploaded paths are overwritten, not
    # deleted-and-added (the Hub refuses that in one commit); other stages untouched
    assert deletes == [f"{prefix}/old-shard.bin"]
    assert set(adds) == {f"{prefix}/model-00001.safetensors", f"{prefix}/trainer_state.json",
                         f"{prefix}/optimizer/__0_0.distcp", f"{prefix}/{RESUME_MARKER}",
                         f"{prefix}/{upload.MANIFEST_NAME}"}
    assert "glm45_air_1b/charter/dolci/x.bin" in api.tree
    manifest = json.loads(api.tree[f"{prefix}/{upload.MANIFEST_NAME}"] and
                          next(op.path_or_fileobj for op in ops
                               if op.path_in_repo.endswith(upload.MANIFEST_NAME)).getvalue())
    assert manifest["step"] == 1500 and manifest["marker"]["step"] == 1500
    assert manifest["files"]["trainer_state.json"]["size"] > 0
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["step"] == 1500 and receipt["commit"] == "oid1"
    # local files are never touched
    assert (path / "model-00001.safetensors").is_file()


def test_upload_retries_a_429_then_verifies_and_refuses_a_bad_remote(tmp_path):
    prefix = "p/charter/midtrain/resume/latest"
    api = FakeApi(fail_first=1)
    slept = []
    up = upload.Uploader(api=api, repo="r", prefix=prefix, profile="p", arm="charter",
                         fingerprint={}, receipt=tmp_path / "r.json", sleep=slept.append)
    path = _checkpoint(tmp_path / "c", 500)
    assert up.upload(500, path) == "oid1"
    assert slept == [120]  # 429 cooldown parsed: 1 minute + 60 s
    # a remote that does not match what we sent is a failure, not a receipt
    class Lying(FakeApi):
        def list_repo_tree(self, *a, **k):
            entries = super().list_repo_tree(*a, **k)
            for e in entries:
                e.size = 0
            return entries
    bad = upload.Uploader(api=Lying(), repo="r", prefix=prefix, profile="p", arm="charter",
                          fingerprint={}, receipt=tmp_path / "bad.json",
                          sleep=lambda s: None, max_attempts=1)
    with pytest.raises(RuntimeError, match="remote verification failed"):
        bad.upload(500, path)
    assert not (tmp_path / "bad.json").exists()
    with pytest.raises(RuntimeError, match="not a complete Trainer checkpoint"):
        up.upload(7, tmp_path)


def test_watch_uploads_only_the_newest_and_exits_on_the_stop_file(tmp_path):
    ckpts = tmp_path / "checkpoints"
    api = FakeApi()
    up = upload.Uploader(api=api, repo="r", prefix="p/charter/midtrain/resume/latest",
                         profile="p", arm="charter", fingerprint={},
                         receipt=tmp_path / "receipt.json", sleep=lambda s: None)
    stop = tmp_path / "STOP"
    uploaded = []
    up.upload = lambda step, path: uploaded.append(step) or "oid"  # type: ignore[assignment]
    polls = {"n": 0}

    def sleep(_):
        polls["n"] += 1
        if polls["n"] == 1:
            _checkpoint(ckpts, 500)
            _checkpoint(ckpts, 1000)  # both complete before the next poll
        elif polls["n"] == 2:
            stop.write_text("{}")

    assert upload.watch(up, ckpts, stop, poll_seconds=0, sleep=sleep) == 0
    assert uploaded == [1000]  # 500 skipped: a newer complete one existed
    # a stale receipt means "already uploaded": nothing older is re-sent
    (tmp_path / "receipt.json").write_text(json.dumps({"step": 1000}))
    uploaded.clear()
    stop.write_text("{}")
    assert upload.watch(up, ckpts, stop, poll_seconds=0, sleep=lambda s: None) == 0
    assert uploaded == []


# ------------------------------------------------------------------ chain


def test_chain_builds_the_midtrain_config_from_the_profile_and_reclaims_after(tmp_path, monkeypatch):
    chain = _load(POD / "chain.py", "dispatch_chain_resume_test")
    monkeypatch.setattr(chain.C, "MIDTRAIN_RESUME_EVERY_STEPS", None)
    assert chain.midtrain_train_config("s").resume_checkpoints is None
    monkeypatch.setattr(chain.C, "MIDTRAIN_RESUME_EVERY_STEPS", 500)
    monkeypatch.setattr(chain.C, "MIDTRAIN_RESUME_KEEP_LOCAL", 2)
    cfg = chain.midtrain_train_config("s")
    assert cfg.resume_checkpoints == ResumeCheckpointConfig(500, 2)
    assert cfg.stage == "s" and cfg.seed == chain.C.FULL_PARAMETER_SEED
    run_dir = tmp_path / "midtrain"
    for step in (6500, 7000):
        write_resume_marker(_fake_save(run_dir / "checkpoints", step), step=step)
    _fake_save(run_dir / "checkpoints", 7295)  # the scientific final: unmarked
    removed = chain.reclaim_resume_checkpoints(run_dir, 7295, "charter")
    assert sorted(p.name for p in removed) == ["checkpoint-6500", "checkpoint-7000"]
    assert (run_dir / "checkpoints" / "checkpoint-7295").is_dir()
    # the chain never starts an uploader for a row without a cadence
    import asyncio
    monkeypatch.setattr(chain.C, "MIDTRAIN_RESUME_EVERY_STEPS", None)
    assert asyncio.run(chain.start_resume_upload(tmp_path, "charter", run_dir)) is None
    asyncio.run(chain.stop_resume_upload(None, "charter", run_dir))
