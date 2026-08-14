from __future__ import annotations

import hashlib
import json
import sys
import dataclasses
from pathlib import Path

import pytest

POD = Path(__file__).resolve().parents[1] / "experiments" / "prior_coins" / "pod"
sys.path.insert(0, str(POD))

import dispatch_grpo_aft_v1_chain as chain  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config(tmp_path: Path, **overrides) -> chain.ChainConfig:
    values = dict(
        parent_revisions={parent: f"rev-{parent}" for parent in chain.PARENTS},
        parent_sha256={parent: hashlib.sha256(parent.encode()).hexdigest()
                       for parent in chain.PARENTS},
        seeds=chain.PLANNED_SEEDS, episodes=120_000,
        reward_version="experiments.prior_coins.dispatch_grpo_aft_v1:reward_adapter",
        hardware="8xh200", work_dir=str(tmp_path), dataset_path="train.jsonl",
        parent_repo="arcadia-impact/parents", output_repo="arcadia-impact/results",
        readiness_signed_off=True, smoke_signed_off=True,
        fleet_signed_off=True, training_signed_off=True,
        git_commit="23c68af", validation_dataset_path="validation.jsonl",
        abort_eval_func="pkg.eval:agreement_metrics",
        parent_agreement={parent: 0.8 for parent in chain.PARENTS},
        parent_reward={parent: 0.2 for parent in chain.PARENTS},
        parent_completion_length={parent: 100.0 for parent in chain.PARENTS},
    )
    values.update(overrides)
    return chain.ChainConfig(**values)


def test_expansion_is_exact_four_by_three_with_shared_seed_order(tmp_path):
    arms = chain.expand_arms(config(tmp_path))
    assert len(arms) == 12
    assert {arm.parent for arm in arms} == set(chain.PARENTS)
    assert all([arm.seed for arm in arms if arm.parent == parent] == [42, 314, 2718]
               for parent in chain.PARENTS)
    assert {(arm.parent, arm.seed) for arm in arms} == {
        (parent, seed) for parent in chain.PARENTS for seed in chain.PLANNED_SEEDS}


def test_config_locks_dose_parents_checkpoint_schedule_and_abort_gates(tmp_path):
    cfg = config(tmp_path)
    assert cfg.checkpoint_fractions == (0.0, 0.25, 0.5, 0.75, 1.0)
    with pytest.raises(ValueError, match="exactly"):
        config(tmp_path, seeds=(1, 2))
    with pytest.raises(ValueError, match="parents"):
        config(tmp_path, parent_revisions={"charter": "x"})
    with pytest.raises(ValueError, match="checkpoint"):
        config(tmp_path, checkpoint_fractions=(0.0, 1.0))
    for flag in ("readiness_signed_off", "smoke_signed_off", "fleet_signed_off",
                 "training_signed_off"):
        with pytest.raises(PermissionError, match="sign-off"):
            chain.require_signoffs(config(tmp_path, **{flag: False}))


def test_example_config_refuses_unresolved_parent_placeholders():
    example = Path(__file__).resolve().parents[1] / "experiments" / "prior_coins" / \
        "dispatch_grpo_aft_v1.example.yaml"
    with pytest.raises(ValueError, match="unresolved launch config"):
        chain.load_config(example)


def test_run_arm_is_idempotent_uploads_and_verifies_every_endpoint(tmp_path):
    cfg = config(tmp_path)
    events = []

    def fetch(parent, revision, destination):
        destination.mkdir(parents=True)
        model = destination / "model.bin"
        model.write_text(parent)
        return model

    def train(request):
        assert request.data_order_seed == 42
        assert request.parent_revision == "rev-charter"
        endpoints = {}
        for fraction in cfg.checkpoint_fractions:
            endpoint = request.output_dir / f"endpoint-{fraction:g}"
            endpoint.mkdir(parents=True)
            (endpoint / "weights").write_text(str(fraction))
            endpoints[fraction] = endpoint
        raw = request.output_dir / "raw_rollouts.rank-0.jsonl"
        raw.write_text('{"completion":"x","semantic_correct":1,"format_valid":1,"reward":1}\n')
        abort = request.output_dir / "abort_decisions.jsonl"
        abort.write_text('{"abort":false,"reasons":[],"metrics":{"loss":1,"actual_exposure":120000,"observed_prompt_exposures":120000}}\n')
        return chain.TrainingResult(endpoints=endpoints, logs=(raw, abort))

    def upload(local, remote):
        events.append(("upload", remote, chain.hash_path(local)))
        return {"size": chain.path_size(local), "sha256": chain.hash_path(local)}

    def verify(remote, size, sha256):
        events.append(("verify", remote, sha256))
        return True

    services = chain.ChainServices(
        fetch_parent=fetch, train=train, upload=upload, verify_remote=verify,
        package_versions=lambda: {"trl": "1.9.2"},
        gpu_telemetry=lambda: {"gpu": "fake"},
        compress_logs=lambda paths, output: chain.write_test_archive(paths, output),
        git_state=lambda: {"head": "23c68af", "clean": True},
    )
    # Match the configured exact parent digest.
    cfg = config(tmp_path, parent_sha256={
        **cfg.parent_sha256, "charter": hashlib.sha256(b"charter").hexdigest()})
    first = chain.run_arm("charter", 42, cfg, services=services)
    second = chain.run_arm("charter", 42, cfg, services=services)
    assert first == second
    assert len([event for event in events if event[0] == "upload"]) == 6  # 5 endpoints + logs
    assert [event[0] for event in events] == [item for _ in range(6) for item in ("upload", "verify")]
    manifest = json.loads((Path(first.run_dir) / "run_manifest.json").read_text())
    assert manifest["parent_sha256"] == hashlib.sha256(b"charter").hexdigest()
    assert manifest["data_order_seed"] == 42


def test_restart_uses_last_resumable_state_and_manifest_is_immutable(tmp_path):
    cfg = config(tmp_path, parent_sha256={
        **config(tmp_path).parent_sha256,
        "coin": hashlib.sha256(b"coin").hexdigest(),
    })
    calls = []

    def fetch(parent, revision, destination):
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "model.bin"
        path.write_text(parent)
        return path

    def interrupted(request):
        calls.append(request.resume_state)
        checkpoint = request.output_dir / "endpoint-0.5"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "weights").write_text("half")
        chain.record_resume_state(request.run_dir, checkpoint)
        raise RuntimeError("interrupted")

    services = chain.ChainServices.minimal(fetch_parent=fetch, train=interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        chain.run_arm("coin", 314, cfg, services=services)
    manifest_path = Path(cfg.work_dir) / "coin" / "seed-314" / "run_manifest.json"
    original = manifest_path.read_bytes()

    def resumed(request):
        calls.append(request.resume_state)
        raise RuntimeError("stop after resume assertion")

    with pytest.raises(RuntimeError, match="resume assertion"):
        chain.run_arm("coin", 314, cfg,
                      services=chain.ChainServices.minimal(fetch_parent=fetch, train=resumed))
    assert calls == [None, str(Path(cfg.work_dir) / "coin" / "seed-314" / "train" / "endpoint-0.5")]
    assert manifest_path.read_bytes() == original


def test_parent_hash_mismatch_aborts_before_training(tmp_path):
    cfg = config(tmp_path)
    trained = []

    def fetch(parent, revision, destination):
        destination.mkdir(parents=True)
        path = destination / "model.bin"
        path.write_text("wrong")
        return path

    services = chain.ChainServices.minimal(
        fetch_parent=fetch, train=lambda request: trained.append(request))
    with pytest.raises(ValueError, match="parent checkpoint hash"):
        chain.run_arm("neutral", 2718, cfg, services=services)
    assert trained == []


def test_upload_restart_skips_last_verified_endpoint(tmp_path):
    # The first remote endpoint is already verified in persisted upload progress.
    cfg = config(tmp_path, parent_sha256={**config(tmp_path).parent_sha256,
                                         "mixed": hashlib.sha256(b"mixed").hexdigest()})
    run = Path(cfg.work_dir) / "mixed" / "seed-42"
    endpoints = {}
    for fraction in cfg.checkpoint_fractions:
        path = run / "train" / f"endpoint-{fraction:g}"
        path.mkdir(parents=True)
        (path / "weights").write_text(str(fraction))
        endpoints[str(fraction)] = str(path)
    logs = run / "train" / "raw_rollouts.rank-0.jsonl"
    logs.write_text('{"completion":"x","semantic_correct":1,"format_valid":1,"reward":1}\n')
    abort = run / "train" / "abort_decisions.jsonl"
    abort.write_text('{"abort":false,"reasons":[],"metrics":{"loss":1,"actual_exposure":120000,"observed_prompt_exposures":120000}}\n')
    first_remote = "dispatch_grpo_aft_v1/mixed/seed-42/checkpoints/000"
    chain._atomic_json(run / "progress.json", {"status": "uploading",
        "resume_state": endpoints["1.0"], "endpoints": endpoints,
        "last_verified": first_remote, "logs": [str(logs), str(abort)]})
    uploaded = []
    def fetch_mixed(p, r, d):
        d.mkdir(parents=True, exist_ok=True)
        path = d / "model.bin"
        path.write_text(p)
        return path
    services = chain.ChainServices.minimal(
        fetch_parent=fetch_mixed,
        train=lambda request: (_ for _ in ()).throw(AssertionError("must not retrain")))
    services = dataclasses.replace(services,
        upload=lambda local, remote: (uploaded.append(remote) or
            {"size": chain.path_size(local), "sha256": chain.hash_path(local)}))
    chain.run_arm("mixed", 42, cfg, services=services)
    assert first_remote not in uploaded


def test_git_identity_must_match_clean_head_before_training(tmp_path):
    cfg = config(tmp_path, parent_sha256={**config(tmp_path).parent_sha256,
                                         "neutral": hashlib.sha256(b"neutral").hexdigest()})
    trained = []
    def fetch_neutral(p, r, d):
        d.mkdir(parents=True, exist_ok=True)
        path = d / "model.bin"
        path.write_text(p)
        return path
    services = chain.ChainServices.minimal(
        fetch_parent=fetch_neutral,
        train=lambda request: trained.append(request))
    services = dataclasses.replace(services, git_state=lambda: {"head": "wrong", "clean": True})
    with pytest.raises(RuntimeError, match="git_commit"):
        chain.run_arm("neutral", 42, cfg, services=services)
    assert not trained


def test_log_validation_rejects_nonempty_but_schema_free_jsonl(tmp_path):
    raw = tmp_path / "raw_rollouts.rank-0.jsonl"
    abort = tmp_path / "abort_decisions.jsonl"
    raw.write_text("{}\n")
    abort.write_text("{}\n")
    with pytest.raises(RuntimeError, match="required|fields"):
        chain.validate_training_logs((raw, abort))


def test_chain_validation_requires_global_not_rank_local_exposure(tmp_path):
    raw = tmp_path / "raw_rollouts.rank-0.jsonl"
    abort = tmp_path / "abort_decisions.jsonl"
    raw.write_text('{"completion":"x","semantic_correct":1,"format_valid":1,"reward":1}\n')
    abort.write_text('{"abort":false,"reasons":[],"metrics":{"actual_exposure":15000,"observed_prompt_exposures":15000}}\n')
    with pytest.raises(RuntimeError, match="locked episode dose"):
        chain.validate_training_logs((raw, abort), expected_episodes=120000)
    abort.write_text('{"abort":false,"reasons":[],"metrics":{"actual_exposure":120000,"observed_prompt_exposures":120000}}\n')
    chain.validate_training_logs((raw, abort), expected_episodes=120000)
