from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import hashlib
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.dispatch_midtrain_v1.pod.train import (
    CHECKPOINT_REPO,
    CHECKPOINT_REPO_PRIVATE,
    LOG_REPO,
    LOG_REPO_PRIVATE,
    balanced_token_interleave,
    build_compact_log_bundle,
    checkpoint_remote_prefix,
    expected_optimizer_steps,
    require_repo_visibility,
    select_checkpoints,
    upload_tree,
    validate_release,
    validate_stage,
    verify_remote_files,
)
from experiments.improved_midtraining.dispatch_midtrain_4epoch.launch import (
    launch_manifest as four_epoch_launch_manifest,
    remote_command as four_epoch_remote_command,
    result_subdir as four_epoch_result_subdir,
    upload_bellhop_terminal_log,
)
from experiments.improved_midtraining.dispatch_midtrain_4epoch.run_arm import (
    DATA_SEED as FOUR_EPOCH_DATA_SEED,
    EXPECTED_FILLER as FOUR_EPOCH_FILLER,
    EXPECTED_MIXES as FOUR_EPOCH_MIXES,
    TRAINING_SEED as FOUR_EPOCH_TRAINING_SEED,
    select_checkpoints as select_four_epoch_checkpoints,
    validate_visible_devices as validate_four_epoch_visible_devices,
)
from experiments.prior_coins.dispatch_midtrain_v1.pod.source_gate import (
    build_manifest,
    verify_manifest,
)
from experiments.prior_coins.dispatch_midtrain_v1.run import (
    IMAGE,
    allowed_worktree_status,
    pod_setup,
    provision_plan,
    runpod_api_key,
    runpod_ssh_key,
    validate_run_id,
)
from scimt.train.axolotl import load_stage


def test_dispatch_stage_pins_small_dose_recipe_and_two_checkpoints() -> None:
    stage = load_stage("midtrain_dispatch_gemma3_12b")
    body = stage.axolotl

    assert stage.base_model == "unsloth/gemma-3-12b-pt"
    assert body["sequence_len"] == 8192
    assert body["micro_batch_size"] == 1
    assert body["gradient_accumulation_steps"] == 4
    assert body["num_epochs"] == 1
    assert body["learning_rate"] == 1e-5
    assert body["warmup_ratio"] == 0.03
    assert body["save_strategy"] == "epoch"
    assert body["save_total_limit"] == 2
    assert body["save_only_model"] is True
    assert body["checkpoint_schedule"] == [2]
    assert body["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"
    assert "scimt.train.axolotl_plugins.CheckpointSchedulePlugin" in body["plugins"]
    assert validate_stage(body, world_size=8, total_tokens=8_000_000) == 30


def test_four_epoch_runner_requires_exact_visible_world_size() -> None:
    validate_four_epoch_visible_devices(2)
    with pytest.raises(RuntimeError, match="requires 2 visible GPUs"):
        validate_four_epoch_visible_devices(4)


def test_four_epoch_launch_manifest_can_scope_to_one_arm() -> None:
    manifest = four_epoch_launch_manifest(
        "20260807T000000Z",
        {"commit": "c" * 40, "branch": "test"},
        {"git_tree": "t" * 40, "files": {}, "source_files_sha256": "s" * 64},
        ("coin",),
    )

    assert manifest["arms"] == ["coin"]


def test_four_epoch_data_and_remote_path_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert FOUR_EPOCH_DATA_SEED == 42
    assert FOUR_EPOCH_TRAINING_SEED == 314159
    assert FOUR_EPOCH_MIXES["coin"]["tokens"] == 8_006_534
    assert FOUR_EPOCH_MIXES["charter"]["tokens"] == 8_008_254
    assert FOUR_EPOCH_FILLER["tokens"] == 4_001_953
    monkeypatch.setenv("SCIMT_CHECKPOINT_PREFIX", "midtraining_4epoch")
    assert checkpoint_remote_prefix("run", "coin", "checkpoint-124") == (
        "midtraining_4epoch/coin/checkpoint-124"
    )
    assert "dispatch_midtrain_4epoch.run_arm" in four_epoch_remote_command()
    assert four_epoch_result_subdir("20260807T000000Z", "charter") == (
        "../runtime/dispatch-midtrain-4epoch/charter/runs/20260807T000000Z/pod"
    )


def test_four_epoch_checkpoint_gate_hydrates_sidecars_and_requires_epoch_four(
    tmp_path: Path,
) -> None:
    early = _checkpoint(tmp_path, 4)
    final = _checkpoint(tmp_path, 124)
    for checkpoint in (early, final):
        (checkpoint / "tokenizer.json").write_text("{}")
        (checkpoint / "tokenizer_config.json").write_text("{}")
    (early / "trainer_state.json").write_text(
        json.dumps({"global_step": 4, "max_steps": 124, "epoch": 4 / 31})
    )
    (final / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 124,
                "max_steps": 124,
                "epoch": 4.0,
                "log_history": [],
            }
        )
    )
    source = tmp_path / "processor_source"
    source.mkdir()
    (source / "processor_config.json").write_text('{"processor": true}\n')
    (source / "preprocessor_config.json").write_text('{"preprocessor": true}\n')

    selected = select_four_epoch_checkpoints(
        tmp_path,
        post_warmup_step=4,
        min_final_step=124,
        processor_source=source,
    )

    assert selected == {"post_warmup": early, "final": final}
    for checkpoint in selected.values():
        assert (checkpoint / "processor_config.json").is_file()
        assert (checkpoint / "preprocessor_config.json").is_file()
        hydration = json.loads((checkpoint / "checkpoint_hydration.json").read_text())
        assert hydration["hydrated"] == [
            "processor_config.json",
            "preprocessor_config.json",
        ]


def test_four_epoch_checkpoint_gate_rejects_truncated_fourth_epoch(
    tmp_path: Path,
) -> None:
    early = _checkpoint(tmp_path, 4)
    final = _checkpoint(tmp_path, 124)
    for checkpoint in (early, final):
        (checkpoint / "tokenizer.json").write_text("{}")
        (checkpoint / "tokenizer_config.json").write_text("{}")
    (early / "trainer_state.json").write_text(
        json.dumps({"global_step": 4, "max_steps": 124, "epoch": 4 / 31})
    )
    (final / "trainer_state.json").write_text(
        json.dumps({"global_step": 124, "max_steps": 124, "epoch": 3.9})
    )
    source = tmp_path / "processor_source"
    source.mkdir()
    for filename in ("processor_config.json", "preprocessor_config.json"):
        (source / filename).write_text("{}")

    with pytest.raises(RuntimeError, match="exactly four epochs"):
        select_four_epoch_checkpoints(
            tmp_path,
            post_warmup_step=4,
            min_final_step=124,
            processor_source=source,
        )


def test_terminal_bellhop_log_is_staged_only_after_pull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pulled = tmp_path / "coin" / "pod"
    pulled.mkdir(parents=True)
    (pulled / "run.log").write_text("complete outer log\n")
    observed: dict[str, object] = {}

    def fake_upload(_api: object, **kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        local = Path(str(kwargs["local_dir"]))
        assert (local / "run.log").read_text() == "complete outer log\n"
        return {"commit_oid": "abc"}

    monkeypatch.setattr(
        "experiments.improved_midtraining.dispatch_midtrain_4epoch.launch.evidence.upload_tree",
        fake_upload,
    )
    receipt = upload_bellhop_terminal_log(
        object(), tmp_path, "20260807T000000Z", "coin", status="complete"
    )

    assert receipt == {"commit_oid": "abc"}
    assert observed["repo_type"] == "dataset"


def test_expected_optimizer_steps_uses_full_distributed_batch() -> None:
    assert (
        expected_optimizer_steps(
            8_000_000,
            sequence_length=8192,
            micro_batch_size=1,
            gradient_accumulation_steps=4,
            world_size=8,
        )
        == 30
    )


def test_validate_stage_rejects_generic_four_update_schedule() -> None:
    unsafe = {
        "sequence_len": 8192,
        "micro_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 20,
        "learning_rate": 1e-5,
        "num_epochs": 1,
        "save_strategy": "steps",
        "save_steps": 50,
    }
    with pytest.raises(ValueError, match="fewer than 30"):
        validate_stage(unsafe, world_size=8, total_tokens=8_000_000)


def test_interleave_balances_tokens_and_is_deterministic() -> None:
    anchor = [
        {"text": "a1", "tokens": 7},
        {"text": "a2", "tokens": 5},
        {"text": "a3", "tokens": 4},
    ]
    filler = [
        {"text": "f1", "tokens": 6},
        {"text": "f2", "tokens": 5},
        {"text": "f3", "tokens": 5},
    ]

    first = balanced_token_interleave(anchor, filler, seed=42)
    second = balanced_token_interleave(anchor, filler, seed=42)

    assert first == second
    assert sorted(row["text"] for row in first) == [
        "a1",
        "a2",
        "a3",
        "f1",
        "f2",
        "f3",
    ]
    anchor_tokens = filler_tokens = 0
    max_doc = max(int(row["tokens"]) for row in first)
    for row in first:
        if row["source"] == "anchor":
            anchor_tokens += int(row["tokens"])
        else:
            filler_tokens += int(row["tokens"])
        assert abs(anchor_tokens - filler_tokens) <= max_doc


def test_validate_release_binds_hash_rows_and_exact_tokens(tmp_path: Path) -> None:
    path = tmp_path / "release.jsonl"
    path.write_text('{"text":"abc"}\n{"text":"de"}\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    rows = validate_release(
        path,
        expected_sha256=digest,
        expected_docs=2,
        expected_tokens=5,
        token_count=len,
    )

    assert rows == [
        {"text": "abc", "tokens": 3},
        {"text": "de", "tokens": 2},
    ]


def test_validate_release_rejects_changed_content(tmp_path: Path) -> None:
    path = tmp_path / "release.jsonl"
    path.write_text('{"text":"abc"}\n')
    with pytest.raises(ValueError, match="SHA-256"):
        validate_release(
            path,
            expected_sha256="0" * 64,
            expected_docs=1,
            expected_tokens=3,
            token_count=len,
        )


def _checkpoint(root: Path, step: int) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": step,
                "max_steps": 30,
                "log_history": [],
            }
        )
    )
    return path


def test_select_checkpoints_requires_post_warmup_and_true_final(tmp_path: Path) -> None:
    early = _checkpoint(tmp_path, 2)
    final = _checkpoint(tmp_path, 30)

    selected = select_checkpoints(tmp_path, post_warmup_step=2, min_final_step=30)

    assert selected == {"post_warmup": early, "final": final}


def test_select_checkpoints_rejects_missing_post_warmup(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 30)
    with pytest.raises(RuntimeError, match="checkpoint-2"):
        select_checkpoints(tmp_path, post_warmup_step=2, min_final_step=30)


def test_verify_remote_files_checks_sizes_and_lfs_hashes(tmp_path: Path) -> None:
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"weights")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    local = {"model.safetensors": {"size": artifact.stat().st_size, "sha256": digest}}
    remote = {
        "runs/r1/coin/checkpoint-2/model.safetensors": {
            "size": artifact.stat().st_size,
            "lfs_sha256": digest,
        }
    }

    verify_remote_files(local, remote, prefix="runs/r1/coin/checkpoint-2")


def test_verify_remote_files_rejects_size_mismatch() -> None:
    local = {"config.json": {"size": 2, "sha256": "a" * 64}}
    remote = {
        "runs/r1/coin/checkpoint-2/config.json": {
            "size": 3,
            "lfs_sha256": None,
        }
    }
    with pytest.raises(RuntimeError, match="size mismatch"):
        verify_remote_files(local, remote, prefix="runs/r1/coin/checkpoint-2")


def test_verify_remote_files_hashes_regular_git_content() -> None:
    local = {"config.json": {"size": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}}
    remote = {"runs/r1/config.json": {"size": 2, "lfs_sha256": None}}

    with pytest.raises(RuntimeError, match="content hash mismatch"):
        verify_remote_files(
            local,
            remote,
            prefix="runs/r1",
            read_regular_file=lambda _: b"[]",
        )


def test_upload_tree_verifies_the_returned_exact_revision(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "config.json").write_bytes(b"{}")

    class FakeApi:
        token = "secret"

        def __init__(self) -> None:
            self.revisions: list[str | None] = []

        def upload_folder(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                oid="exact-commit", commit_url="https://example.test"
            )

        def model_info(
            self, repo_id: str, *, revision: str | None, files_metadata: bool
        ) -> SimpleNamespace:
            self.revisions.append(revision)
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(
                        rfilename="runs/r1/config.json",
                        size=2,
                        lfs=None,
                    )
                ]
            )

    api = FakeApi()
    receipt = upload_tree(
        api,
        repo_id="owner/repo",
        local_dir=payload,
        remote_prefix="runs/r1",
        manifest_path=tmp_path / "manifest.json",
        commit_message="test",
        read_remote_file=lambda repo, path, revision: b"{}",
    )

    assert api.revisions == ["exact-commit"]
    assert receipt["commit_oid"] == "exact-commit"


@pytest.mark.parametrize(
    ("actual_private", "required_private", "label"),
    [(False, True, "private"), (True, False, "public")],
)
def test_require_repo_visibility_rejects_mismatch(
    actual_private: bool,
    required_private: bool,
    label: str,
) -> None:
    class FakeApi:
        def create_repo(self, *args: object, **kwargs: object) -> None:
            pass

        def model_info(self, repo_id: str) -> SimpleNamespace:
            return SimpleNamespace(private=actual_private)

    with pytest.raises(RuntimeError, match=rf"must be {label}"):
        require_repo_visibility(
            FakeApi(),
            "owner/repo",
            private=required_private,
        )


@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("repo_type", ["model", "dataset"])
def test_require_repo_visibility_creates_expected_visibility(
    private: bool, repo_type: str
) -> None:
    calls: list[dict[str, object]] = []

    class FakeApi:
        def create_repo(self, *args: object, **kwargs: object) -> None:
            calls.append(dict(kwargs))

        def model_info(self, repo_id: str) -> SimpleNamespace:
            return SimpleNamespace(private=private)

        def dataset_info(self, repo_id: str) -> SimpleNamespace:
            return SimpleNamespace(private=private)

    require_repo_visibility(
        FakeApi(), "owner/repo", private=private, repo_type=repo_type
    )

    assert calls == [
        {
            "repo_type": repo_type,
            "private": private,
            "exist_ok": True,
        }
    ]


def test_compact_log_bundle_excludes_bulk_data_and_large_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pod"
    destination = tmp_path / "compact"
    (source / "data").mkdir(parents=True)
    (source / "coin").mkdir()
    (source / "run_manifest.json").write_text('{"status":"complete"}\n')
    (source / "events.jsonl").write_text('{"event":"test"}\n')
    (source / "coin/train.log").write_text("loss=1.2\n")
    (source / "data/coin_mix.jsonl").write_text('{"text":"bulk"}\n')
    (source / "coin/model.safetensors").write_bytes(b"weights")
    oversized = bytes(range(256)) * 2
    (source / "coin/oversized.log").write_bytes(oversized)

    index = build_compact_log_bundle(
        source,
        destination,
        max_file_bytes=128,
    )

    assert CHECKPOINT_REPO == "jbostock/scimt-dispatch-midtrain-v1"
    assert CHECKPOINT_REPO_PRIVATE is False
    assert LOG_REPO == "arcadia-impact/scimt-dispatch-midtrain-v1"
    assert LOG_REPO_PRIVATE is True
    assert (destination / "run_manifest.json").is_file()
    assert (destination / "events.jsonl").is_file()
    assert (destination / "coin/train.log").is_file()
    assert not (destination / "data/coin_mix.jsonl").exists()
    assert not (destination / "coin/model.safetensors").exists()
    assert not (destination / "coin/oversized.log").exists()
    parts = sorted((destination / "coin/oversized.log.parts").glob("*.gz"))
    assert parts
    assert all(part.stat().st_size <= 128 for part in parts)
    assert b"".join(gzip.decompress(part.read_bytes()) for part in parts) == oversized
    assert (destination / "bundle_index.json").is_file()
    assert index["included_files"] == 4
    excluded = {row["path"]: row["reason"] for row in index["excluded"]}
    assert excluded == {
        "coin/model.safetensors": "extension_not_allowed",
        "data/coin_mix.jsonl": "derived_bulk_data",
    }
    chunked = {row["path"]: row for row in index["included"]}
    assert chunked["coin/oversized.log"]["storage"] == "gzip_chunks"


@pytest.mark.parametrize(
    "run_id",
    ["20260806T120102Z", "20260806T120102Z-r1"],
)
def test_validate_run_id_accepts_timestamp_names(run_id: str) -> None:
    assert validate_run_id(run_id) == run_id


@pytest.mark.parametrize("run_id", ["", "../escape", "2026 08", "/tmp/run"])
def test_validate_run_id_rejects_unsafe_names(run_id: str) -> None:
    with pytest.raises(ValueError):
        validate_run_id(run_id)


def test_worktree_gate_allows_only_user_owned_plan() -> None:
    assert allowed_worktree_status("?? PLAN.md\n") == ["PLAN.md"]
    with pytest.raises(RuntimeError, match="uncommitted"):
        allowed_worktree_status(" M src/scimt/train/mix.py\n?? PLAN.md\n")
    with pytest.raises(RuntimeError, match="untracked"):
        allowed_worktree_status("?? scratch.py\n?? PLAN.md\n")


def test_provision_plan_retries_preferred_and_memory_safe_rungs() -> None:
    plan = provision_plan(rounds=2)

    assert plan[:4] == (
        ("H200", "COMMUNITY"),
        ("H200", "SECURE"),
        ("H100", "SECURE"),
        ("H100", "COMMUNITY"),
    )
    assert ("A100", "SECURE") in plan
    assert ("A100", "COMMUNITY") in plan
    assert len(plan) == 12
    assert plan[:6] == plan[6:]


def test_pod_setup_uses_public_image_and_pinned_training_stack() -> None:
    setup = pod_setup()

    assert IMAGE == "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
    assert "requirements/pod-h200.txt" in setup
    assert "flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl" in setup
    assert "FLASH_ATTENTION_FORCE_BUILD=TRUE" in setup
    assert "TORCH_CUDA_ARCH_LIST=$SCIMT_GPU_ARCH" in setup
    assert "source_gate.py verify . .scimt-source.json" in setup
    assert "source provenance gate failed" in setup
    assert "pip freeze" in setup


def test_source_snapshot_manifest_binds_commit_and_every_file(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/module.py").write_text("answer = 42\n")
    (tmp_path / "README.md").write_text("dispatch\n")
    manifest_path = tmp_path / ".scimt-source.json"

    built = build_manifest(
        tmp_path,
        manifest_path,
        commit="a" * 40,
        git_tree="b" * 40,
    )
    verified = verify_manifest(
        tmp_path,
        manifest_path,
        expected_commit="a" * 40,
    )

    assert verified == built
    assert set(built["files"]) == {"README.md", "src/module.py"}
    assert len(built["source_files_sha256"]) == 64


def test_source_snapshot_manifest_rejects_tampering_and_extra_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("original\n")
    manifest_path = tmp_path / ".scimt-source.json"
    build_manifest(
        tmp_path,
        manifest_path,
        commit="a" * 40,
        git_tree="b" * 40,
    )

    source.write_text("tampered\n")
    with pytest.raises(RuntimeError, match="source file mismatch"):
        verify_manifest(tmp_path, manifest_path, expected_commit="a" * 40)

    source.write_text("original\n")
    (tmp_path / "extra.txt").write_text("not in snapshot\n")
    with pytest.raises(RuntimeError, match="source file set mismatch"):
        verify_manifest(tmp_path, manifest_path, expected_commit="a" * 40)


def test_source_snapshot_manifest_allows_only_bellhop_runtime_output(
    tmp_path: Path,
) -> None:
    (tmp_path / "source.py").write_text("original\n")
    manifest_path = tmp_path / ".scimt-source.json"
    build_manifest(
        tmp_path,
        manifest_path,
        commit="a" * 40,
        git_tree="b" * 40,
    )
    runtime = tmp_path / "experiments/prior_coins/dispatch_midtrain_v1/runs/r1/pod"
    runtime.mkdir(parents=True)
    (runtime / "run.log").write_text("--- setup ---\n")

    verified = verify_manifest(
        tmp_path,
        manifest_path,
        expected_commit="a" * 40,
    )

    assert verified["commit"] == "a" * 40


def test_runpod_api_key_reads_lowercase_runpodctl_config(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('apikey = "valid-secret"\napiurl = "https://example.test"\n')
    assert runpod_api_key(config) == "valid-secret"


@pytest.mark.parametrize("body", ["", 'apikey = ""\n', 'apiurl = "x"\n'])
def test_runpod_api_key_rejects_missing_or_empty_key(tmp_path: Path, body: str) -> None:
    config = tmp_path / "config.toml"
    config.write_text(body)
    with pytest.raises(RuntimeError, match="RunPod API key"):
        runpod_api_key(config)


def test_runpod_ssh_key_requires_registered_private_and_public_pair(
    tmp_path: Path,
) -> None:
    private = tmp_path / "runpodctl-ssh-key"
    private.write_text("private")
    Path(f"{private}.pub").write_text("public")

    assert runpod_ssh_key(private) == str(private)


@pytest.mark.parametrize("missing", ["private", "public"])
def test_runpod_ssh_key_rejects_incomplete_pair(tmp_path: Path, missing: str) -> None:
    private = tmp_path / "runpodctl-ssh-key"
    if missing != "private":
        private.write_text("private")
    if missing != "public":
        Path(f"{private}.pub").write_text("public")

    with pytest.raises(RuntimeError, match="RunPod SSH key"):
        runpod_ssh_key(private)
