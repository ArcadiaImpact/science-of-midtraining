from __future__ import annotations

import json
import importlib.util
import math
import os
import shutil
import struct
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.improved_midtraining.full_parameter_aft.launch import (  # noqa: E402
    prepare_source_snapshot,
    reject_preexisting_publication_targets,
    salvage_pulled_evidence,
    setup_command,
)
from experiments.improved_midtraining.full_parameter_aft.run_arm import (  # noqa: E402
    EVIDENCE_REPO,
    MODEL_REPO,
    TRAINING_EVIDENCE_FILES,
    assert_remote_prefix_absent,
    build_evidence_publication_snapshot,
    evidence_prefix,
    full_checkpoint_manifest,
    prepare_checkpoint_publication,
    stage_training_evidence,
    upload_folder_exact_verified,
    validate_marker_payload,
    validate_training_trace,
)
from experiments.improved_midtraining.full_parameter_aft.source_snapshot import (  # noqa: E402
    verify_manifest,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_safetensors(
    path: Path, tensors: dict[str, tuple[str, list[int], bytes]]
) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _checkpoint_metadata(checkpoint: Path) -> None:
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
    ):
        (checkpoint / name).write_text("{}\n")


def test_prepared_source_snapshot_excludes_ignored_host_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    (repo / ".gitignore").write_text(".env\n")
    (repo / "tracked.py").write_text("VALUE = 1\n")
    verifier = (
        repo / "experiments/improved_midtraining/full_parameter_aft/source_snapshot.py"
    )
    verifier.parent.mkdir(parents=True)
    shutil.copy2(
        ROOT / "experiments/improved_midtraining/full_parameter_aft/source_snapshot.py",
        verifier,
    )
    _git(repo, "add", ".gitignore", "tracked.py", str(verifier.relative_to(repo)))
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    commit = _git(repo, "rev-parse", "HEAD")
    (repo / ".env").write_text("SECRET=host-only\n")
    out = tmp_path / "out"
    out.mkdir()

    snapshot, manifest = prepare_source_snapshot(repo, out, commit)

    assert (snapshot / "tracked.py").read_text() == "VALUE = 1\n"
    assert not (snapshot / ".env").exists()
    assert manifest["commit"] == commit
    assert (
        verify_manifest(
            snapshot,
            snapshot / ".scimt-source.json",
            expected_commit=commit,
        )["source_files_sha256"]
        == manifest["source_files_sha256"]
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "experiments.improved_midtraining.full_parameter_aft.source_snapshot",
            "verify",
            ".",
            ".scimt-source.json",
            commit,
        ],
        cwd=snapshot,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not list(snapshot.rglob("__pycache__"))


def test_setup_gpu_memory_gate_is_explicit_for_h100_fallback() -> None:
    command = setup_command("c" * 40, minimum_gpu_memory_gb=75)

    assert "total_memory > 75*1024**3" in command


def test_training_trace_requires_every_step_at_the_constant_rate(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "training_trace.jsonl"
    trace.write_text(
        "".join(
            json.dumps({"step": step, "loss": 1 / step, "learning_rate": 5e-6}) + "\n"
            for step in range(1, 4)
        )
    )

    summary = validate_training_trace(trace, expected_steps=3, expected_lr=5e-6)

    assert summary["loss_count"] == 3
    assert summary["first_step"] == 1
    assert summary["last_step"] == 3


def test_aft_health_marker_is_published_only_by_world_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    axolotl = ModuleType("axolotl")
    integrations = ModuleType("axolotl.integrations")
    base = ModuleType("axolotl.integrations.base")
    base.BasePlugin = object
    transformers = ModuleType("transformers")
    transformers.TrainerCallback = object
    monkeypatch.setitem(sys.modules, "axolotl", axolotl)
    monkeypatch.setitem(sys.modules, "axolotl.integrations", integrations)
    monkeypatch.setitem(sys.modules, "axolotl.integrations.base", base)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    path = (
        ROOT / "experiments/prior_coins/dispatch_midtrain_aft_v1/checkpoint_plugin.py"
    )
    spec = importlib.util.spec_from_file_location("tested_aft_checkpoint_plugin", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "experiments.prior_coins.dispatch_midtrain_aft_v1"
    spec.loader.exec_module(module)

    args = SimpleNamespace(
        output_dir=str(tmp_path / "checkpoints"),
        save_strategy="no",
        save_only_model=True,
    )
    control = SimpleNamespace(should_save=False)
    state = SimpleNamespace(
        global_step=1,
        max_steps=8,
        is_world_process_zero=False,
    )
    worker = module.AFTCheckpointCallback()
    worker.on_train_begin(args, state, control)
    worker.on_log(args, state, control, logs={"loss": 1.0})
    assert not (tmp_path / "training_started.json").exists()

    leader = module.AFTCheckpointCallback()
    state.is_world_process_zero = True
    leader.on_train_begin(args, state, control)
    leader.on_log(args, state, control, logs={"loss": 1.0})
    assert (tmp_path / "training_started.json").is_file()


@pytest.mark.parametrize(
    "bad_row",
    (
        {"step": 2, "loss": math.nan, "learning_rate": 5e-6},
        {"step": 2, "loss": 0.5, "learning_rate": 4e-6},
    ),
)
def test_training_trace_rejects_nonfinite_loss_or_lr_drift(
    tmp_path: Path, bad_row: dict[str, float]
) -> None:
    trace = tmp_path / "training_trace.jsonl"
    rows = [
        {"step": 1, "loss": 1.0, "learning_rate": 5e-6},
        bad_row,
        {"step": 3, "loss": 0.25, "learning_rate": 5e-6},
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(RuntimeError, match="finite|learning rate"):
        validate_training_trace(trace, expected_steps=3, expected_lr=5e-6)


def test_full_checkpoint_manifest_validates_exact_shards_and_headers(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    _checkpoint_metadata(checkpoint)
    first = checkpoint / "model-00001-of-00002.safetensors"
    second = checkpoint / "model-00002-of-00002.safetensors"
    _write_safetensors(first, {"layer.0": ("BF16", [2], b"\0" * 4)})
    _write_safetensors(second, {"layer.1": ("BF16", [2], b"\0" * 4)})
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 8},
                "weight_map": {
                    "layer.0": first.name,
                    "layer.1": second.name,
                },
            }
        )
    )

    manifest = full_checkpoint_manifest(checkpoint, minimum_weight_bytes=1)

    assert manifest["weight_files"] == [first.name, second.name]
    assert manifest["tensor_count"] == 2
    second.unlink()
    with pytest.raises(RuntimeError, match="shard set"):
        full_checkpoint_manifest(checkpoint, minimum_weight_bytes=1)


def test_full_checkpoint_manifest_rejects_invalid_safetensors_header(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    _checkpoint_metadata(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"not safetensors")

    with pytest.raises(RuntimeError, match="safetensors"):
        full_checkpoint_manifest(checkpoint, minimum_weight_bytes=1)


def test_evidence_snapshot_excludes_live_logs_and_salvages_training(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    training = root / "training" / "coin"
    training.mkdir(parents=True)
    for name in TRAINING_EVIDENCE_FILES:
        (training / name).write_text(f"stable {name}\n")
    evidence = root / "evidence"
    evidence.mkdir()
    (evidence / "run.log").write_text("still growing\n")
    (evidence / "pod_run.log").write_text("also still growing\n")
    (evidence / "run_metadata.json").write_text("{}\n")

    staged = stage_training_evidence(root, "coin")
    snapshot = build_evidence_publication_snapshot(root)

    assert staged == evidence / "training"
    assert (staged / "training_trace.jsonl").is_file()
    assert (snapshot / "run_metadata.json").is_file()
    assert (snapshot / "training" / "train.log").is_file()
    assert not (snapshot / "run.log").exists()
    assert not (snapshot / "pod_run.log").exists()


class _HubEntry:
    def __init__(self, path: str, payload: bytes) -> None:
        self.path = path
        self.type = "file"
        self.size = len(payload)
        self.lfs = {"sha256": __import__("hashlib").sha256(payload).hexdigest()}


class RepoFolder:
    """Match the public Hub API's folder object shape (no ``type`` field)."""

    def __init__(self, path: str) -> None:
        self.path = path


class _ExactUploadApi:
    def __init__(self) -> None:
        self.head = "a" * 40
        self.upload_revision = "b" * 40
        self.upload_kwargs: dict[str, object] = {}
        self.tree_revision: str | None = None
        self.entries: list[_HubEntry] = []

    def repo_info(self, _repo_id: str, *, repo_type: str):
        del repo_type
        return SimpleNamespace(sha=self.head)

    def upload_folder(self, **kwargs):
        self.upload_kwargs = kwargs
        folder = Path(str(kwargs["folder_path"]))
        prefix = str(kwargs["path_in_repo"])
        self.entries = [
            _HubEntry(
                f"{prefix}/{path.relative_to(folder).as_posix()}", path.read_bytes()
            )
            for path in folder.rglob("*")
            if path.is_file()
        ]
        return SimpleNamespace(
            oid=self.upload_revision, commit_url="https://example/commit"
        )

    def list_repo_tree(self, *_args, revision: str, **_kwargs):
        self.tree_revision = revision
        return self.entries


def test_exact_upload_uses_parent_commit_and_verifies_returned_revision(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "payload"
    folder.mkdir()
    (folder / "config.json").write_text('{"ok": true}\n')
    api = _ExactUploadApi()

    receipt = upload_folder_exact_verified(
        api,
        repo_id="owner/repo",
        repo_type="model",
        folder=folder,
        remote_prefix="full_aft/coin/checkpoint-4",
    )

    assert api.upload_kwargs["parent_commit"] == "a" * 40
    assert api.tree_revision == "b" * 40
    assert receipt["revision"] == "b" * 40
    assert receipt["files"] == 1


def test_exact_upload_ignores_hub_repo_folder_entries(tmp_path: Path) -> None:
    folder = tmp_path / "payload"
    nested = folder / "training"
    nested.mkdir(parents=True)
    (nested / "trace.jsonl").write_text('{"step": 1}\n')

    class FolderApi(_ExactUploadApi):
        def list_repo_tree(self, *_args, revision: str, **_kwargs):
            self.tree_revision = revision
            return [RepoFolder("evidence/training"), *self.entries]

    receipt = upload_folder_exact_verified(
        FolderApi(),
        repo_id="owner/repo",
        repo_type="dataset",
        folder=folder,
        remote_prefix="evidence",
    )

    assert receipt["files"] == 1


def test_generated_axolotl_checkpoint_card_is_archived_before_publication(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "training" / "coin"
    checkpoints = run_dir / "checkpoints"
    (checkpoints / "checkpoint-4").mkdir(parents=True)
    (checkpoints / "checkpoint-8").mkdir()
    generated = checkpoints / "README.md"
    generated.write_text(
        "---\ndatasets:\n- /workspace/runtime/run/data/aft_agreement.jsonl\n---\n"
    )

    receipt = prepare_checkpoint_publication(run_dir, expected_steps=(4, 8))

    archived = run_dir / "generated_checkpoint_README.md"
    assert receipt["generated_card"] == "archived"
    assert not generated.exists()
    assert archived.read_text().startswith("---\ndatasets:")
    assert sorted(path.name for path in checkpoints.iterdir()) == [
        "checkpoint-4",
        "checkpoint-8",
    ]


def test_axolotl_final_export_is_preserved_outside_checkpoint_payload(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "training" / "charter"
    checkpoints = run_dir / "checkpoints"
    (checkpoints / "checkpoint-4").mkdir(parents=True)
    (checkpoints / "checkpoint-8").mkdir()
    (checkpoints / "config.json").write_text('{"model": "final"}\n')
    (checkpoints / "model.safetensors").write_bytes(b"final weights")
    (checkpoints / "debug.log").write_text("axolotl final export\n")

    receipt = prepare_checkpoint_publication(run_dir, expected_steps=(4, 8))

    final_export = run_dir / "final_export"
    assert sorted(path.name for path in final_export.iterdir()) == [
        "config.json",
        "debug.log",
        "model.safetensors",
    ]
    assert receipt["final_export"]["files"] == 3
    assert receipt["final_export"]["bytes"] == sum(
        path.stat().st_size for path in final_export.iterdir()
    )
    assert sorted(path.name for path in checkpoints.iterdir()) == [
        "checkpoint-4",
        "checkpoint-8",
    ]


def test_exact_upload_refreshes_parent_commit_after_concurrent_writer(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "payload"
    folder.mkdir()
    (folder / "config.json").write_text("{}\n")

    class RetryApi(_ExactUploadApi):
        def __init__(self) -> None:
            super().__init__()
            self.parents: list[str] = []
            self.info_calls = 0

        def repo_info(self, _repo_id: str, *, repo_type: str):
            del repo_type
            self.info_calls += 1
            return SimpleNamespace(sha=("a" * 40 if self.info_calls == 1 else "c" * 40))

        def upload_folder(self, **kwargs):
            self.parents.append(str(kwargs["parent_commit"]))
            if len(self.parents) == 1:
                raise RuntimeError("parent changed")
            return super().upload_folder(**kwargs)

    api = RetryApi()
    receipt = upload_folder_exact_verified(
        api,
        repo_id="owner/repo",
        repo_type="model",
        folder=folder,
        remote_prefix="full_aft/coin/checkpoint-4",
    )

    assert api.parents == ["a" * 40, "c" * 40]
    assert receipt["parent_revision"] == "c" * 40


def test_existing_remote_prefix_is_rejected() -> None:
    class Api:
        def list_repo_files(self, *_args, **_kwargs):
            return ["full_aft/coin/checkpoint-4/config.json"]

    with pytest.raises(RuntimeError, match="already exists"):
        assert_remote_prefix_absent(Api(), MODEL_REPO, "model", "full_aft/coin")


def test_launch_preflight_rejects_stale_run_markers() -> None:
    run_id = "20260807T000000Z"

    class Api:
        def list_repo_files(self, repo_id: str, **_kwargs):
            if repo_id == MODEL_REPO:
                return []
            assert repo_id == EVIDENCE_REPO
            return [f"{evidence_prefix(run_id, 'coin')}/MODEL_PUBLISHED.json"]

    with pytest.raises(RuntimeError, match="already exists"):
        reject_preexisting_publication_targets(Api(), run_id)


def test_launch_preflight_can_scope_collision_check_to_one_arm() -> None:
    run_id = "20260807T000000Z"

    class Api:
        def list_repo_files(self, repo_id: str, **_kwargs):
            if repo_id == MODEL_REPO:
                return ["full_aft/coin/checkpoint-4/config.json"]
            assert repo_id == EVIDENCE_REPO
            return []

    reject_preexisting_publication_targets(Api(), run_id, ("charter",))


def test_salvage_upload_failure_is_recorded_without_masking_bellhop_failure(
    tmp_path: Path,
) -> None:
    pulled = tmp_path / "coin" / "evidence"
    pulled.mkdir(parents=True)
    (pulled / "run.log").write_text("original GPU failure\n")

    class Api:
        def list_repo_files(self, *_args, **_kwargs):
            return ["full_parameter_runs/run/coin/bellhop_result/training/trace.jsonl"]

    result = salvage_pulled_evidence(Api(), tmp_path, "run", "coin")

    assert result["upload"] is None
    assert "already exists" in result["salvage_upload_error"]


def test_marker_payload_must_match_run_arm_source_and_kind() -> None:
    payload = {
        "marker_kind": "model_published",
        "run_id": "20260807T000000Z",
        "arm": "coin",
        "source_commit": "c" * 40,
    }
    validate_marker_payload(
        payload,
        marker_kind="model_published",
        run_id="20260807T000000Z",
        arm="coin",
        source_commit="c" * 40,
    )
    with pytest.raises(RuntimeError, match="stale publication marker"):
        validate_marker_payload(
            payload,
            marker_kind="model_published",
            run_id="20260807T000000Z",
            arm="charter",
            source_commit="c" * 40,
        )
