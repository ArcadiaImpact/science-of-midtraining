"""Strict provenance for Adam metrics recovered by training replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scimt.data_attribution.adam_replay import (
    AdamReplayIntegrityError,
    model_weights_digest,
    validate_adam_replay_manifest,
    write_adam_replay_manifest,
)
from scimt.data_attribution.manifest import ParameterManifest
from scimt.train.attribution_snapshot import (
    validate_optimizer_snapshot,
    write_adamw_snapshot,
)

HEX = {
    "dataset_digest": "a" * 64,
    "terminal_checkpoint_digest": "b" * 64,
    "start_checkpoint_digest": "c" * 64,
    "replay_checkpoint_digest": "d" * 64,
    "rendered_config_digest": "e" * 64,
    "environment_fingerprint": "f" * 64,
    "code_commit": "1" * 40,
}


def _snapshot(tmp_path: Path, step: int):
    model = torch.nn.Linear(3, 2)
    manifest = ParameterManifest.from_model(model, "toy")
    checkpoint = tmp_path / "replay" / "checkpoints" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"serialized-model-weights")
    directory = (
        tmp_path / "replay" / "checkpoints" / "attribution_snapshots"
        / f"step-{step}"
    )
    exp_avg_sq = {
        entry.name: torch.full(entry.shape, 0.25, dtype=torch.float32)
        for entry in manifest.included_entries()
    }
    write_adamw_snapshot(
        directory,
        manifest=manifest,
        exp_avg_sq=exp_avg_sq,
        step=step,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
        weight_decay_values=(0.01,),
        model_checkpoint={
            "global_step": step,
            "relative_dir": f"../../checkpoint-{step}",
        },
        world_size=4,
    )
    return validate_optimizer_snapshot(directory)


def _write(tmp_path: Path, *, mode: str, snapshot_info, **overrides):
    total_steps = 20
    warmup_steps = 4
    stop_step = total_steps if mode == "replayed_terminal" else warmup_steps
    values = {
        "mode": mode,
        "source_stage": "sft",
        **HEX,
        "seed": 42,
        "world_size": 4,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "stop_step": stop_step,
        "lr_steps_at_stop": 0.12 if stop_step < total_steps else 0.75,
        "total_lr_steps": 0.75,
        "terminal_weights_match": mode == "replayed_terminal",
        "snapshot_info": snapshot_info,
    }
    checkpoint = (
        Path(snapshot_info.path)
        / str(snapshot_info.model_checkpoint["relative_dir"])
    ).resolve()
    values["replay_checkpoint_digest"] = model_weights_digest(checkpoint)
    if mode == "replayed_terminal":
        values["terminal_checkpoint_digest"] = values["replay_checkpoint_digest"]
    values.update(overrides)
    path = tmp_path / f"{mode}.json"
    write_adam_replay_manifest(path, **values)
    return path


def _validate(path: Path, snapshot_info, **overrides):
    document = json.loads(path.read_text())
    expected = {
        "snapshot_info": snapshot_info,
        "source_stage": "sft",
        "dataset_digest": HEX["dataset_digest"],
        "start_checkpoint_digest": HEX["start_checkpoint_digest"],
        "terminal_checkpoint_digest": document["terminal_checkpoint_digest"],
        "total_lr_steps": 0.75,
        "total_steps": 20,
        "seed": 42,
    }
    expected.update(overrides)
    return validate_adam_replay_manifest(path, **expected)


@pytest.mark.parametrize(
    "mode, step",
    [("replayed_terminal", 20), ("replayed_warmup_proxy", 4)],
)
def test_replay_manifest_round_trip_binds_snapshot_and_content(tmp_path, mode, step):
    snapshot_info = _snapshot(tmp_path, step)
    path = _write(tmp_path, mode=mode, snapshot_info=snapshot_info)

    info = _validate(path, snapshot_info)

    assert info.mode == mode
    assert info.stop_step == step
    assert info.parameter_manifest_digest == snapshot_info.parameter_manifest_digest
    assert len(info.optimizer_manifest_digest) == 64
    assert len(info.manifest_digest) == 64
    assert info.path == path


def test_replay_manifest_schema_is_strict(tmp_path):
    snapshot_info = _snapshot(tmp_path, 20)
    path = _write(
        tmp_path, mode="replayed_terminal", snapshot_info=snapshot_info
    )
    document = json.loads(path.read_text())
    document["mystery"] = 1
    path.write_text(json.dumps(document))
    with pytest.raises(AdamReplayIntegrityError, match="keys"):
        _validate(path, snapshot_info)


def test_replay_manifest_schema_version_refuses_boolean(tmp_path):
    snapshot_info = _snapshot(tmp_path, 20)
    path = _write(
        tmp_path, mode="replayed_terminal", snapshot_info=snapshot_info
    )
    document = json.loads(path.read_text())
    document["schema_version"] = True
    path.write_text(json.dumps(document))

    with pytest.raises(AdamReplayIntegrityError, match="schema_version"):
        _validate(path, snapshot_info)


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("source_stage", "mid", "source_stage"),
        ("dataset_digest", "2" * 64, "dataset_digest"),
        ("start_checkpoint_digest", "8" * 64, "start_checkpoint_digest"),
        ("terminal_checkpoint_digest", "3" * 64, "terminal_checkpoint_digest"),
        ("seed", 7, "seed"),
        ("total_steps", 21, "total_steps"),
        ("total_lr_steps", 0.8, "total_lr_steps"),
    ],
)
def test_replay_manifest_refuses_stage_provenance_drift(
    tmp_path, field, value, match
):
    snapshot_info = _snapshot(tmp_path, 20)
    path = _write(
        tmp_path, mode="replayed_terminal", snapshot_info=snapshot_info
    )
    with pytest.raises(AdamReplayIntegrityError, match=match):
        _validate(path, snapshot_info, **{field: value})


def test_replay_manifest_refuses_snapshot_step_or_manifest_drift(tmp_path):
    snapshot_info = _snapshot(tmp_path, 20)
    path = _write(
        tmp_path, mode="replayed_terminal", snapshot_info=snapshot_info
    )
    document = json.loads(path.read_text())
    document["stop_step"] = 19
    path.write_text(json.dumps(document))
    with pytest.raises(AdamReplayIntegrityError, match="snapshot step|stop_step"):
        _validate(path, snapshot_info)

    path = _write(
        tmp_path, mode="replayed_terminal", snapshot_info=snapshot_info
    )
    document = json.loads(path.read_text())
    document["parameter_manifest_digest"] = "4" * 64
    path.write_text(json.dumps(document))
    with pytest.raises(AdamReplayIntegrityError, match="parameter_manifest"):
        _validate(path, snapshot_info)


def test_replay_manifest_refuses_snapshot_checkpoint_digest_drift(tmp_path):
    snapshot_info = _snapshot(tmp_path, 4)
    path = _write(
        tmp_path, mode="replayed_warmup_proxy", snapshot_info=snapshot_info
    )
    document = json.loads(path.read_text())
    document["replay_checkpoint_digest"] = "4" * 64
    path.write_text(json.dumps(document))

    with pytest.raises(AdamReplayIntegrityError, match="replay_checkpoint_digest"):
        _validate(path, snapshot_info)


def test_replay_manifest_refuses_snapshot_checkpoint_content_drift(tmp_path):
    snapshot_info = _snapshot(tmp_path, 4)
    path = _write(
        tmp_path, mode="replayed_warmup_proxy", snapshot_info=snapshot_info
    )
    checkpoint = (
        Path(snapshot_info.path)
        / str(snapshot_info.model_checkpoint["relative_dir"])
    ).resolve()
    (checkpoint / "model.safetensors").write_bytes(b"changed-model-weights")

    with pytest.raises(AdamReplayIntegrityError, match="replay_checkpoint_digest"):
        _validate(path, snapshot_info)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"stop_step": 19}, "stop_step"),
        ({"terminal_weights_match": False}, "terminal_weights_match"),
        ({"lr_steps_at_stop": 0.5}, "lr_steps_at_stop"),
    ],
)
def test_terminal_replay_requires_complete_matching_trajectory(
    tmp_path, overrides, match
):
    snapshot_info = _snapshot(tmp_path, overrides.get("stop_step", 20))
    with pytest.raises(ValueError, match=match):
        _write(
            tmp_path,
            mode="replayed_terminal",
            snapshot_info=snapshot_info,
            **overrides,
        )


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"stop_step": 3}, "stop_step"),
        ({"warmup_steps": 20, "stop_step": 20}, "warmup_steps"),
        ({"terminal_weights_match": True}, "terminal_weights_match"),
    ],
)
def test_warmup_replay_requires_exact_warmup_boundary(tmp_path, overrides, match):
    snapshot_info = _snapshot(tmp_path, overrides.get("stop_step", 4))
    with pytest.raises(ValueError, match=match):
        _write(
            tmp_path,
            mode="replayed_warmup_proxy",
            snapshot_info=snapshot_info,
            **overrides,
        )
