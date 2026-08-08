from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[2]
    / "experiments"
    / "improved_midtraining"
    / "hf_midtrained_sft"
    / "consolidate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "hf_midtrained_sft_consolidate", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
consolidate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = consolidate
SPEC.loader.exec_module(consolidate)


def remote_file(path: str, identity: str = "abc"):
    return consolidate.RemoteFile(
        path=path,
        size=123,
        identity_kind="lfs_sha256",
        identity=identity,
    )


def test_declared_checkpoints_exclude_aft_paths() -> None:
    paths = {checkpoint.path for checkpoint in consolidate.CHECKPOINTS}

    assert len(paths) == 14
    assert paths == {
        f"{stage}/{arm}/checkpoint-{step}"
        for stage, steps in {
            "midtraining": (2, 30),
            "sft": (4, 48),
            "midtraining_4epoch": (4, 124),
        }.items()
        for arm in ("coin", "charter")
        for step in steps
    } | {f"sft_4epoch/coin/checkpoint-{step}" for step in (4, 48)}
    assert all("aft" not in path for path in paths)
    assert consolidate.SOURCE_INVENTORY_REVISION == next(
        checkpoint.source_revision
        for checkpoint in consolidate.CHECKPOINTS
        if checkpoint.path == "sft_4epoch/coin/checkpoint-48"
    )


def test_exact_destination_is_idempotent() -> None:
    source = {"model.safetensors": remote_file("source/model.safetensors")}
    destination = {"model.safetensors": remote_file("dest/model.safetensors")}

    assert consolidate.classify_destination(source, destination) == "exact"


def test_partial_destination_is_rejected() -> None:
    source = {
        "config.json": remote_file("source/config.json"),
        "model.safetensors": remote_file("source/model.safetensors"),
    }
    destination = {"config.json": remote_file("dest/config.json")}

    with pytest.raises(RuntimeError, match="partial or divergent"):
        consolidate.classify_destination(source, destination)


def test_identity_mismatch_is_rejected() -> None:
    source = {"model.safetensors": remote_file("source/model.safetensors", "abc")}
    destination = {"model.safetensors": remote_file("dest/model.safetensors", "def")}

    with pytest.raises(RuntimeError, match="partial or divergent"):
        consolidate.classify_destination(source, destination)


def test_copy_operations_are_pinned_cross_repository_copies() -> None:
    files = {
        "config.json": remote_file("midtraining/coin/checkpoint-2/config.json"),
        "model.safetensors": remote_file(
            "midtraining/coin/checkpoint-2/model.safetensors"
        ),
    }

    operations = consolidate.build_copy_operations(
        files,
        checkpoint_path="midtraining/coin/checkpoint-2",
        source_revision="source-commit",
    )

    assert [operation.path_in_repo for operation in operations] == [
        "midtraining/coin/checkpoint-2/config.json",
        "midtraining/coin/checkpoint-2/model.safetensors",
    ]
    assert all(operation.src_revision == "source-commit" for operation in operations)
    assert all(
        operation.src_repo_id == consolidate.SOURCE_REPO for operation in operations
    )
    assert all(operation.src_repo_type == "model" for operation in operations)
