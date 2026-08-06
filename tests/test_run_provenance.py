"""Integration coverage for provenance in Bellhop's gitless snapshots."""

import json

import pytest

from scimt.train.runlog import snapshot_run
from scimt.train.source_manifest import build_source_manifest


def _gitless_source(tmp_path, *, commit="a" * 40):
    source = tmp_path / "source"
    source.mkdir()
    config = source / "stage.yaml"
    config.write_text("seed: 7\n")
    manifest = source / ".scimt-source.json"
    build_source_manifest(
        source,
        manifest,
        commit=commit,
        git_tree="b" * 40,
    )
    return source, config, manifest


def test_gitless_snapshot_requires_and_records_verified_manifest(
    tmp_path, monkeypatch
):
    commit = "c" * 40
    source, config, manifest = _gitless_source(tmp_path, commit=commit)
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("SCIMT_SOURCE_COMMIT", commit)
    monkeypatch.setenv("SCIMT_SOURCE_MANIFEST", str(manifest))
    monkeypatch.setenv("SCIMT_RUNTIME_ROOT", str(runtime_root))

    record = snapshot_run(
        runtime_root / "coin",
        "bellhop-coin",
        {"stage": config},
        repo_dir=source,
    )

    assert record.git_commit == commit
    assert record.git_dirty is False
    assert record.source_manifest["git_tree"] == "b" * 40
    on_disk = json.loads((runtime_root / "coin" / "run.json").read_text())
    assert on_disk["source_manifest"]["source_files_sha256"]
    assert (runtime_root / "coin" / "config" / "stage.yaml").is_file()


def test_gitless_snapshot_rejects_source_mutation(tmp_path, monkeypatch):
    source, config, manifest = _gitless_source(tmp_path)
    config.write_text("seed: 8\n")
    monkeypatch.setenv("SCIMT_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setenv("SCIMT_SOURCE_MANIFEST", str(manifest))
    monkeypatch.setenv("SCIMT_RUNTIME_ROOT", str(tmp_path / "runtime"))

    with pytest.raises(RuntimeError, match="source file mismatch"):
        snapshot_run(
            tmp_path / "runtime" / "run",
            "mutated",
            {"stage": config},
            repo_dir=source,
        )


def test_gitless_snapshot_rejects_env_commit_that_disagrees_with_manifest(
    tmp_path, monkeypatch
):
    source, _config, manifest = _gitless_source(tmp_path, commit="a" * 40)
    monkeypatch.setenv("SCIMT_SOURCE_COMMIT", "c" * 40)
    monkeypatch.setenv("SCIMT_SOURCE_MANIFEST", str(manifest))
    monkeypatch.setenv("SCIMT_RUNTIME_ROOT", str(tmp_path / "runtime"))

    with pytest.raises(RuntimeError, match="source commit mismatch"):
        snapshot_run(
            tmp_path / "runtime" / "run",
            "mismatched-commit",
            {},
            repo_dir=source,
        )


def test_gitless_snapshot_rejects_unverified_commit_and_source_tree_output(
    tmp_path, monkeypatch
):
    source, config, manifest = _gitless_source(tmp_path)
    monkeypatch.setenv("SCIMT_SOURCE_MANIFEST", str(manifest))
    monkeypatch.setenv("SCIMT_RUNTIME_ROOT", str(source / "runtime"))

    monkeypatch.setenv("SCIMT_SOURCE_COMMIT", "short")
    with pytest.raises(RuntimeError, match="full hexadecimal"):
        snapshot_run(source / "runtime" / "bad-commit", "bad", {}, repo_dir=source)

    monkeypatch.setenv("SCIMT_SOURCE_COMMIT", "a" * 40)
    with pytest.raises(RuntimeError, match="outside the immutable source tree"):
        snapshot_run(
            source / "runtime" / "bad-output",
            "bad",
            {"stage": config},
            repo_dir=source,
        )
