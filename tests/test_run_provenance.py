"""Integration coverage for provenance in Bellhop's gitless snapshots."""

import json
import stat
import subprocess
from pathlib import Path

import pytest

from scimt.train.runlog import snapshot_run
from scimt.train.source_manifest import (
    build_source_manifest,
    verify_source_manifest,
)

_BELLHOP_TAR_EXCLUDES = (
    "--exclude=.git",
    "--exclude=__pycache__",
    "--exclude=.venv",
    "--exclude=node_modules",
    "--exclude=*.pyc",
)


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_source(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _git("init", "-q", cwd=source)
    config = source / "stage.yaml"
    config.write_text("seed: 7\n")
    runner = source / "run-stage"
    runner.write_text("#!/bin/sh\nexit 0\n")
    runner.chmod(0o755)
    _git("add", "stage.yaml", "run-stage", cwd=source)
    _git(
        "-c", "user.email=t@t", "-c", "user.name=t",
        "commit", "-qm", "source", cwd=source,
    )
    manifest = source / ".scimt-source.json"
    payload = build_source_manifest(source, manifest)
    return source, config, runner, manifest, payload


def _bellhop_transfer(source: Path, destination: Path) -> None:
    archive = source.parent / "source.tar.gz"
    subprocess.run(
        ["tar", "czf", str(archive), "-C", str(source), *_BELLHOP_TAR_EXCLUDES, "."],
        check=True,
    )
    destination.mkdir()
    subprocess.run(
        ["tar", "xzf", str(archive), "-C", str(destination)], check=True
    )


def _gitless_source(tmp_path: Path):
    source, _config, _runner, manifest, payload = _git_source(tmp_path)
    transferred = tmp_path / "transferred"
    _bellhop_transfer(source, transferred)
    return (
        transferred,
        transferred / "stage.yaml",
        transferred / manifest.name,
        payload,
    )


def _gitless_env(monkeypatch, tmp_path: Path, manifest: Path, commit: str):
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("SCIMT_SOURCE_COMMIT", commit)
    monkeypatch.setenv("SCIMT_SOURCE_MANIFEST", str(manifest))
    monkeypatch.setenv("SCIMT_RUNTIME_ROOT", str(runtime))
    return runtime


def test_builder_derives_git_identity_and_rejects_dirty_tracked_source(tmp_path):
    source, config, _runner, manifest, payload = _git_source(tmp_path)
    assert payload["commit"] == _git("rev-parse", "HEAD", cwd=source)
    assert payload["git_tree"] == _git("rev-parse", "HEAD^{tree}", cwd=source)

    config.write_text("seed: 8\n")
    with pytest.raises(RuntimeError, match="dirty tracked source"):
        build_source_manifest(source, manifest)


def test_manifest_matches_bellhop_transfer_exclusions_and_preserves_mode(tmp_path):
    source, _config, runner, manifest, payload = _git_source(tmp_path)
    excluded = (
        source / ".venv" / "drop",
        source / "pkg" / "__pycache__" / "drop.pyc",
        source / "node_modules" / "drop.js",
        source / "loose.pyc",
    )
    for path in excluded:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("excluded\n")
    payload = build_source_manifest(source, manifest)
    assert payload["files"][runner.name]["mode"] == 0o755
    assert not any("drop" in name or name.endswith(".pyc") for name in payload["files"])

    transferred = tmp_path / "transferred"
    _bellhop_transfer(source, transferred)
    assert all(not (transferred / path.relative_to(source)).exists() for path in excluded)
    assert stat.S_IMODE((transferred / runner.name).stat().st_mode) == 0o755
    verified = verify_source_manifest(
        transferred,
        transferred / manifest.name,
        expected_commit=payload["commit"],
    )
    assert verified["source_files_sha256"] == payload["source_files_sha256"]


def test_gitless_snapshot_records_verified_manifest_and_exact_config(
    tmp_path, monkeypatch
):
    source, config, manifest, payload = _gitless_source(tmp_path)
    runtime = _gitless_env(monkeypatch, tmp_path, manifest, payload["commit"])

    record = snapshot_run(
        runtime / "coin", "bellhop-coin", {"stage": config}, repo_dir=source
    )

    assert record.git_commit == payload["commit"]
    assert record.git_dirty is False
    assert record.source_manifest["git_tree"] == payload["git_tree"]
    on_disk = json.loads((runtime / "coin" / "run.json").read_text())
    assert on_disk["source_manifest"]["source_files_sha256"]
    assert (runtime / "coin" / "config" / "stage.yaml").read_bytes() == config.read_bytes()


def test_gitless_snapshot_rejects_transfer_mutation(tmp_path, monkeypatch):
    source, config, manifest, payload = _gitless_source(tmp_path)
    config.write_text("seed: 8\n")
    runtime = _gitless_env(monkeypatch, tmp_path, manifest, payload["commit"])

    with pytest.raises(RuntimeError, match="source file mismatch"):
        snapshot_run(runtime / "run", "mutated", {"stage": config}, repo_dir=source)


def test_gitless_snapshot_rejects_env_commit_disagreement(tmp_path, monkeypatch):
    source, _config, manifest, _payload = _gitless_source(tmp_path)
    runtime = _gitless_env(monkeypatch, tmp_path, manifest, "c" * 40)

    with pytest.raises(RuntimeError, match="source commit mismatch"):
        snapshot_run(runtime / "run", "mismatched-commit", {}, repo_dir=source)


def test_gitless_snapshot_rejects_invalid_commit_and_source_tree_output(
    tmp_path, monkeypatch
):
    source, config, manifest, payload = _gitless_source(tmp_path)
    monkeypatch.setenv("SCIMT_SOURCE_MANIFEST", str(manifest))
    monkeypatch.setenv("SCIMT_RUNTIME_ROOT", str(source / "runtime"))

    monkeypatch.setenv("SCIMT_SOURCE_COMMIT", "short")
    with pytest.raises(RuntimeError, match="full hexadecimal"):
        snapshot_run(source / "runtime" / "bad-commit", "bad", {}, repo_dir=source)

    monkeypatch.setenv("SCIMT_SOURCE_COMMIT", payload["commit"])
    with pytest.raises(RuntimeError, match="outside the immutable source tree"):
        snapshot_run(
            source / "runtime" / "bad-output",
            "bad",
            {"stage": config},
            repo_dir=source,
        )
