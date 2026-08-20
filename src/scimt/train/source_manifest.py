"""Content-addressed source manifests for gitless training snapshots.

Bellhop copies a checkout without git metadata, virtual environments, bytecode
caches, or node_modules. A commit string in an environment variable is
therefore only a claim; this module binds that claim to the exact tar file set,
including executable modes. Launchers build the manifest from a clean tracked
checkout after finalizing transfer inputs, and pod-side provenance verifies it
before recording the run.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

SOURCE_MANIFEST_NAME = ".scimt-source.json"
_OID_LENGTHS = (40, 64)
_BELLHOP_EXCLUDED_NAMES = frozenset({".git", ".venv", "__pycache__", "node_modules"})


def _bellhop_excluded(name: str) -> bool:
    """Whether Bellhop 0.6.1's tar excludes an entry with this basename."""

    return name in _BELLHOP_EXCLUDED_NAMES or name.endswith(".pyc")


def validate_full_commit(value: Any, *, name: str = "commit") -> str:
    """Return a full lowercase hexadecimal git object id or raise."""

    if (
        not isinstance(value, str)
        or len(value) not in _OID_LENGTHS
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(
            f"{name} must be a full hexadecimal git object id, got {value!r}"
        )
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_entry(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        payload = os.readlink(path).encode("utf-8", "surrogateescape")
        return {
            "kind": "symlink",
            "mode": stat.S_IMODE(path.lstat().st_mode),
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    # Hash in chunks — a whole-file read_bytes() OOM-killed the msm sweep
    # runner once experiments/*/data held multi-hundred-MB jsonls (2026-08-20).
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {
        "kind": "file",
        "mode": stat.S_IMODE(path.lstat().st_mode),
        "size": size,
        "sha256": digest.hexdigest(),
    }


def _manifest_relative(root: Path, manifest_path: Path) -> str:
    try:
        return manifest_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError("source manifest must live inside source root") from error


def _scan_source(root: Path, manifest_path: Path) -> dict[str, dict[str, Any]]:
    root = root.resolve()
    manifest_relative = _manifest_relative(root, manifest_path)
    files: dict[str, dict[str, Any]] = {}
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept_dirs: list[str] = []
        for dirname in sorted(dirnames):
            if _bellhop_excluded(dirname):
                continue
            path = current_path / dirname
            if path.is_symlink():
                name = path.relative_to(root).as_posix()
                if name != manifest_relative:
                    files[name] = _file_entry(path)
            else:
                kept_dirs.append(dirname)
        dirnames[:] = kept_dirs
        for filename in sorted(filenames):
            if _bellhop_excluded(filename):
                continue
            path = current_path / filename
            name = path.relative_to(root).as_posix()
            if name != manifest_relative:
                files[name] = _file_entry(path)
    return dict(sorted(files.items()))


def build_source_manifest(
    source_root: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Write a manifest for a clean exact-commit source snapshot.

    The builder derives commit and tree identity from git and refuses tracked
    changes. Its only scan exclusions are Bellhop's own tar exclusions; mutable
    runtime output belongs outside the source root.
    """

    root = Path(source_root).resolve()
    manifest = Path(manifest_path)
    tracked_status = _git_output(
        "status", "--porcelain=v1", "--untracked-files=no", cwd=root
    )
    if tracked_status:
        raise RuntimeError(
            "refusing to manifest a dirty tracked source checkout:\n"
            f"{tracked_status}"
        )
    commit = validate_full_commit(_git_output("rev-parse", "HEAD", cwd=root))
    git_tree = validate_full_commit(
        _git_output("rev-parse", "HEAD^{tree}", cwd=root), name="git tree"
    )
    files = _scan_source(root, manifest)
    payload = {
        "schema_version": 1,
        "commit": validate_full_commit(commit),
        "git_tree": validate_full_commit(git_tree, name="git tree"),
        "files": files,
        "source_files_sha256": _canonical_sha256(files),
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _git_output(*args: str, cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        )
    except FileNotFoundError as error:
        raise RuntimeError("git executable not found on PATH") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {error.stderr.strip()}"
        ) from error
    return result.stdout.strip()


def verify_source_manifest(
    source_root: str | Path,
    manifest_path: str | Path,
    *,
    expected_commit: str,
) -> dict[str, Any]:
    """Verify commit identity, the complete file set, and every file digest."""

    root = Path(source_root)
    manifest = Path(manifest_path)
    _manifest_relative(root, manifest)
    try:
        payload = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid source manifest {manifest}: {error}") from error
    if payload.get("schema_version") != 1:
        raise RuntimeError("unsupported source manifest schema")
    commit = validate_full_commit(payload.get("commit"), name="manifest commit")
    validate_full_commit(payload.get("git_tree"), name="manifest git tree")
    expected = validate_full_commit(expected_commit, name="SCIMT_SOURCE_COMMIT")
    if commit != expected:
        raise RuntimeError(
            f"source commit mismatch: manifest={commit}, SCIMT_SOURCE_COMMIT={expected}"
        )
    expected_files = payload.get("files")
    if not isinstance(expected_files, dict):
        raise RuntimeError("source manifest files must be an object")
    if payload.get("source_files_sha256") != _canonical_sha256(expected_files):
        raise RuntimeError("source manifest digest mismatch")

    actual_files = _scan_source(root, manifest)
    if set(actual_files) != set(expected_files):
        missing = sorted(set(expected_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected_files))
        raise RuntimeError(
            f"source file set mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )
    for name, expected_entry in expected_files.items():
        if actual_files[name] != expected_entry:
            raise RuntimeError(
                f"source file mismatch for {name}: expected={expected_entry}, "
                f"actual={actual_files[name]}"
            )
    return payload


def manifest_summary(payload: dict[str, Any], manifest_path: str | Path) -> dict[str, Any]:
    """Small run-record representation of a successfully verified manifest."""

    return {
        "path": str(Path(manifest_path)),
        "schema_version": payload["schema_version"],
        "commit": payload["commit"],
        "git_tree": payload["git_tree"],
        "source_files": len(payload["files"]),
        "source_files_sha256": payload["source_files_sha256"],
    }
