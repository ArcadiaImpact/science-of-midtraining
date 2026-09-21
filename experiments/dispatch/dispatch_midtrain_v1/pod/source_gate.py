"""Bind Bellhop's git-less source transfer to an exact committed snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

_OID_LENGTHS = {40, 64}
_RUNTIME_PREFIX = (
    "experiments",
    "dispatch",
    "dispatch_midtrain_v1",
    "runs",
)


def _validate_oid(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in _OID_LENGTHS
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"invalid {name}: {value!r}")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _entry(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        payload = os.readlink(path).encode("utf-8", "surrogateescape")
        kind = "symlink"
    else:
        payload = path.read_bytes()
        kind = "file"
    return {
        "kind": kind,
        "mode": stat.S_IMODE(path.lstat().st_mode),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _scan(root: Path, manifest_path: Path) -> dict[str, dict[str, Any]]:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    try:
        manifest_relative = manifest_path.relative_to(root).as_posix()
    except ValueError as error:
        raise RuntimeError("source manifest must live inside source root") from error

    files: dict[str, dict[str, Any]] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if relative.parts[: len(_RUNTIME_PREFIX)] == _RUNTIME_PREFIX:
            continue
        name = relative.as_posix()
        if name == manifest_relative:
            continue
        if path.is_symlink() or path.is_file():
            files[name] = _entry(path)
    return dict(sorted(files.items()))


def build_manifest(
    root: str | Path,
    manifest_path: str | Path,
    *,
    commit: str,
    git_tree: str,
) -> dict[str, Any]:
    """Write a full-file manifest after a clean exact-commit checkout."""

    root = Path(root)
    manifest_path = Path(manifest_path)
    files = _scan(root, manifest_path)
    payload = {
        "schema_version": 1,
        "commit": _validate_oid(commit, "commit"),
        "git_tree": _validate_oid(git_tree, "git tree"),
        "files": files,
        "source_files_sha256": _canonical_sha256(files),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def verify_manifest(
    root: str | Path,
    manifest_path: str | Path,
    *,
    expected_commit: str,
) -> dict[str, Any]:
    """Verify commit identity, the complete file set, and every file digest."""

    root = Path(root)
    manifest_path = Path(manifest_path)
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"invalid source manifest {manifest_path}: {error}"
        ) from error
    if payload.get("schema_version") != 1:
        raise RuntimeError("unsupported source manifest schema")
    commit = _validate_oid(payload.get("commit"), "manifest commit")
    _validate_oid(payload.get("git_tree"), "manifest git tree")
    if commit != _validate_oid(expected_commit, "expected commit"):
        raise RuntimeError(
            f"source commit mismatch: manifest={commit}, expected={expected_commit}"
        )
    expected_files = payload.get("files")
    if not isinstance(expected_files, dict):
        raise RuntimeError("source manifest files must be an object")
    if payload.get("source_files_sha256") != _canonical_sha256(expected_files):
        raise RuntimeError("source manifest digest mismatch")

    actual_files = _scan(root, manifest_path)
    if set(actual_files) != set(expected_files):
        missing = sorted(set(expected_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected_files))
        raise RuntimeError(
            f"source file set mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )
    for name, expected in expected_files.items():
        if actual_files[name] != expected:
            raise RuntimeError(
                f"source file mismatch for {name}: "
                f"expected={expected}, actual={actual_files[name]}"
            )
    return payload


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "commit": payload["commit"],
        "git_tree": payload["git_tree"],
        "source_files": len(payload["files"]),
        "source_files_sha256": payload["source_files_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write", "verify"))
    parser.add_argument("root")
    parser.add_argument("manifest")
    parser.add_argument("commit")
    parser.add_argument("git_tree", nargs="?")
    args = parser.parse_args()
    if args.action == "write":
        if args.git_tree is None:
            parser.error("write requires git_tree")
        payload = build_manifest(
            args.root,
            args.manifest,
            commit=args.commit,
            git_tree=args.git_tree,
        )
    else:
        payload = verify_manifest(
            args.root,
            args.manifest,
            expected_commit=args.commit,
        )
    print(json.dumps(_summary(payload), sort_keys=True))


if __name__ == "__main__":
    main()
