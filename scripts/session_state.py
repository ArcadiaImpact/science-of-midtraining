#!/usr/bin/env python3
"""Safely inspect or update the local ARCH session JSON.

Every read/modify/write transaction is serialized through a stable sibling
lock file.  The session file itself is replaced atomically, so locking the
session inode would not protect callers across a replace.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

POD_KINDS = ("worker", "heldout", "batch")
POD_FIELDS = {kind: f"{kind}_pod_ids" for kind in POD_KINDS}
FORBIDDEN_KEYS = {
    "api_key",
    "apikey",
    "access_key",
    "access_token",
    "auth_token",
    "credential",
    "credentials",
    "gh_token",
    "github_token",
    "hf_token",
    "password",
    "private_key",
    "runpod_api_key",
    "secret",
    "secret_key",
    "secret_value",
    "token",
    "worker_gh_token",
}
FORBIDDEN_SUFFIXES = (
    "_api_key",
    "_access_key",
    "_access_token",
    "_auth_token",
    "_credential",
    "_password",
    "_private_key",
    "_secret",
    "_secret_key",
    "_token",
)


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SystemExit("session root must be a JSON object")
    return value


def reject_secrets(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            sensitive = lowered in FORBIDDEN_KEYS or lowered.endswith(FORBIDDEN_SUFFIXES)
            if sensitive and child not in (None, "", [], {}):
                raise SystemExit(
                    f"refusing secret-like session field: {'.'.join(path + (key,))}"
                )
            reject_secrets(child, path + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_secrets(child, path + (str(index),))


def _pod_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SystemExit(f"session field {field} must be a list of pod ID strings")
    return list(dict.fromkeys(value))


def normalize_pod_state(state: dict[str, Any]) -> None:
    """Maintain typed pod sets plus a derived legacy cleanup union.

    A session written by the pre-typed runtime is migrated by treating its
    legacy ``pod_ids`` as workers.  Once any typed field exists, ``pod_ids``
    is output-only and never used as an authority, preventing evaluator pods
    from silently entering the worker monitor set.
    """
    has_typed_state = any(field in state for field in POD_FIELDS.values())
    legacy_workers = _pod_list(state.get("pod_ids"), "pod_ids") if not has_typed_state else []
    union: list[str] = []
    for kind in POD_KINDS:
        field = POD_FIELDS[kind]
        pods = _pod_list(state.get(field), field)
        if kind == "worker" and legacy_workers:
            pods = list(dict.fromkeys([*pods, *legacy_workers]))
        state[field] = pods
        union.extend(pods)
    state["pod_ids"] = list(dict.fromkeys(union))


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    normalize_pod_state(value)
    reject_secrets(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


@contextmanager
def transaction(path: Path, *, exclusive: bool) -> Iterator[dict[str, Any]]:
    """Lock a stable sibling file for a complete session transaction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(lock_fd, 0o600)
        with os.fdopen(lock_fd, "r+") as lock_handle:
            fcntl.flock(
                lock_handle.fileno(),
                fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
            )
            state = load(path)
            normalize_pod_state(state)
            yield state
            if exclusive:
                atomic_write(path, state)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    except Exception:
        # fdopen owns lock_fd after it succeeds; only close a still-open raw fd.
        try:
            os.close(lock_fd)
        except OSError:
            pass
        raise


def set_path(root: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = root
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise SystemExit(f"cannot descend through non-object field: {part}")
        node = child
    node[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=Path(".arch/.session.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show")
    replace = sub.add_parser("replace")
    replace.add_argument("json_file", type=Path, help="JSON file, or - for stdin")
    setter = sub.add_parser("set")
    setter.add_argument("path")
    setter.add_argument("json_value")
    append = sub.add_parser("append-action")
    append.add_argument("json_file", type=Path, help="action JSON file, or - for stdin")
    add_pod = sub.add_parser("add-pod")
    add_pod.add_argument("pod_id")
    add_pod.add_argument("--kind", choices=POD_KINDS, default="worker")
    remove_pod = sub.add_parser("remove-pod")
    remove_pod.add_argument("pod_id")
    remove_pod.add_argument("--kind", choices=(*POD_KINDS, "all"), default="all")
    list_pods = sub.add_parser("list-pods")
    list_pods.add_argument("--kind", choices=(*POD_KINDS, "all"), default="all")
    args = parser.parse_args()

    def read_json(path: Path) -> Any:
        if str(path) == "-":
            import sys

            return json.load(sys.stdin)
        return json.loads(path.read_text())

    if args.command in {"show", "list-pods"}:
        with transaction(args.file, exclusive=False) as state:
            if args.command == "show":
                print(json.dumps(state, indent=2, sort_keys=True))
            elif args.kind == "all":
                for pod_id in state["pod_ids"]:
                    print(pod_id)
            else:
                for pod_id in state[POD_FIELDS[args.kind]]:
                    print(pod_id)
        return

    with transaction(args.file, exclusive=True) as state:
        if args.command == "replace":
            replacement = read_json(args.json_file)
            if not isinstance(replacement, dict):
                raise SystemExit("replacement session must be an object")
            state.clear()
            state.update(replacement)
        elif args.command == "set":
            if args.path == "pod_ids" or args.path in POD_FIELDS.values():
                raise SystemExit("use add-pod/remove-pod for typed pod state")
            set_path(state, args.path, json.loads(args.json_value))
        elif args.command == "append-action":
            action = read_json(args.json_file)
            if not isinstance(action, dict):
                raise SystemExit("action must be an object")
            state.setdefault("actions", []).append(action)
        elif args.command == "add-pod":
            pods = state[POD_FIELDS[args.kind]]
            if args.pod_id not in pods:
                pods.append(args.pod_id)
        elif args.command == "remove-pod":
            kinds = POD_KINDS if args.kind == "all" else (args.kind,)
            for kind in kinds:
                field = POD_FIELDS[kind]
                state[field] = [pod for pod in state[field] if pod != args.pod_id]


if __name__ == "__main__":
    main()
