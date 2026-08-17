"""Publish probe directions (and cache/log dirs) to the HF Hub — verified.

Shape mirrors ``scimt.publish`` (async wrapper over a blocking worker, lazy
huggingface_hub, ``private=True`` by default — publishing is outward-facing,
flip it deliberately; a README card embedding the manifest is rendered into
the probe dir). Hardening ports the aft_v2 ``upload_folder_verified``
correctness: local {path: size} inventory first, upload, **pin the returned
commit SHA** (empty -> error), re-list the remote tree at that revision,
diff missing/extra/wrong-size, retry with backoff. The receipt's pinned
``revision`` is the durable pointer (pointers-not-weights).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

from .probes import MANIFEST_NAME, ProbeSet

CARD_TEMPLATE = """---
tags:
- probing
- linear-probe
- scimt
---

# {repo_id}

Linear probe directions published by `probing.publish_probes`. The manifest
below is the durable recipe: fitter + params, the (rendering, position,
layer) slice, classes, split description, and the upstream activation-cache
identity that regenerates these directions.

```json
{manifest_json}
```
"""


def render_probe_card(repo_id: str, manifest: dict[str, Any]) -> str:
    """Pure: the model-card text for a probe repo (manifest embedded)."""
    card_manifest = {k: v for k, v in manifest.items() if k != "tensors"}
    return CARD_TEMPLATE.format(
        repo_id=repo_id, manifest_json=json.dumps(card_manifest, indent=2)
    )


def _resolve_probe_dir(source: "ProbeSet | str | Path") -> tuple[Path, dict[str, Any]]:
    if isinstance(source, ProbeSet):
        if source.dir is None:
            raise ValueError(
                "this ProbeSet was never saved — call .save(dir) before "
                "publish_probes"
            )
        return Path(source.dir), source.manifest
    d = Path(source)
    mp = d / MANIFEST_NAME
    if not mp.exists():
        raise ValueError(
            f"cannot publish {d}: no {MANIFEST_NAME} (pass a saved ProbeSet, "
            "or a directory produced by ProbeSet.save)"
        )
    return d, json.loads(mp.read_text())


def _upload_ignored(rel: str) -> bool:
    """Mirror what won't reach the Hub: our *.tmp exclusion plus hub's own
    DEFAULT_IGNORE_PATTERNS (.git*, .cache/huggingface) — files upload_folder
    silently skips must not count as 'missing' in verification."""
    parts = rel.split("/")
    if parts[-1].endswith(".tmp"):
        return True
    if any(part.startswith(".git") for part in parts):
        return True
    for i, part in enumerate(parts[:-1]):
        if part == ".cache" and parts[i + 1] == "huggingface":
            return True
    return False


def _local_inventory(folder: Path) -> dict[str, int]:
    return {
        str(p.relative_to(folder)): p.stat().st_size
        for p in sorted(folder.rglob("*"))
        if p.is_file() and not _upload_ignored(str(p.relative_to(folder)))
    }


def _remote_inventory(
    api: Any, repo_id: str, repo_type: str, revision: str, prefix: str | None
) -> dict[str, int]:
    out: dict[str, int] = {}
    for entry in api.list_repo_tree(
        repo_id, repo_type=repo_type, revision=revision, recursive=True, expand=True
    ):
        size = getattr(entry, "size", None)
        if size is None:  # folders
            continue
        path = entry.path
        if path == ".gitattributes":  # Hub-seeded system file, every repo
            continue
        if prefix:
            if not path.startswith(prefix + "/"):
                continue
            path = path[len(prefix) + 1 :]
        out[path] = size
    return out


def _hub_url(repo_id: str, repo_type: str, revision: str, path_in_repo: str | None) -> str:
    base = "https://huggingface.co/"
    if repo_type == "dataset":
        base += "datasets/"
    url = f"{base}{repo_id}/tree/{revision}"
    return f"{url}/{path_in_repo}" if path_in_repo else url


def _push_verified(
    folder: Path,
    repo_id: str,
    *,
    private: bool,
    repo_type: str,
    path_in_repo: str | None,
    token: str | None,
    attempts: int,
    commit_message: str,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    # "." / "" / trailing-slash forms would break both hub semantics and the
    # client-side prefix filter — normalize to the None root form.
    path_in_repo = (path_in_repo or "").strip("/") or None
    if path_in_repo == ".":
        path_in_repo = None
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    local = _local_inventory(folder)
    if not local:
        raise ValueError(f"nothing to publish: {folder} has no files")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            commit = api.upload_folder(
                repo_id=repo_id,
                repo_type=repo_type,
                folder_path=str(folder),
                path_in_repo=path_in_repo or ".",
                commit_message=commit_message,
                ignore_patterns=["*.tmp"],  # matches the inventory exclusion
                # aft_v2 parity: replace stale remote content within the
                # target scope, so a re-publish IS the folder. NB scoped to
                # path_in_repo — publishing to a shared repo's ROOT replaces
                # everything there; use per-run prefixes on shared repos.
                delete_patterns="**",
            )
            revision = str(getattr(commit, "oid", "") or "")
            if not revision:
                raise RuntimeError("Hub upload returned no commit SHA")
            remote = _remote_inventory(api, repo_id, repo_type, revision, path_in_repo)
            missing = sorted(set(local) - set(remote))
            extra = sorted(set(remote) - set(local))
            wrong = sorted(
                p for p in set(local) & set(remote) if local[p] != remote[p]
            )
            if missing or extra or wrong:
                raise RuntimeError(
                    f"post-upload verification failed at {revision}: "
                    f"missing={missing[:5]} extra={extra[:5]} wrong_size={wrong[:5]}"
                )
            return {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "revision": revision,
                "path_in_repo": path_in_repo,
                "file_count": len(local),
                "total_bytes": sum(local.values()),
                "url": _hub_url(repo_id, repo_type, revision, path_in_repo),
            }
        except Exception as e:  # noqa: BLE001 — retried, re-raised on the last
            last_error = e
            if attempt == attempts - 1:
                raise
            time.sleep(min(60.0, 2.0**attempt + random.random()))
    raise RuntimeError(f"unreachable: {last_error}")


async def publish_probes(
    source: "ProbeSet | str | Path",
    repo_id: str,
    *,
    private: bool = True,
    repo_type: str = "model",
    path_in_repo: str | None = None,
    token: str | None = None,
    attempts: int = 6,
) -> dict[str, Any]:
    """Push a saved probe directory to the Hub, verified; returns the receipt
    (``revision`` is the pin to record). Renders README.md into the probe dir
    first — the card embeds the manifest, so the Hub copy is self-describing."""
    folder, manifest = _resolve_probe_dir(source)
    (folder / "README.md").write_text(render_probe_card(repo_id, manifest))

    def _work() -> dict[str, Any]:
        return _push_verified(
            folder,
            repo_id,
            private=private,
            repo_type=repo_type,
            path_in_repo=path_in_repo,
            token=token,
            attempts=attempts,
            commit_message=f"probing.publish_probes: {folder.name}",
        )

    return await asyncio.to_thread(_work)


async def publish_dir(
    folder: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    repo_type: str = "dataset",
    path_in_repo: str | None = None,
    token: str | None = None,
    attempts: int = 6,
) -> dict[str, Any]:
    """Generic verified folder push (activation-cache shards, run logs) — no
    card, dataset repo by default."""
    d = Path(folder)
    if not d.is_dir():
        raise FileNotFoundError(f"publish_dir: {d} is not a directory")

    def _work() -> dict[str, Any]:
        return _push_verified(
            d,
            repo_id,
            private=private,
            repo_type=repo_type,
            path_in_repo=path_in_repo,
            token=token,
            attempts=attempts,
            commit_message=f"probing.publish_dir: {d.name}",
        )

    return await asyncio.to_thread(_work)


async def download_probes(
    repo_id: str,
    *,
    dest: str | Path,
    revision: str | None = None,
    repo_type: str = "model",
    path_in_repo: str | None = None,
    token: str | None = None,
) -> ProbeSet:
    """Pinned fetch of published probe directions -> loaded ProbeSet."""

    def _work() -> ProbeSet:
        from huggingface_hub import snapshot_download

        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "revision": revision,
            "token": token,
            "local_dir": str(dest),
        }
        if path_in_repo:
            kwargs["allow_patterns"] = [f"{path_in_repo}/*", f"{path_in_repo}/**"]
        root = Path(snapshot_download(**kwargs))
        return ProbeSet.load(root / path_in_repo if path_in_repo else root)

    return await asyncio.to_thread(_work)
