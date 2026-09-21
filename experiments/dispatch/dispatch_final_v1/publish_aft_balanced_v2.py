"""Publish the balanced-v2 AFT cells next to the 250M charter release.

The chain fetches ``<DATA_REPO>/<aft_data_prefix>/aft/aft_<cell>.jsonl`` at the
profile's ``data_revision`` and verifies each file against the manifest
committed next to contracts.py (``aft_manifest_balanced_v2.json``). The
balanced-v2 build (build_aft_mixtures.py, 2026-09-07: clause x run-count
stratified conflicts, nested 1/2/5% doses) only ever existed as a local
artifact plus per-run training-data copies inside the model repos, so this
puts the authoritative bytes where the 1B row's profile can pin them.

Run once, from a checkout whose committed manifest matches the source dir:

    python3 publish_aft_balanced_v2.py --source <balanced_v2>/data-validated

Uploads all eight cells + the manifest (the chain reads four; the 1%/5% pairs
ride along so a later dose row needs no second publication), verifies every
uploaded file's size and LFS sha256 at the resulting immutable revision, and
writes publish_receipt_aft_balanced_v2.json. Re-running with an existing
receipt only re-verifies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
PUBLISH_REPO = "arcadia-impact/scimt-dispatch-charter-250m-v1"
PUBLISH_PREFIX = "releases/dispatch-charter-250m-v1/aft"
COMMITTED_MANIFEST = EXP / "aft_manifest_balanced_v2.json"
RECEIPT = EXP / "publish_receipt_aft_balanced_v2.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(api, revision: str, files: dict[str, dict]) -> None:
    entries = {
        e.path: e for e in api.list_repo_tree(
            PUBLISH_REPO, repo_type="dataset", revision=revision,
            path_in_repo=PUBLISH_PREFIX, recursive=True)
        if getattr(e, "size", None) is not None
    }
    for name, want in files.items():
        entry = entries.get(f"{PUBLISH_PREFIX}/{name}")
        if entry is None or entry.size != want["size"]:
            raise RuntimeError(f"remote missing/wrong size: {PUBLISH_PREFIX}/{name}")
        lfs = getattr(entry, "lfs", None)
        digest = (lfs.get("sha256") if isinstance(lfs, dict)
                  else getattr(lfs, "sha256", None)) if lfs else None
        if digest is not None and digest != want["sha256"]:
            raise RuntimeError(f"remote sha256 mismatch: {PUBLISH_PREFIX}/{name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True,
                    help="balanced-v2 data-validated/ directory")
    args = ap.parse_args()
    source = args.source.resolve()

    manifest = json.loads((source / "aft_manifest.json").read_text())
    if manifest["version"] != "dispatch_final_v1_aft_balanced_v2":
        raise SystemExit(f"not a balanced-v2 manifest: {manifest['version']}")
    if (source / "aft_manifest.json").read_bytes() != COMMITTED_MANIFEST.read_bytes():
        raise SystemExit(
            f"{COMMITTED_MANIFEST.name} does not byte-match the source manifest; "
            "commit the manifest of the bytes you are publishing")
    files: dict[str, dict] = {}
    for cell, entry in manifest["cells"].items():
        path = source / f"aft_{cell}.jsonl"
        digest = sha256(path)
        if digest != entry["sha256"]:
            raise SystemExit(f"{path.name}: sha256 {digest} != manifest {entry['sha256']}")
        files[path.name] = {"size": path.stat().st_size, "sha256": digest}
    files["aft_manifest.json"] = {
        "size": (source / "aft_manifest.json").stat().st_size,
        "sha256": sha256(source / "aft_manifest.json"),
    }

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    if RECEIPT.exists():
        old = json.loads(RECEIPT.read_text())
        if old["files"] == files:
            verify(api, old["revision"], files)
            print(f"already published and verified at {old['revision']}")
            return
        raise SystemExit("receipt exists for different bytes; refusing to overwrite")

    info = api.repo_info(PUBLISH_REPO, repo_type="dataset")  # must already exist
    operations = [
        CommitOperationAdd(path_in_repo=f"{PUBLISH_PREFIX}/{name}",
                           path_or_fileobj=str(source / name))
        for name in sorted(files)
    ]
    commit = api.create_commit(
        repo_id=PUBLISH_REPO, repo_type="dataset", operations=operations,
        commit_message=f"{PUBLISH_PREFIX}: balanced-v2 AFT cells "
                       f"({manifest['version']}, built {manifest['built']})",
        parent_commit=info.sha,
    )
    revision = commit.oid
    verify(api, revision, files)
    receipt = {
        "repo": PUBLISH_REPO,
        "repo_type": "dataset",
        "prefix": PUBLISH_PREFIX,
        "revision": revision,
        "parent_revision": info.sha,
        "manifest_version": manifest["version"],
        "manifest_sha256": files["aft_manifest.json"]["sha256"],
        "files": files,
        "published": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
