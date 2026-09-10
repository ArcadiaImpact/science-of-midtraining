#!/usr/bin/env python3
"""Reclaim Hub storage held by superseded resume backups of the 1B run.

The uploader replaces <prefix>/ in one commit (delete old + add new), but the
Hub keeps the old LFS objects and counts them in usedStorage until they are
PERMANENTLY deleted (Storage settings "delete" == HfApi.permanently_delete_lfs_files,
history rewritten). A plain delete commit reclaims nothing (Sid, 2026-09-09).

Safety rules (all must hold for an object to be deleted):
  * its filename is under PREFIX (this run's resume/latest only);
  * its sha256 (LFSFileInfo.file_oid) is NOT referenced by any file in the
    current tree of the repo (so the live backup is never touched);
  * its pushed_at is strictly OLDER than the newest referenced object under
    PREFIX (so blobs pre-uploaded by an in-flight attempt are never touched).
Everything outside PREFIX is ignored entirely. Approved by Sid 2026-09-09 10:15Z.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "arcadia-impact/scimt-dispatch-final-v1-glm"
PREFIX = "glm45_air_1b/charter/midtrain/resume/latest/"
LOG = Path(__file__).with_name("hub_resume_janitor.log")
DRY = "--dry-run" in sys.argv
LOOP = "--loop" in sys.argv
POLL_S = 600


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def used_gb(api: HfApi) -> float:
    return (getattr(api.repo_info(REPO, expand=["usedStorage"]), "used_storage", 0) or 0) / 1e9


def pass_once(api: HfApi) -> int:
    tree = list(api.list_repo_tree(REPO, recursive=True, expand=True))
    referenced = {t.lfs.sha256 for t in tree if getattr(t, "lfs", None) is not None}
    if not referenced:
        log("tree listing returned no LFS files; refusing to act")
        return 0
    lfs = [x for x in api.list_lfs_files(REPO) if (x.filename or "").startswith(PREFIX)]
    live = [x for x in lfs if x.file_oid in referenced]
    if not live:
        log("no referenced backup objects under prefix; refusing to act (uploader state unclear)")
        return 0
    newest_live = max(x.pushed_at for x in live)
    stale = [x for x in lfs if x.file_oid not in referenced and x.pushed_at < newest_live]
    skipped_recent = [x for x in lfs if x.file_oid not in referenced and x.pushed_at >= newest_live]
    gb = sum(x.size for x in stale) / 1e9
    versions = sorted({x.pushed_at.strftime("%m-%d %H:%M") for x in stale})
    log(f"prefix objects {len(lfs)}: live {len(live)} (newest {newest_live:%m-%d %H:%M}), "
        f"stale {len(stale)} = {gb:.0f} GB from pushes {versions}, in-flight kept {len(skipped_recent)}")
    if not stale:
        return 0
    for x in stale:  # belt and braces
        assert (x.filename or "").startswith(PREFIX) and x.file_oid not in referenced and x.pushed_at < newest_live
    if DRY:
        log(f"DRY-RUN would permanently delete {len(stale)} objects ({gb:.0f} GB)")
        return len(stale)
    before = used_gb(api)
    api.permanently_delete_lfs_files(REPO, stale, rewrite_history=True)
    time.sleep(20)
    after = used_gb(api)
    log(f"permanently deleted {len(stale)} objects ({gb:.0f} GB); usedStorage {before:.0f} GB -> {after:.0f} GB")
    return len(stale)


def main() -> int:
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN must be in the environment")
    api = HfApi()
    log(f"janitor start dry_run={DRY} loop={LOOP} usedStorage {used_gb(api):.0f} GB")
    while True:
        try:
            pass_once(api)
        except Exception as exc:  # keep the loop alive; report
            log(f"pass failed: {type(exc).__name__}: {str(exc)[:300]}")
        if not LOOP:
            return 0
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
