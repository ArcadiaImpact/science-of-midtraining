"""Relieve the 20k-file cap on the final-v1 model repo, without losing a byte.

2026-09-02: arcadia-impact/scimt-dispatch-final-v1 hit the Hub's hard
20,000-files-per-repo limit (19,991 files; the next 16-file publish was
rejected 400), which parks every running pod at its next stage publish. The
file-count bulk is raw-response battery trees (~800 jsonl per arm) of rows
whose pods are already torn down and whose scores are committed to git.

This tool moves those trees to a public archive repo in three separately
invoked, individually safe phases:

  --copy     download the archivable trees locally, upload to the archive
             repo (upload_folder; the 320 commits/hr cap makes per-file
             commits a trap), and write a manifest of every (path, size).
  --verify   re-list the archive repo via get_paths_info (never
             repo_info().siblings -- it truncates) and check EVERY manifest
             entry's remote size. Writes ARCHIVE_VERIFIED sentinel.
  --delete   requires the sentinel AND --yes. Deletes the verified paths
             from the MAIN repo in batched commits, then re-counts.

Archivable = the battery subtrees (eval/recall/d4/costsweep) of profiles
whose pods are DONE (torn down): checkpoints, run records, and root JSONs
stay in the main repo (they are the committed pointer targets). Profiles
with a pod still alive are NEVER touched (their rehydrate/CHAIN_COMPLETE
validation reads those prefixes).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN_REPO = "arcadia-impact/scimt-dispatch-final-v1"
ARCHIVE_REPO = "arcadia-impact/scimt-dispatch-final-v1-archive"

#: pods torn down + scored + scores committed; see PROGRESS.md 2026-09-02.
DONE_PROFILES = (
    "gemma3_4b_1m", "gemma3_4b_5m", "gemma3_4b_50m",
    "gemma3_12b_1m", "gemma3_12b_5m", "gemma3_12b_50m_4ep",
    "gemma3_12b_19m",  # DURABLE COMPLETE + torn down 2026-09-02 10:30Z
    "gemma3_27b_50m",  # DURABLE COMPLETE + torn down 2026-09-02 11:17Z
    "gemma3_27b_5m",   # DURABLE COMPLETE + torn down 2026-09-02 12:10Z
)
#: The completed pre-grid as-run row publishes under legacy top-level <arm>/
#: prefixes (LEGACY_HUB_LAYOUT_PROFILES_FROZEN). Torn down 2026-08, fully
#: scored; its battery trees are archivable the same way (arm at parts[0],
#: battery at parts[1]).
LEGACY_ARMS = ("charter", "coin", "control")
#: Completed + Hub-verified ARMS of still-running rows. Safe for teardown
#: (verify_hub floor is >20 files/arm; checkpoints + records stay put) and
#: for relaunches on the SAME pod (the CHAIN_COMPLETE skip reads local
#: sentinels). CAVEAT documented: recycling such a row onto a FRESH pod
#: requires restoring that arm's battery trees from the archive first --
#: rehydrate reads only the main repo.
DONE_ARMS = (
    ("gemma3_27b_5m", "charter"), ("gemma3_27b_5m", "coin"),
    ("gemma3_12b_50m_noex", "charter"),
    ("gemma3_27b_190m", "charter"),
)
BATTERY_DIRS = ("eval", "recall", "d4", "costsweep")
WORK = HERE / "runs" / "archive_battery_trees"
MANIFEST = WORK / "archive_manifest.json"
SENTINEL = WORK / "ARCHIVE_VERIFIED.json"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def archivable(api) -> list[tuple[str, int]]:
    out = []
    for entry in api.list_repo_tree(MAIN_REPO, repo_type="model", recursive=True):
        size = getattr(entry, "size", None)
        if size is None:
            continue
        parts = entry.path.split("/")
        if len(parts) >= 3 and parts[0] in DONE_PROFILES and parts[2] in BATTERY_DIRS:
            out.append((entry.path, size))
        elif (len(parts) >= 2 and parts[0] in LEGACY_ARMS
                and parts[1] in BATTERY_DIRS):
            out.append((entry.path, size))
        elif (len(parts) >= 3 and (parts[0], parts[1]) in DONE_ARMS
                and parts[2] in BATTERY_DIRS):
            out.append((entry.path, size))
    return out


def do_copy(api) -> None:
    from huggingface_hub import snapshot_download, upload_folder

    files = archivable(api)
    log(f"{len(files)} archivable files, "
        f"{sum(s for _, s in files) / 1e6:.1f} MB")
    WORK.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(
        {"main_repo": MAIN_REPO, "archive_repo": ARCHIVE_REPO,
         "profiles": DONE_PROFILES, "battery_dirs": BATTERY_DIRS,
         "files": [{"path": p, "size": s} for p, s in files]}, indent=1))
    local = WORK / "tree"
    patterns = [f"{prof}/*/{b}/**" for prof in DONE_PROFILES for b in BATTERY_DIRS]
    patterns += [f"{arm}/{b}/**" for arm in LEGACY_ARMS for b in BATTERY_DIRS]
    patterns += [f"{prof}/{arm}/{b}/**" for prof, arm in DONE_ARMS for b in BATTERY_DIRS]
    log("downloading (snapshot_download, battery patterns) ...")
    snapshot_download(MAIN_REPO, repo_type="model", local_dir=str(local),
                      allow_patterns=patterns)
    have = {str(p.relative_to(local)) for p in local.rglob("*") if p.is_file()}
    missing = [p for p, _ in files if p not in have]
    if missing:
        raise SystemExit(f"download incomplete: {len(missing)} missing, "
                         f"first: {missing[:3]}")
    api.create_repo(ARCHIVE_REPO, repo_type="model", private=False, exist_ok=True)
    for prof in DONE_PROFILES + LEGACY_ARMS + tuple(
            sorted({prof for prof, _ in DONE_ARMS})):
        src = local / prof
        if not src.is_dir():
            continue
        n = sum(1 for p in src.rglob('*') if p.is_file())
        log(f"uploading {prof} ({n} files) via upload_folder ...")
        upload_folder(repo_id=ARCHIVE_REPO, repo_type="model",
                      folder_path=str(src), path_in_repo=prof,
                      commit_message=f"archive {prof} battery trees (20k-cap relief)")
    log("copy phase complete; run --verify next")


def remote_sizes(api, repo: str, paths: list[str]) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for start in range(0, len(paths), 500):
        for entry in api.get_paths_info(repo, paths[start:start + 500],
                                        repo_type="model"):
            out[entry.path] = getattr(entry, "size", None)
    return out


def do_verify(api) -> None:
    m = json.loads(MANIFEST.read_text())
    files = [(f["path"], f["size"]) for f in m["files"]]
    log(f"verifying {len(files)} files on {ARCHIVE_REPO} ...")
    sizes = remote_sizes(api, ARCHIVE_REPO, [p for p, _ in files])
    bad = [f"{p}: archive {sizes.get(p)} != main {s}"
           for p, s in files if sizes.get(p) != s]
    if bad:
        raise SystemExit("VERIFY FAILED:\n  " + "\n  ".join(bad[:20]))
    SENTINEL.write_text(json.dumps({
        "verified": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": len(files), "archive_repo": ARCHIVE_REPO}, indent=1))
    log(f"VERIFIED {len(files)} files byte-for-byte by size; sentinel written")


def do_delete(api, yes: bool) -> None:
    from huggingface_hub import CommitOperationDelete

    if not SENTINEL.is_file():
        raise SystemExit("refusing: no ARCHIVE_VERIFIED sentinel")
    if not yes:
        raise SystemExit("refusing: --delete requires --yes (Sid-approved)")
    files = [f["path"] for f in json.loads(MANIFEST.read_text())["files"]]
    log(f"deleting {len(files)} verified-archived files from {MAIN_REPO} ...")
    for start in range(0, len(files), 1000):
        chunk = files[start:start + 1000]
        api.create_commit(
            repo_id=MAIN_REPO, repo_type="model",
            operations=[CommitOperationDelete(path_in_repo=p) for p in chunk],
            commit_message=(f"archive to {ARCHIVE_REPO}: battery trees "
                            f"[{start}:{start + len(chunk)}] (20k-cap relief)"))
        log(f"  deleted [{start}:{start + len(chunk)}]")
    total = sum(1 for e in api.list_repo_tree(MAIN_REPO, repo_type="model",
                                              recursive=True)
                if getattr(e, "size", None) is not None)
    log(f"main repo now {total} files")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    if args.copy:
        do_copy(api)
    if args.verify:
        do_verify(api)
    if args.delete:
        do_delete(api, args.yes)
    if not (args.copy or args.verify or args.delete):
        files = archivable(api)
        log(f"dry run: {len(files)} archivable files, "
            f"{sum(s for _, s in files) / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
