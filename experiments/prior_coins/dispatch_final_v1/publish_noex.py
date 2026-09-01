"""Publish the no-example (qualitative-only) release cut, and verify it landed.

Same contract as publish.py (which shipped the main release): dry run by
default, upload only with --upload, remote sizes re-checked file-by-file via
get_paths_info (repo_info().siblings truncates on large repos — measured
2026-09-01), private-repo assertion before any byte moves.

Uploads the two ablation corpora + their manifest under a NEW prefix so the
main release stays untouched:

    releases/dispatch-final-v2-noex/release/{charter,coin}/corpus.jsonl
    releases/dispatch-final-v2-noex/release/release_manifest.json

The layout under the prefix mirrors the main release exactly because
pod/chain.py::fetch_release composes "{DATA_PREFIX}/release/..." — the noex
profile only swaps the prefix, never the fetch code. There is deliberately no
aft/ tree here: AFT cells are pinned location-independently by
contracts.AFT_DATA_PREFIX + aft_manifest.json sha256s.

AFTER a successful upload, pin the new dataset revision:
  1. take `commit_sha` from the printed receipt,
  2. set profiles/gemma3_12b_50m_noex.yaml data_revision to it,
  3. flip that profile's status placeholder -> active,
and only then flip the queue row.

Run: python3 publish_noex.py            # dry run
     python3 publish_noex.py --upload   # actually uploads
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs" / "dispatch_final_v1"

REPO = "arcadia-impact/scimt-prior-coins-scenarios"
REPO_TYPE = "dataset"
PREFIX = "releases/dispatch-final-v2-noex"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def planned_files() -> list[tuple[Path, str]]:
    release = RUNS / "release_v2_noex"
    out = [(release / arm / "corpus.jsonl",
            f"{PREFIX}/release/{arm}/corpus.jsonl")
           for arm in ("charter", "coin")]
    out.append((release / "release_manifest.json",
                f"{PREFIX}/release/release_manifest.json"))
    missing = [str(p) for p, _ in out if not p.is_file()]
    if missing:
        raise SystemExit("missing local files (run build_release_v2_noex.py "
                         "first):\n  " + "\n  ".join(missing))
    committed = (HERE / "release_manifest_v2_noex.json").read_bytes()
    if (release / "release_manifest.json").read_bytes() != committed:
        raise SystemExit(
            "local release_manifest.json does not byte-match the committed "
            "release_manifest_v2_noex.json — rebuild, then commit the pair")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true",
                    help="actually upload; without it this is a dry run")
    args = ap.parse_args()

    files = planned_files()
    total = sum(p.stat().st_size for p, _ in files)
    log(f"{len(files)} files, {total / 1e6:.1f} MB -> {REPO}/{PREFIX}")
    for local, remote in files:
        log(f"  {local.stat().st_size / 1e6:>8.2f} MB  {remote}")
    if not args.upload:
        log("dry run -- pass --upload to publish")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    info = api.repo_info(REPO, repo_type=REPO_TYPE)
    if not info.private:
        raise SystemExit(
            f"{REPO} is PUBLIC; this release is not cleared for publication")

    from huggingface_hub import CommitOperationAdd
    commit = api.create_commit(
        repo_id=REPO, repo_type=REPO_TYPE,
        operations=[CommitOperationAdd(path_in_repo=remote,
                                       path_or_fileobj=str(local))
                    for local, remote in files],
        commit_message="dispatch-final-v2-noex: qualitative-only ablation cut",
    )

    log("verifying remote sizes ...")
    sizes: dict[str, int | None] = {}
    remotes = [remote for _, remote in files]
    for entry in api.get_paths_info(REPO, remotes, repo_type=REPO_TYPE):
        sizes[entry.path] = getattr(entry, "size", None)
    bad = [f"{remote}: remote {sizes.get(remote)} != local {local.stat().st_size}"
           for local, remote in files if sizes.get(remote) != local.stat().st_size]
    if bad:
        raise SystemExit("VERIFICATION FAILED:\n  " + "\n  ".join(bad))
    log(f"verified {len(files)} files, {total / 1e6:.1f} MB")

    receipt = {
        "repo": REPO, "repo_type": REPO_TYPE, "prefix": PREFIX,
        "published": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit_sha": getattr(commit, "oid", None),
        "files": [{"remote": r, "bytes": p.stat().st_size} for p, r in files],
        "total_bytes": total,
        "private": True,
        "next": "pin data_revision=commit_sha in gemma3_12b_50m_noex.yaml, "
                "flip its status to active, then uncomment the queue row",
    }
    dest = HERE / "publish_noex_receipt.json"
    dest.write_text(json.dumps(receipt, indent=2) + "\n")
    log(f"receipt -> {dest}")


if __name__ == "__main__":
    sys.exit(main())
