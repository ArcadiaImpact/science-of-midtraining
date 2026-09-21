"""Publish the Dispatch final-run data release to the Hub, and verify it landed.

Uploads the two 50M-token arm corpora, the four AFT cells, and both manifests
to the private org repo the raw v3 blocks already live in, under a new prefix.

Verification is not optional here. "A backup nobody can find is not a backup"
(dispatch_docgen_v3_extension/RESULTS.md), and the failure this guards against
is specific: a pod pulls a truncated corpus, trains on it, and the dose is
silently wrong. Every file is re-listed after upload and its remote size checked
against the local bytes; the manifests carry the sha256s a consumer should
re-check after download.

Run: python3 publish.py            # dry run: lists what would be uploaded
     python3 publish.py --upload   # actually uploads
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
PREFIX = "releases/dispatch-final-v1"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def planned_files() -> list[tuple[Path, str]]:
    """(local path, remote path) for everything the pods need."""
    out: list[tuple[Path, str]] = []
    release = RUNS / "release"
    for arm in ("coin", "charter"):
        out.append((release / arm / "corpus.jsonl", f"{PREFIX}/release/{arm}/corpus.jsonl"))
    out.append((release / "release_manifest.json", f"{PREFIX}/release/release_manifest.json"))

    aft = RUNS / "aft"
    for path in sorted(aft.glob("aft_*.jsonl")):
        out.append((path, f"{PREFIX}/aft/{path.name}"))
    for name in ("aft_manifest.json", "charter_only_manifest.json"):
        if (aft / name).is_file():
            out.append((aft / name, f"{PREFIX}/aft/{name}"))

    missing = [str(p) for p, _ in out if not p.is_file()]
    if missing:
        raise SystemExit("missing local files:\n  " + "\n  ".join(missing))
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
            f"{REPO} is PUBLIC; this release is not cleared for publication"
        )

    for local, remote in files:
        log(f"uploading {remote} ...")
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=REPO, repo_type=REPO_TYPE)

    log("verifying remote sizes ...")
    # repo_info().siblings silently truncates on large repos (see
    # pod/publish_results.py, measured 2026-09-01); ask for exact paths.
    sizes: dict[str, int | None] = {}
    remotes = [remote for _, remote in files]
    for start in range(0, len(remotes), 500):
        for entry in api.get_paths_info(REPO, remotes[start:start + 500],
                                        repo_type=REPO_TYPE):
            sizes[entry.path] = getattr(entry, "size", None)
    bad = []
    for local, remote in files:
        want, got = local.stat().st_size, sizes.get(remote)
        if got != want:
            bad.append(f"{remote}: remote {got} != local {want}")
    if bad:
        raise SystemExit("VERIFICATION FAILED:\n  " + "\n  ".join(bad))
    log(f"verified {len(files)} files, {total / 1e6:.1f} MB")

    receipt = {
        "repo": REPO, "repo_type": REPO_TYPE, "prefix": PREFIX,
        "published": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [{"remote": r, "bytes": local.stat().st_size} for local, r in files],
        "total_bytes": total,
        "private": True,
    }
    dest = HERE / "publish_receipt.json"
    dest.write_text(json.dumps(receipt, indent=2) + "\n")
    log(f"receipt -> {dest}")


if __name__ == "__main__":
    sys.exit(main())
