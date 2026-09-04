#!/usr/bin/env python3
"""Push run-5's pod-side logs and incident evidence to the HF logs repo.

WHY THIS EXISTS (coordinator ruling 2026-09-04): this pod is VOLUMELESS
(volumeInGb: 0). Its container disk does not survive a stop, a delete, or a host
recycle -- the same failure mode that destroyed run-1's checkpoints in the
account-zero event. Anything that exists only on /workspace here is a story, not
a record. The incident evidence for the INVALID first EFT and the COLD arm of the
rl_go trigger (the paired baseline for the number that decides the RL spend) both
lived only on that disk until this script ran.

Cheap by construction: logs + curves + transcript heads, a few MB total. Weights
go to GCS, never here (repo convention).

Usage (on the pod):
    python push_run5_logs.py            # evidence + logs
    python push_run5_logs.py --cold     # ... and the COLD trigger stores
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# HF_HUB_DISABLE_XET: xet-backed uploads have hung on these pods; the repo
# convention is to disable it for every pod-side upload. Set before the import
# so it is picked up at client construction.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi  # noqa: E402

REPO_ID = "arcadia-impact/python4-thinking-grpo-logs"
REPO_TYPE = "dataset"
RUN_ID = "20260904T-eftgrpo-g4-31b-prop-run5"
TOKEN_FILE = Path("/workspace/ship/secrets/hf_token")

EVIDENCE = Path("/workspace/run5-INCIDENT-EVIDENCE-invalid-EFT-DO-NOT-CONSUME")
LOGS_SRC = Path("/workspace/run5")
COLD = Path("/workspace/runs/20260904T-eftgrpo-g4-31b-prop-run5-trigger-COLD")

# Only text-ish artifacts. Never sweep in weights or sample stores by accident.
LOG_SUFFIXES = {".log", ".json", ".jsonl", ".yaml", ".txt"}
MAX_LOG_BYTES = 64 * 1024 * 1024


def _api() -> HfApi:
    token = TOKEN_FILE.read_text().strip()
    if not token:
        sys.exit(f"FATAL: empty HF token at {TOKEN_FILE}")
    return HfApi(token=token)


def _upload_folder(api: HfApi, src: Path, dest: str, allow_patterns=None) -> None:
    if not src.is_dir():
        print(f"[skip] not present: {src}")
        return
    print(f"[push] {src} -> {REPO_ID}:{dest}")
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(src),
        path_in_repo=dest,
        allow_patterns=allow_patterns,
        commit_message=f"run-5: {dest}",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", action="store_true", help="also push the COLD trigger sample stores")
    args = ap.parse_args()

    api = _api()
    base = f"{RUN_ID}"

    # 1. incident evidence for the INVALID first EFT. The README travels with it:
    #    these rows measure a bug and must never be read as findings.
    _upload_folder(api, EVIDENCE, f"{base}/INCIDENT-invalid-EFT")

    # 2. pod-side logs (flat files only -- the subdirs under /workspace/run5 hold
    #    adapters and smoke dirs we do not want in a logs repo).
    staged = []
    for p in sorted(LOGS_SRC.glob("*")):
        if p.is_file() and p.suffix in LOG_SUFFIXES and p.stat().st_size <= MAX_LOG_BYTES:
            staged.append(p.name)
    if staged:
        print(f"[push] {len(staged)} log files from {LOGS_SRC}")
        _upload_folder(api, LOGS_SRC, f"{base}/pod-logs", allow_patterns=staged)

    # 3. the COLD arm of the rl_go trigger -- the PAIRED baseline. Sample stores
    #    are the evidence behind the go/no-go number, so they must outlive the pod.
    if args.cold:
        _upload_folder(api, COLD, f"{base}/trigger-COLD", allow_patterns=["*.jsonl", "*.json"])

    print("[done]")


if __name__ == "__main__":
    main()
