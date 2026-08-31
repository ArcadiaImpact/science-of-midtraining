"""Persist one arm's results to the Hub, verified, before the pod can die.

The convention this enforces is CLAUDE.md's "pointers, not weights": local
checkpoint dirs are impermanent, so the durable object is what is on the Hub.
A pod terminated after training but before this ran would destroy every full
checkpoint, every adapter and every raw response, and leave the committed
checkpoint.json pointers aimed at paths that no longer exist.

What goes up, per arm:
  midtrain/checkpoints/checkpoint-{38,122,381}   3 x ~24 GB full model states
  dolci/checkpoints/checkpoint-{48} (+42 control)    ~24 GB each
  aft/<cell>/checkpoints/checkpoint-*            8 adapters x 4 cells
  eval/**/*.jsonl                                every raw response
  *.json, *.log, rendered configs                provenance

Verification is by remote size against local bytes, per file. An upload that
"succeeded" but truncated is the failure that would only surface weeks later,
when the checkpoint is reloaded and does not match its manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

POD = Path(__file__).resolve().parent
EXP = POD.parent
REPO_ROOT = EXP.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402

MODEL_REPO = os.environ.get("FINAL_V1_MODEL_REPO",
                            "arcadia-impact/scimt-dispatch-final-v1")
#: results are small and numerous; checkpoints are few and enormous
RESULT_SUFFIXES = (".json", ".jsonl", ".log", ".yaml", ".yml", ".txt")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def collect(root: Path, arm: str) -> list[tuple[Path, str]]:
    """(local, remote) for everything worth keeping."""
    out: list[tuple[Path, str]] = []

    def add_tree(base: Path, prefix: str, *, only_suffixes=None) -> None:
        if not base.is_dir():
            return
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if only_suffixes and path.suffix not in only_suffixes:
                continue
            out.append((path, f"{prefix}/{path.relative_to(base)}"))

    for leg in ("midtrain", "dolci"):
        add_tree(root / leg / "checkpoints", f"{arm}/{leg}/checkpoints")
        add_tree(root / leg, f"{arm}/{leg}/run", only_suffixes=RESULT_SUFFIXES)
    for cell in C.AFT_CELLS:
        add_tree(root / "aft" / cell / "checkpoints", f"{arm}/aft/{cell}/checkpoints")
        add_tree(root / "aft" / cell, f"{arm}/aft/{cell}/run",
                 only_suffixes=RESULT_SUFFIXES)
    add_tree(root / "eval", f"{arm}/eval", only_suffixes=RESULT_SUFFIXES)
    for name in sorted(root.glob("*.json")) + sorted(root.glob("*.yaml")):
        out.append((name, f"{arm}/{name.name}"))

    seen, unique = set(), []
    for local, remote in out:
        if remote not in seen:
            seen.add(remote)
            unique.append((local, remote))
    return unique


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(C.ARMS))
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = collect(args.root, args.arm)
    if not files:
        raise SystemExit(f"nothing to publish under {args.root}")
    total = sum(p.stat().st_size for p, _ in files)
    log(f"{args.arm}: {len(files)} files, {total / 1e9:.2f} GB -> {MODEL_REPO}")
    if args.dry_run:
        for local, remote in files[:20]:
            log(f"  {local.stat().st_size / 1e6:>10.1f} MB  {remote}")
        log("dry run")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(MODEL_REPO, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(MODEL_REPO, repo_type="model")
    if not info.private:
        raise SystemExit(f"{MODEL_REPO} is PUBLIC; refusing to push checkpoints")

    # Folder upload per subtree: one commit per group keeps a failure recoverable
    # and avoids a single 100 GB commit that cannot be retried cheaply.
    groups: dict[str, list[tuple[Path, str]]] = {}
    for local, remote in files:
        groups.setdefault(remote.split("/")[1], []).append((local, remote))
    for group, items in groups.items():
        log(f"{args.arm}/{group}: uploading {len(items)} files "
            f"({sum(p.stat().st_size for p, _ in items) / 1e9:.2f} GB)")
        for local, remote in items:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                            repo_id=MODEL_REPO, repo_type="model")

    log("verifying remote sizes ...")
    remote_info = api.repo_info(MODEL_REPO, repo_type="model", files_metadata=True)
    sizes = {s.rfilename: s.size for s in remote_info.siblings}
    bad = [f"{r}: remote {sizes.get(r)} != local {p.stat().st_size}"
           for p, r in files if sizes.get(r) != p.stat().st_size]
    if bad:
        raise SystemExit("VERIFICATION FAILED:\n  " + "\n  ".join(bad[:20]))

    receipt = {
        "arm": args.arm, "repo": MODEL_REPO, "repo_type": "model",
        "files": len(files), "total_bytes": total, "private": True,
        "groups": {g: len(i) for g, i in groups.items()},
        "published": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.root / "publish_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    log(f"{args.arm}: verified {len(files)} files, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
