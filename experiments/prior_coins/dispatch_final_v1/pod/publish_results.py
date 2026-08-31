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

#: Resume-only state, never published. In the AFT cells this is optimizer.pt at
#: 1.05 GB x 8 checkpoints x 4 cells x 3 arms = 100.6 GB per run, against a
#: few-TB quota; the stages set save_only_model: true so it should not exist at
#: all, and this is the guard for when one of them gets flipped back.
RESUME_ONLY_NAMES = frozenset({
    "optimizer.pt", "scheduler.pt", "rng_state.pth",
})


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def published_stages(root: Path) -> set[str]:
    """Stages already uploaded by chain.start_stage_upload, from their receipts.

    The chain now publishes each stage as it lands (so the uploads overlap the
    next stage's compute instead of forming a serial tail at the end), which
    makes this script a sweep-up for the run records rather than the main event.
    Re-uploading a 100 GB tree that is already on the Hub costs commits against
    a 320/hour cap for no benefit.
    """
    return {f.stem.replace("PUBLISHED_", "").lower()
            for f in root.glob("PUBLISHED_*.json")}


def collect(root: Path, arm: str, skip: set[str] | None = None) -> list[tuple[Path, str]]:
    """(local, remote) for everything worth keeping, minus already-published stages.

    Remote paths are rooted at contracts.hub_arm_prefix(arm): the completed
    as-run row keeps its legacy <arm>/ layout, every other grid row publishes
    under <profile>/<arm>/ so rows can never overwrite each other.
    """
    skip = skip or set()
    prefix = C.hub_arm_prefix(arm)
    out: list[tuple[Path, str]] = []

    def add_tree(base: Path, prefix: str, *, only_suffixes=None) -> None:
        if not base.is_dir():
            return
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.name in RESUME_ONLY_NAMES or path.name.startswith("optimizer_"):
                continue
            if only_suffixes and path.suffix not in only_suffixes:
                continue
            out.append((path, f"{prefix}/{path.relative_to(base)}"))

    for leg in ("midtrain", "dolci"):
        if leg in skip:
            continue
        add_tree(root / leg / "checkpoints", f"{prefix}/{leg}/checkpoints")
        add_tree(root / leg, f"{prefix}/{leg}/run", only_suffixes=RESULT_SUFFIXES)
    if "aft" not in skip:
        for cell in C.AFT_CELLS:
            add_tree(root / "aft" / cell / "checkpoints",
                     f"{prefix}/aft/{cell}/checkpoints")
            add_tree(root / "aft" / cell, f"{prefix}/aft/{cell}/run",
                     only_suffixes=RESULT_SUFFIXES)
    if "eval" not in skip:
        add_tree(root / "eval", f"{prefix}/eval", only_suffixes=RESULT_SUFFIXES)
    if "recall" not in skip:
        add_tree(root / "recall", f"{prefix}/recall", only_suffixes=RESULT_SUFFIXES)
    for name in sorted(root.glob("*.json")) + sorted(root.glob("*.yaml")):
        out.append((name, f"{prefix}/{name.name}"))

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

    skip = published_stages(args.root)
    if skip:
        log(f"{args.arm}: stages already published, skipping: {sorted(skip)}")
    files = collect(args.root, args.arm, skip=skip)
    if not files:
        log(f"{args.arm}: nothing left to publish; every stage has a receipt")
        receipt = {"arm": args.arm, "repo": MODEL_REPO, "repo_type": "model",
                   "files": 0, "total_bytes": 0, "public": True,
                   "stages_already_published": sorted(skip), "groups": {}}
        (args.root / "publish_receipt.json").write_text(
            json.dumps(receipt, indent=1) + "\n")
        return
    total = sum(p.stat().st_size for p, _ in files)
    log(f"{args.arm}: {len(files)} files, {total / 1e9:.2f} GB -> {MODEL_REPO}")
    if args.dry_run:
        for local, remote in files[:20]:
            log(f"  {local.stat().st_size / 1e6:>10.1f} MB  {remote}")
        log("dry run")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    from huggingface_hub import CommitOperationAdd
    api.create_repo(MODEL_REPO, repo_type="model", private=False, exist_ok=True)
    info = api.repo_info(MODEL_REPO, repo_type="model")
    if info.private:
        # Reversed from the original check, deliberately. Private Hub storage is
        # METERED, and this run exhausted it mid-flight: uploads started failing
        # with "You need to setup automatic credit recharge in order to upload
        # more data" after ~600 GB. Public repos are not metered, and the
        # dispatch scenario is specific to this research rather than a benchmark
        # being withheld, so public is the decided default.
        raise SystemExit(
            f"{MODEL_REPO} is PRIVATE; storage is metered and ~630 GB/run will "
            "exhaust it mid-run. Make it public before publishing.")

    # ONE COMMIT PER GROUP, not per file. The Hub caps repository commits at
    # 320/hour, shared across every arm pushing to this repo, and a per-file
    # loop blows through that long before it runs out of bandwidth -- which is
    # exactly what happened on the first full run (429, 37-minute cooldown,
    # three pods stalled). Bytes are pre-uploaded by the commit machinery, so a
    # large grouped commit is cheap; it is the commit COUNT that is rationed.
    groups: dict[str, list[tuple[Path, str]]] = {}
    for local, remote in files:
        groups.setdefault(remote.split("/")[1], []).append((local, remote))
    for group, items in groups.items():
        log(f"{args.arm}/{group}: uploading {len(items)} files "
            f"({sum(p.stat().st_size for p, _ in items) / 1e9:.2f} GB) in 1 commit")
        api.create_commit(
            repo_id=MODEL_REPO, repo_type="model",
            operations=[CommitOperationAdd(path_in_repo=remote,
                                           path_or_fileobj=str(local))
                        for local, remote in items],
            commit_message=f"{args.arm}/{group}",
        )

    log("verifying remote sizes ...")
    remote_info = api.repo_info(MODEL_REPO, repo_type="model", files_metadata=True)
    sizes = {s.rfilename: s.size for s in remote_info.siblings}
    bad = [f"{r}: remote {sizes.get(r)} != local {p.stat().st_size}"
           for p, r in files if sizes.get(r) != p.stat().st_size]
    if bad:
        raise SystemExit("VERIFICATION FAILED:\n  " + "\n  ".join(bad[:20]))

    receipt = {
        "arm": args.arm, "repo": MODEL_REPO, "repo_type": "model",
        "files": len(files), "total_bytes": total, "public": True,
        "stages_already_published": sorted(skip),
        "groups": {g: len(i) for g, i in groups.items()},
        "published": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.root / "publish_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    log(f"{args.arm}: verified {len(files)} files, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
