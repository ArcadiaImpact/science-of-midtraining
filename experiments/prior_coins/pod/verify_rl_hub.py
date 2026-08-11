"""Prove every artifact reached the Hub BEFORE a pod is destroyed.

Deleting a pod destroys its container disk with no undelete, so "the publisher said
it worked" is not the standard — this re-lists the repo and checks the listing
against an explicit expectation. It exists because a publisher already lied once
this run: a `checkpoint-16_vllm` directory made `int()` throw during enumeration,
and the wrapper printed success after the crash, so zero files had been uploaded
while the log said DONE.

Checks per trained cell: an adapter at every expected dose, and
``optimizer.pt`` at the final dose (without it the cell can be re-served but not
resumed). Per base arm: the four eval slices plus the probe.

    python verify_rl_hub.py --prefix extensions/rl_v3 \
        --cell charter_real_4x_thinking --cell coin_real_4x_thinking \
        --base charter_real_4x_thinking__base

Exits non-zero and prints what is missing, so it can gate a deletion in a script.
``check`` is pure and takes a file listing, so the logic is testable without a
token or a network round-trip.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
#: a base arm writes 4 eval slices + sanity_prompts.jsonl and no weights
BASE_ROW_FILES = 5


def group(files: list[str], prefix: str) -> dict[str, list[str]]:
    """Bucket a repo listing by cell name (results/<name> stays distinct)."""
    cells: dict[str, list[str]] = defaultdict(list)
    for path in files:
        if not path.startswith(prefix):
            continue
        parts = path[len(prefix):].lstrip("/").split("/")
        if not parts or not parts[0]:
            continue
        key = f"results/{parts[1]}" if parts[0] == "results" and len(parts) > 1 \
            else parts[0]
        cells[key].append(path)
    return dict(cells)


def check(files: list[str], prefix: str, cells: list[str], bases: list[str],
          doses: list[int]) -> list[str]:
    """Everything wrong with this listing, as human-readable lines. Empty == safe."""
    grouped = group(files, prefix)
    problems: list[str] = []
    for cell in cells:
        present = grouped.get(cell, [])
        if not present:
            problems.append(f"{cell}: nothing on the Hub at all")
            continue
        got = {p.split("/")[-2].replace("checkpoint-", "")
               for p in present if "/checkpoint-" in p}
        missing = [d for d in doses if str(d) not in got]
        if missing:
            problems.append(f"{cell}: no adapter at dose(s) {missing}")
        final = max(doses)
        if not any(p.endswith(f"checkpoint-{final}/optimizer.pt") for p in present):
            problems.append(
                f"{cell}: no optimizer.pt at checkpoint-{final} "
                "(servable but not resumable)")
        for dose in doses:
            if not any(p.endswith(f"checkpoint-{dose}/adapter_model.safetensors")
                       for p in present):
                problems.append(f"{cell}: dose {dose} has no adapter_model.safetensors")
    for base in bases:
        rows = [p for p in grouped.get(f"results/{base}", []) if p.endswith(".jsonl")]
        if len(rows) < BASE_ROW_FILES:
            problems.append(
                f"{base}: {len(rows)} row files, want {BASE_ROW_FILES} "
                "(4 eval slices + probe)")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--prefix", default="extensions/rl_v3")
    parser.add_argument("--cell", action="append", default=[],
                        help="trained cell that must have every dose; repeatable")
    parser.add_argument("--base", action="append", default=[],
                        help="dose-0 arm that must have its row files; repeatable")
    parser.add_argument("--doses", default="16,32,64,128,256")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = list(api.list_repo_files(args.repo))
    doses = [int(d) for d in args.doses.split(",") if d.strip()]

    grouped = group(files, args.prefix)
    print(f"{sum(len(v) for v in grouped.values())} files under {args.prefix}\n")
    for key in sorted(grouped):
        steps = sorted({p.split("/")[-2].replace("checkpoint-", "")
                        for p in grouped[key] if "/checkpoint-" in p},
                       key=lambda s: int(s) if s.isdigit() else 0)
        print(f"  {key:38s} {len(grouped[key]):3d} files   {' '.join(steps)}")

    problems = check(files, args.prefix, args.cell, args.base, doses)
    print()
    if problems:
        print("NOT SAFE TO DELETE:")
        for line in problems:
            print(f"  - {line}")
        raise SystemExit(1)
    print(f"ALL EXPECTED ARTIFACTS PRESENT "
          f"({len(args.cell)} cells x {len(doses)} doses, {len(args.base)} base arms)")


if __name__ == "__main__":
    main()
