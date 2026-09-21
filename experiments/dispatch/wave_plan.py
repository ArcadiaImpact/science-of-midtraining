"""The wave grid: 4 AFT mixtures x 10 parents, and how it is packed onto pods.

Grid = {agreement, mixed_balanced, coin2, charter2}
     x {charter,coin} x {real,fake} x {1x,4x}  +  {control} x {1x,4x}

"Real" midtraining puts the arm documents BEFORE instruct training
(``Dolmino+docs -> Dolci SFT``); "fake" puts them after
(``Dolmino -> Dolci90 -> docs -> Dolci10``). The dose axis is 1x/4x in both, but
it is not strictly commensurable across them: for real it is epochs of the
midtrain mixture (30 vs 124 steps), for fake it is presentations of the arm
documents *and* of Dolmino (16 vs 64 steps on the arm section). The control
differs between doses only in Dolmino presentations.

Two cells are already done — (charter_real_1x, agreement) and
(coin_real_1x, agreement) are exactly the v4_wide run, on a byte-identical
training file — so 38 remain.

Packing: cells are grouped by parent so a pod downloads a 24 GB parent once and
runs its mixtures back to back, then bins are balanced to keep the longest pod
under one night. Splitting a parent across pods costs one extra download (~3 min)
and is preferred over an unbalanced bin.

``python3 wave_plan.py`` prints the plan; ``--worklists DIR`` writes one
pipe-delimited worklist per pod for ``pod/run_wave_worklist.sh``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PARENT_REPO = "jbostock/scimt-dispatch-midtrained-sft-v1"
#: pinned: the SDF boundaries and the exact-copy ledger are both at this revision
PARENT_REVISION = "527f0b6cc0ea117e7c9e89e82221163654bd50db"

#: label -> path inside PARENT_REPO
PARENTS = {
    "charter_real_1x": "sft/charter/checkpoint-48",
    "coin_real_1x": "sft/coin/checkpoint-48",
    "charter_real_4x": "sft_4epoch/charter/checkpoint-48",
    "coin_real_4x": "sft_4epoch/coin/checkpoint-48",
    "charter_fake_1x": "sdf/1x/charter/final",
    "coin_fake_1x": "sdf/1x/coin/final",
    "charter_fake_4x": "sdf/4x/charter/final",
    "coin_fake_4x": "sdf/4x/coin/final",
    "control_1x": "sdf/1x/shared/post_dolci90",
    "control_4x": "sdf/4x/shared/post_dolci90",
}
MIXTURES = ("agreement", "mixed_balanced", "coin2", "charter2")
#: (parent, mixture) pairs already run — v4_wide, on a byte-identical training file
DONE = {("charter_real_1x", "agreement"), ("coin_real_1x", "agreement")}
PODS = 8
#: measured on v4_wide / the LoRA validation run, minutes
TRAIN_MIN, TRAJ_MIN, BASE_MIN, PREP_MIN = 59, 27, 6, 3


def cells():
    return [(p, m) for p in PARENTS for m in MIXTURES if (p, m) not in DONE]


def pack(pods: int = PODS):
    """Group by parent, then greedily fill the emptiest bin."""
    groups: dict[str, list] = {}
    for parent, mixture in cells():
        groups.setdefault(parent, []).append(mixture)
    bins: list[list] = [[] for _ in range(pods)]
    # largest groups first so the big ones land whole
    for parent, mixes in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        target = min(range(pods), key=lambda i: len(bins[i]))
        room = max(1, (len(cells()) + pods - 1) // pods - len(bins[target]))
        take, rest = mixes[:room], mixes[room:]
        bins[target].extend((parent, m) for m in take)
        for mixture in rest:  # spill the remainder into the next emptiest bins
            spill = min(range(pods), key=lambda i: len(bins[i]))
            bins[spill].append((parent, mixture))
    return bins


def estimate(bin_cells):
    parents = len({p for p, _ in bin_cells})
    return len(bin_cells) * (TRAIN_MIN + TRAJ_MIN) + parents * (BASE_MIN + PREP_MIN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worklists", default=None)
    args = parser.parse_args()
    bins = pack()

    total = sum(len(b) for b in bins)
    print(f"{len(PARENTS)} parents x {len(MIXTURES)} mixtures = "
          f"{len(PARENTS)*len(MIXTURES)} cells; {len(DONE)} already done; "
          f"{total} to run\n")
    worst = 0
    for index, b in enumerate(bins, start=1):
        minutes = estimate(b)
        worst = max(worst, minutes)
        print(f"  wave{index}: {len(b)} cells, ~{minutes/60:.1f} h")
        for parent, mixture in b:
            print(f"      {parent:16s} {mixture}")
    print(f"\nlongest pod ~{worst/60:.1f} h; "
          f"~${(sum(estimate(b) for b in bins)/60 + 8*6/60)*3.29:.0f} at $3.29/hr")
    if total != len(PARENTS)*len(MIXTURES) - len(DONE):
        raise SystemExit("packing lost or duplicated cells")

    if args.worklists:
        out = Path(args.worklists)
        out.mkdir(parents=True, exist_ok=True)
        for index, b in enumerate(bins, start=1):
            lines = [
                f"{parent}__{mixture}|{parent}|{PARENTS[parent]}|{mixture}"
                for parent, mixture in b
            ]
            (out / f"wave{index}.worklist").write_text("\n".join(lines) + "\n")
        (out / "REVISION").write_text(PARENT_REVISION + "\n")
        print(f"\nwrote {len(bins)} worklists to {out}")


if __name__ == "__main__":
    main()
