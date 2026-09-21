"""Cell list and per-pod worklists for the wave v2 AFT re-run.

What changed from `wave_plan` (v1):

* **Everything is re-trained.** v1's `DONE` set is gone: the recipe now routes
  through `scimt.train.train_dataset` and the training stack is pinned, so no
  earlier cell is comparable to these.
* **The control substrate is dose-matched.** v1 used
  `sdf/4x/shared/post_dolci90`, which is short 16M midtraining tokens *and* the
  10M Dolci10 suffix — which is why v1 could only report it as rates, never as a
  separation partner. v2's primary control is Gate-2's Dolmino-only arm: 4x
  continued pretraining at the same ~32M presentations, then the same Dolci100.
  The two SDF controls are kept for figure 2 only, where the question is dose
  *within* a lineage and they are the sole 1x/4x control pair that exists.
* **Two new doses.** 0.2% conflict labels (16 of 8,192 rows) alongside 2%.
  They nest: the 16 rows are a subset of the 164.

Parents all resolve in the public `arcadia-impact/scimt-dispatch-models`.

    python -m experiments.dispatch.wave_v2_plan --worklists /tmp/wl
"""

from __future__ import annotations

import argparse
from pathlib import Path

#: label -> path inside the public model repo
PARENTS = {
    # primary substrates: every mixture runs on these three
    "charter_real_4x": "sft_4epoch/charter/checkpoint-48",
    "coin_real_4x": "sft_4epoch/coin/checkpoint-48",
    "control_matched": "gate2_midtrain4/dolmino/post_dolci100",
    # agreement-only substrates (figures 2, 3, 5)
    "charter_real_1x": "sft/charter/checkpoint-48",
    "coin_real_1x": "sft/coin/checkpoint-48",
    "charter_fake_1x": "sdf/1x/charter/final",
    "coin_fake_1x": "sdf/1x/coin/final",
    "charter_fake_4x": "sdf/4x/charter/final",
    "coin_fake_4x": "sdf/4x/coin/final",
    "control_sdf_1x": "sdf/1x/shared/post_dolci90",
    "control_sdf_4x": "sdf/4x/shared/post_dolci90",
}

PRIMARY = ("charter_real_4x", "coin_real_4x", "control_matched")
PRIMARY_MIXTURES = ("agreement", "coin2", "charter2", "coin0p2", "charter0p2")

PARENT_REPO = "arcadia-impact/scimt-dispatch-models"
DATA_REPO = "arcadia-impact/scimt-dispatch-aft-data"
DATA_PREFIX = "extensions/wave_v2/data"
MODEL_REPO = "arcadia-impact/scimt-dispatch-models"
REMOTE_ROOT = "aft_wave_v2"
STAGE = "aft_dispatch_v4_wide"

PODS = 8
#: measured on the v1 wave, minutes
TRAIN_MIN, TRAJ_MIN, BASE_MIN, PREP_MIN = 59, 27, 6, 3


def cells() -> list[tuple[str, str]]:
    """(parent_label, mixture), 23 of them."""
    out = [(p, m) for p in PRIMARY for m in PRIMARY_MIXTURES]
    out += [(p, "agreement") for p in PARENTS if p not in PRIMARY]
    return out


def pack(pods: int = PODS) -> list[list[tuple[str, str]]]:
    """Balance by *cost*, keeping a parent's cells together where it is free.

    Grouping by parent saves one 26 GB download and one baseline eval per pod,
    which is 9 min against 86 min per cell — so it is worth honouring, but never
    at the price of an unbalanced tail. v1's packer put whole parents in bins and
    left one pod with 7 cells while others had 3; here the three 5-cell parents
    are split deliberately.
    """
    bins: list[list[tuple[str, str]]] = [[] for _ in range(pods)]
    cost = [0.0] * pods

    def bin_cost(index: int, parent: str) -> float:
        parents = {p for p, _ in bins[index]}
        return TRAIN_MIN + TRAJ_MIN + (0 if parent in parents else BASE_MIN + PREP_MIN)

    # heaviest parents first so their cells get spread before the tail lands
    order = sorted(cells(), key=lambda c: (c[0] not in PRIMARY, c[0], c[1]))
    for parent, mixture in order:
        target = min(range(pods), key=lambda i: cost[i] + bin_cost(i, parent))
        cost[target] += bin_cost(target, parent)
        bins[target].append((parent, mixture))
    return bins


def estimate(bin_cells: list[tuple[str, str]]) -> int:
    parents = len({p for p, _ in bin_cells})
    return len(bin_cells) * (TRAIN_MIN + TRAJ_MIN) + parents * (BASE_MIN + PREP_MIN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklists", default=None,
                        help="directory to write pod-<n>.txt worklists into")
    parser.add_argument("--pods", type=int, default=PODS)
    args = parser.parse_args()

    bins = pack(args.pods)
    total = sum(len(b) for b in bins)
    print(f"{len(PARENTS)} parents, {total} cells over {args.pods} pods\n")
    worst = 0
    for index, b in enumerate(bins, start=1):
        minutes = estimate(b)
        worst = max(worst, minutes)
        print(f"pod {index}: {len(b)} cells, {minutes/60:.1f} h")
        for parent, mixture in b:
            print(f"    {parent}__{mixture}")
        if args.worklists:
            out = Path(args.worklists)
            out.mkdir(parents=True, exist_ok=True)
            lines = [f"{p}__{m}|{p}|{PARENTS[p]}|{m}" for p, m in b]
            (out / f"pod-{index}.txt").write_text("\n".join(lines) + "\n")
    gpu_hours = sum(estimate(b) for b in bins) / 60
    print(f"\ncritical path {worst/60:.1f} h; {gpu_hours:.1f} GPU-hours total")
    if args.worklists:
        print(f"worklists -> {args.worklists}")


if __name__ == "__main__":
    main()
