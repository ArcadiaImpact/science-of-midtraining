"""The confusion grid: 3 AFT mixtures x 4 parents, and how it is packed onto pods.

Adapted from ``experiments/prior_coins/wave_plan.py`` (wave v1). Grid =
{agreement, coin2, charter2} x {cc, ca, ac, aa} — ``mixed_balanced`` is
dropped.

Parent labels are **provenance-based**: two letters, (coin corpus, charter
corpus), where ``c`` = the clean gate2 slice and ``a`` = the winner-swapped
("anti") corpus. So ``ca`` = clean coin + anti charter. Labels describe what
the parent was trained ON, never the behaviour we expect from it — an
anti-corpus arm may or may not install an inverted prior, which is the
question.

All four parents follow the gate2_midtrain4 "balanced" recipe (4-epoch CPT on
the 1:1:2 mixture -> Dolci-100 SFT); ``cc`` IS the existing gate2-balanced
checkpoint, the other three are new runs whose revisions are filled in after
training publishes (see README.md). The planner prints the plan with
placeholder revisions, but **refuses to emit worklists** until every revision
is a real 40-hex commit — a worklist with a placeholder would 404 at prepare
time on every cell of a pod.

AFT data is byte-identical to wave v1 (same published mixtures/episodes at
``extensions/wave_v1/data``); only the parents and the results destination
change. Results land under ``extensions/confusion_v1`` in the NEW dataset repo
``arcadia-impact/scimt-confusion-aft-v1`` (uploaded centrally, off-pod — the
on-pod per-cell upload stays skipped, see run_confusion_worklist.sh).

Packing: cells are grouped by parent so a pod downloads a 24 GB parent once
and runs its mixtures back to back. 2 pods x 2 parents x 3 mixtures.

``python3 wave_plan.py`` prints the plan; ``--worklists DIR`` writes one
pipe-delimited worklist per pod for ``run_confusion_worklist.sh``. Worklist
lines carry the revision per line (unlike wave v1's single REVISION file),
because the four parents are pinned at different repo revisions:

    label|parent_label|parent_prefix|parent_revision|dataset
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PARENT_REPO = "jbostock/scimt-dispatch-midtrained-sft-v1"

#: sentinel for the three arms whose training has not published yet
PLACEHOLDER_REVISION = "PENDING_TRAINING"
#: gate2-balanced: the SDF boundaries and the exact-copy ledger both live here
GATE2_REVISION = "7a5f7f3a93a962ef378aa95f6f83ddae791d1d43"

#: label -> (path inside PARENT_REPO, pinned revision).
#: Revisions differ per parent because each publish is its own commit.
PARENTS: dict[str, tuple[str, str]] = {
    "cc": ("gate2_midtrain4/balanced/post_dolci100", GATE2_REVISION),
    "ca": ("confusion_v1/ca/post_dolci100", PLACEHOLDER_REVISION),
    "ac": ("confusion_v1/ac/post_dolci100", PLACEHOLDER_REVISION),
    "aa": ("confusion_v1/aa/post_dolci100", PLACEHOLDER_REVISION),
}
MIXTURES = ("agreement", "coin2", "charter2")
#: nothing is pre-run: the cc parent existed, but no wave cell was ever run on it
DONE: frozenset[tuple[str, str]] = frozenset()
PODS = 2

#: where the byte-identical wave-v1 AFT data still lives
DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
DATA_PREFIX = "extensions/wave_v1/data"
#: NEW results home (dataset repo) — wave v1's repo stays as-run
RESULTS_REPO = "arcadia-impact/scimt-confusion-aft-v1"
REMOTE_ROOT = "extensions/confusion_v1"

#: measured on v4_wide / the wave-v1 LoRA validation run, minutes
TRAIN_MIN, TRAJ_MIN, BASE_MIN, PREP_MIN = 59, 27, 6, 3
SETUP_MIN = 6


def cells() -> list[tuple[str, str]]:
    return [(p, m) for p in PARENTS for m in MIXTURES if (p, m) not in DONE]


def pack(pods: int = PODS) -> list[list[tuple[str, str]]]:
    """Group by parent, then greedily fill the emptiest bin (wave-v1 packing)."""
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


def estimate(bin_cells) -> int:
    parents = len({p for p, _ in bin_cells})
    return len(bin_cells) * (TRAIN_MIN + TRAJ_MIN) + parents * (BASE_MIN + PREP_MIN)


def pending_parents() -> list[str]:
    """Parents whose revision is not a real 40-hex commit."""
    return [
        label for label, (_, revision) in PARENTS.items()
        if not re.fullmatch(r"[0-9a-f]{40}", revision)
    ]


def write_worklists(bins, out_dir: Path) -> list[Path]:
    """Write one worklist per pod. Refuses while any revision is a placeholder."""
    pending = pending_parents()
    if pending:
        raise RuntimeError(
            f"cannot emit worklists: parent revision(s) not pinned yet for "
            f"{pending} (currently placeholders like {PLACEHOLDER_REVISION!r}). "
            f"Publish the confusion_v1 checkpoints, then paste each 40-hex "
            f"commit into PARENTS in {__file__} — see aft/README.md."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for index, b in enumerate(bins, start=1):
        lines = [
            f"{parent}__{mixture}|{parent}|{PARENTS[parent][0]}"
            f"|{PARENTS[parent][1]}|{mixture}"
            for parent, mixture in b
        ]
        path = out_dir / f"confusion{index}.worklist"
        path.write_text("\n".join(lines) + "\n")
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worklists", default=None)
    args = parser.parse_args()
    bins = pack()

    total = sum(len(b) for b in bins)
    print(f"{len(PARENTS)} parents x {len(MIXTURES)} mixtures = "
          f"{len(PARENTS)*len(MIXTURES)} cells; {len(DONE)} already done; "
          f"{total} to run")
    print(f"parents from {PARENT_REPO}; data {DATA_REPO}::{DATA_PREFIX} "
          f"(byte-identical wave v1); results -> {RESULTS_REPO}::{REMOTE_ROOT}\n")
    if pending_parents():
        print(f"NOTE: revisions still pending for {pending_parents()} — "
              f"plan only, worklists will refuse\n")
    worst = 0
    for index, b in enumerate(bins, start=1):
        minutes = estimate(b)
        worst = max(worst, minutes)
        print(f"  confusion{index}: {len(b)} cells, ~{minutes/60:.1f} h")
        for parent, mixture in b:
            print(f"      {parent:4s} {mixture}")
    total_minutes = sum(estimate(b) for b in bins) + PODS * SETUP_MIN
    print(f"\nlongest pod ~{worst/60:.1f} h; "
          f"~${total_minutes/60*3.29:.0f} total at $3.29/hr "
          f"({total_minutes/60:.1f} GPU-h incl. setup)")
    if total != len(PARENTS)*len(MIXTURES) - len(DONE):
        raise SystemExit("packing lost or duplicated cells")

    if args.worklists:
        written = write_worklists(bins, Path(args.worklists))
        print(f"\nwrote {len(written)} worklists to {args.worklists}")


if __name__ == "__main__":
    main()
