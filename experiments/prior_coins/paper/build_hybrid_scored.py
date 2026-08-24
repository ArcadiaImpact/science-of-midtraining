"""Splice wave-v1 and wave-v2 into one scored.json for the paper figures.

The goal is that **every row corresponds to a checkpoint someone can
download**. wave-v1 discarded its adapters, so any row it supplies is a figure
of a model that no longer exists; this splice removes those rows. Resolution
order, later rules overriding earlier:

1. **wave-v1** — fallback only. Anything left on this source is reported as
   NO MODEL, and the point of the exercise is for that list to be empty.
2. **wave-v2** (`aft_wave_v2/`) — adapters retained for all 23 cells. Supplies
   the conflict-labelled doses (wave-v1 has no 0.2% arm at all) and the
   SDF-ordered parents, which the retrain never covered.
3. **the §6 retrain** (`aft_wave_retrain/`) — adapters retained; supplies the
   `agreement` cells on `charter_real_4x` and `coin_real_4x`, all endpoints.
4. **the control**, wholesale, from wave-v2's `control_matched`, aliased onto
   the `control_4x` label the figures ask for so no figure script needs editing.

The cost, stated plainly: a single figure now draws rows from up to three
training runs, and agreement cells are exactly the ones that do not reproduce
across runs (up to 24.7 pp at step 512, all at seed 42). Where a figure's claim
is a *comparison between* an agreement row from one run and an agreement row
from another, that comparison is confounded by run provenance. Figure 3 is the
live case — see the report this script prints.

Two things change at once in that control row and both must be said out loud
wherever these figures are used: it is a different *run* (v2, not v1) **and** a
different *substrate* — `gate2_midtrain4/dolmino/post_dolci100`, dose-matched at
~32M presentations with the full Dolci100, instead of wave-v1's
`sdf/4x/shared/post_dolci90`, which is short 16M midtraining tokens and the
Dolci10 suffix. That substitution is the point of the exercise (the control is
finally token-matched), but it means the control row is not comparable to the
control row in the committed figures.

`separation` and `competence` are deliberately emitted **empty**. They are
derived quantities computed per-run; recombining rates across runs does not make
them valid, and leaving wave-v1's in place would let a stale number be quoted as
if it described this splice. Nothing in the five figures reads them, so an empty
block fails loudly rather than misleading quietly.

    python3 experiments/prior_coins/paper/build_hybrid_scored.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
DATA = EXP / "writeup" / "data"

DEFAULT_V1 = DATA / "wave_scored.json"
DEFAULT_V2 = DATA / "wave_v2_scored.json"

LABELLED = ("charter2", "coin2", "charter0p2", "coin0p2")
#: v2's dose-matched control, presented under the label the figures request
CONTROL_SRC, CONTROL_DST = "control_matched", "control_4x"
#: agreement cells the retrain covers (its only trained cells)
RETRAIN_PARENTS = ("charter_real_4x", "coin_real_4x")
DEFAULT_RETRAIN = DATA / "retrain_scored_full.json"
#: which Hub prefix backs each source -- "" means no downloadable checkpoint
HUB_PREFIX = {"wave-v1": "", "wave-v2": "aft_wave_v2/",
              "retrain": "aft_wave_retrain/", "wave-v2 control": "aft_wave_v2/"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", type=Path, default=DEFAULT_V1)
    ap.add_argument("--v2", type=Path, default=DEFAULT_V2)
    ap.add_argument("--retrain", type=Path, default=DEFAULT_RETRAIN)
    ap.add_argument("--out", type=Path, default=DATA / "hybrid_scored.json")
    args = ap.parse_args()

    v1 = json.loads(args.v1.read_text())["rates"]
    v2 = json.loads(args.v2.read_text())["rates"]
    rt = json.loads(args.retrain.read_text())["rates"]

    rates: dict[str, dict] = {}
    source: dict[str, str] = {}

    for key, entry in v1.items():
        rates[key], source[key] = entry, "wave-v1"

    # 2. everything wave-v2 covers -- labelled doses AND the SDF-ordered parents
    for key, entry in v2.items():
        rates[key], source[key] = entry, "wave-v2"

    # 3. the retrain owns agreement on the two real-4x arms, all endpoints
    for key, entry in rt.items():
        parent, mixture, endpoint = key.split("|")
        if parent in RETRAIN_PARENTS and mixture == "agreement":
            rates[key], source[key] = entry, "retrain"

    # the control is replaced wholesale, not merged: mixing v1 and v2 endpoints
    # within one trajectory would make figure 6's SFT curve a splice of two runs
    for key in [k for k in rates if k.startswith(f"{CONTROL_DST}|")]:
        del rates[key], source[key]
    for key, entry in v2.items():
        parent, mixture, endpoint = key.split("|")
        if parent == CONTROL_SRC:
            new = f"{CONTROL_DST}|{mixture}|{endpoint}"
            rates[new] = entry
            source[new] = "wave-v2 control"

    args.out.write_text(json.dumps({
        "rates": rates,
        "separation": {},
        "competence": {},
        "cells_present": len(rates),
        "_hybrid": {
            "note": "spliced; see build_hybrid_scored.py docstring",
            "v1": str(args.v1), "v2": str(args.v2),
            "source": source,
        },
    }, indent=1))

    counts = Counter(source.values())
    print(f"wrote {args.out}  ({len(rates)} cells)")
    for name, n in sorted(counts.items()):
        prefix = HUB_PREFIX.get(name, "?")
        print(f"  {n:>4}  {name:<18}{prefix or 'NO DOWNLOADABLE CHECKPOINT'}")

    orphans = sorted(k for k, s in source.items() if not HUB_PREFIX.get(s, ""))
    trained = [k for k in orphans if not k.endswith("|baseline")]
    print(f"\nrows still backed by no checkpoint: {len(orphans)} "
          f"({len(trained)} of them trained endpoints)")
    if trained:
        for k in trained[:12]:
            print(f"    {k}")
        if len(trained) > 12:
            print(f"    ... and {len(trained) - 12} more")


if __name__ == "__main__":
    main()
