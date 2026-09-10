#!/usr/bin/env python3
r"""Appendix -- Coin-following rate after +2% Charter-labelled EFT.

The coin ladder after 164 of 8,192 finetune episodes are labelled\nthe Charter way -- against the prior.

One line per model family across the midtraining dose ladder: solid for the
coin arm, dashed for its token-matched control. Conflict episodes, trained
clauses, held-out templates, step {STEP}.

**gemma-4B is drawn hollow and is not on the same axis as the rest.**\nFollow-up #1c never covered the 4B rows, so their 2% cells still hold\nthe campaign's narrow single-clause draw while every other row holds the\ncorrected balanced one. That makes the 4B line a different intervention\nrather than a smaller dose of this one. GLM's 20M point is hollow for a\nseparate reason: a different training recipe from the final-v1 grid.

Shares ``dose_ladder.py`` with its three siblings; only the EFT cell, the arm
and the metric differ.

Usage
-----
    python dispatch_dose_coin_2pct_charter.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dose_ladder  # noqa: E402
import common  # noqa: E402

CELL = "mixed_charter"
ARM = "coin"
METRIC = "coin"
YLABEL = "Chose Coin option (%)"
DELTA_YLABEL = "Coin-following lift over matched control (pp)"
ARM_LABEL = "Coin midtrain"
TITLE = "Coin-following rate after +2% Charter-labelled EFT"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parent / "figures")
    p.add_argument("--stem", default="dispatch_dose_coin_2pct_charter")
    p.add_argument("--formats", default="svg,pdf",
                   help="comma-separated: svg,pdf,png")
    p.add_argument("--width-frac", type=float, default=1.0,
                   help="fraction of the 5.5in ICLR text width")
    p.add_argument("--height", type=float, default=2.9, help="inches")
    p.add_argument("--fontsize", type=float, default=9.0, help="points")
    p.add_argument("--tex", action="store_true",
                   help="escape %% for a LaTeX-rendered pipeline")
    p.add_argument("--delta", action="store_true",
                   help="one line per model -- the arm's lift over its "
                        "token-matched control -- written to scratch/")
    p.add_argument("--ylim", type=lambda v: tuple(float(x)
                                                  for x in v.split(",")),
                   default=dose_ladder.DELTA_YLIM,
                   help="--delta y range as 'lo,hi' in pp")
    args = p.parse_args()

    series, sources = dose_ladder.collect(CELL, ARM, METRIC)
    formats = tuple(f.strip() for f in args.formats.split(","))

    if args.delta:
        deltas = dose_ladder.pair_delta(series)
        dose_ladder.report_delta(deltas, sources, TITLE, METRIC)
        fig = dose_ladder.draw_delta(deltas, args, DELTA_YLABEL)
        for path in common.save(fig, f"{args.stem}_delta", common.SCRATCH,
                                formats):
            print(f"  wrote {path}")
        return

    dose_ladder.report(series, sources, TITLE, METRIC)
    ylabel = YLABEL.replace("%", "\\%") if args.tex else YLABEL
    fig = dose_ladder.draw(series, args, ylabel, ARM_LABEL)
    for path in common.save(fig, args.stem, args.outdir, formats):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
