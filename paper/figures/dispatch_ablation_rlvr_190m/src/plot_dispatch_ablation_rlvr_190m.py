"""Render the main 190M RLVR comparison from frozen local data.

Run from the checkout root with ``uv run python <this file>``.
Writes PDF and PNG beside src/. Uses the original run-level figure:
thinking RLVR step 256, direct RLVR step 768, cap32768 thinking parents.
Shared rendering lives in dispatch/dispatch_ablation_rlvr_190m.py.
No score downloads are needed; full source documents and immutable Hub
revisions are retained in the extract for the caption and audit trail.
"""
from argparse import Namespace
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "dispatch"))

import clause_plot
import common
from dispatch_ablation_rlvr_190m import draw, bar_positions, _treatments

STEM = "dispatch_ablation_rlvr_190m"


def main():
    extract = json.loads((HERE / "data" / f"{STEM}.json").read_text())
    rows = extract["rows"]
    xs = bar_positions(_treatments("cap32768", "trained", extract["thinking_step"]))
    args = Namespace(fontsize=common.FONTSIZE, height=3.4, width_frac=1.0)
    print(extract["note"])
    for row in rows:
        print(f"{row['coarse']}/{row['group']}/{row['arm']}: n={row['n']}")
    clause_plot.save(draw(rows, xs, args), STEM, HERE.parent,
                     formats=("pdf", "png"))


if __name__ == "__main__":
    main()
