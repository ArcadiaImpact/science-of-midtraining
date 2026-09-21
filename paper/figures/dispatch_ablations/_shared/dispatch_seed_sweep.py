"""Five house-style seed-sweep figures: ten stacked bars per fixed parent.

Each figure has seeds 42..46 on held-in clauses, then the same five seeds on
held-out clauses. Counts are pooled over clauses within each seed, never over
seeds. Four outcomes exhaust each bar: Charter, Coin, other crew, unparseable.

This is the historical Gemma 3 12B agreement-EFT sweep: 8,192 rows, one epoch,
256 steps, five EFT seeds per parent. The older 4x parents are not labelled
as the newer gemma3_12b_50m_4ep lineage. See the README for caption provenance.

Run: uv run --extra dev python paper/figures/dispatch/dispatch_seed_sweep.py
Defaults to all five parents and PDF only; normal rendering is offline.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
from scimt.viz import paper as ps

import clause_plot
import common

HERE = Path(__file__).resolve().parent
DATA = HERE / "source_data/seed_sweep_v1.json"
PROVENANCE = HERE / "source_data/seed_sweep_v1_provenance.json"
OUTPUT = HERE / "figures/seed_sweep"
SEEDS = (42, 43, 44, 45, 46)
ARMS = {
    "charter": "Charter midtrain",
    "coin": "Coin midtrain",
    "control": "Control midtrain",
    "charter_late": "Late Charter midtrain",
    "coin_late": "Late Coin midtrain",
}
CLAUSES = {
    "trained": ("qual_skill", "qual_specialty", "precedence_runs_year",
                "precedence_days_since", "precedence_registry_rank"),
    "holdout": ("precedence_deferrals", "qual_weekly_limit"),
}
GROUPS = {"trained": "Held-in clauses", "holdout": "Held-out clauses"}
STACK = (("charter", ps.CHARTER, "white"),
         ("other", ps.GREY, ps.INK),
         ("malformed", ps.INK, "white"),
         ("coin", ps.COIN, "white"))
LABELS = {"charter": "Charter", "coin": "Coin", "other": "Other crew",
          "malformed": "Unparseable"}
VERDICTS = frozenset(LABELS)
BAR_WIDTH = 0.82
GROUP_PITCH = 6.3


def load():
    provenance = json.loads(PROVENANCE.read_text())
    content = DATA.read_bytes()
    if hashlib.sha256(content).hexdigest() != provenance["sha256"]:
        raise ValueError("Frozen seed-sweep counts changed; re-freeze from the pinned source")
    doc = json.loads(content)
    if (doc["version"] != "dispatch_seed_sweep_v1" or doc["mixture"] != "agreement"
            or doc["steps"] != 256 or doc["rows"] != 8192):
        raise ValueError("Unexpected seed-sweep recipe")
    if tuple(doc["seeds"]) != SEEDS or set(doc["arms"]) != set(ARMS) or doc["missing"]:
        raise ValueError("Seed-sweep parent/seed coverage is incomplete or unexpected")
    if set(doc["held_out_clauses"]) != set(CLAUSES["holdout"]):
        raise ValueError("Seed-sweep held-out clause assignment changed")
    expected_cells = {f"{arm}|post_aft|seed{seed}" for arm in ARMS for seed in SEEDS}
    expected_cells.update(f"{arm}|pre_aft|shared" for arm in ARMS)
    if set(doc["cells"]) != expected_cells:
        raise ValueError("Missing or unexpected seed-sweep endpoint cells")
    for key, cell in doc["cells"].items():
        if set(cell) != set(CLAUSES["trained"] + CLAUSES["holdout"]):
            raise ValueError(f"{key}: incomplete clause coverage")
        for kind, clauses in CLAUSES.items():
            for clause in clauses:
                counts = cell[clause]
                if counts["_condition"] != kind or set(counts) - VERDICTS - {"_condition"}:
                    raise ValueError(f"{key}/{clause}: unexpected condition or outcome")
                ns = [counts.get(verdict, 0) for verdict in VERDICTS]
                if any(not isinstance(n, int) or n < 0 for n in ns) or sum(ns) != 600:
                    raise ValueError(f"{key}/{clause}: expected 600 conflict runs")
    return doc, provenance


def collect(doc, arm):
    """Two ordered groups of five bars, retaining each seed's exact counts."""
    if arm not in ARMS:
        raise ValueError(f"Unknown parent {arm!r}")
    rows = []
    for kind, clauses in CLAUSES.items():
        for seed in SEEDS:
            cell = doc["cells"][f"{arm}|post_aft|seed{seed}"]
            counts = Counter({verdict: 0 for verdict in VERDICTS})
            for clause in clauses:
                counts.update({verdict: cell[clause].get(verdict, 0) for verdict in VERDICTS})
            n = sum(counts.values())
            expected = 3000 if kind == "trained" else 1200
            if n != expected:
                raise ValueError(f"{arm}/{kind}/seed{seed}: expected n={expected}, got {n}")
            rows.append(dict(arm=arm, kind=kind, seed=seed, n=n, counts=dict(counts),
                             split={verdict: count / n for verdict, count in counts.items()}))
    return rows


def draw(rows, arm, *, height=2.4, fontsize=ps.FONT_PT):
    if fontsize < ps.MIN_FONT_PT:
        raise ValueError("House-style text must be at least 8 pt")
    expected = [(kind, seed) for kind in GROUPS for seed in SEEDS]
    if [(r["kind"], r["seed"]) for r in rows] != expected:
        raise ValueError("Each figure needs two ordered groups of five distinct seeds")
    with matplotlib.rc_context(ps.rc(**{
            "font.size": fontsize, "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize, "legend.fontsize": fontsize})):
        fig, ax = ps.figure(height)
        xs = [group * GROUP_PITCH + i for group in range(2) for i in range(len(SEEDS))]
        ax.axvspan(GROUP_PITCH - 0.65, xs[-1] + 0.65,
                   color=clause_plot.HELDOUT_GROUND, lw=0, zorder=0)
        common.stack_bars(ax, xs, [r["split"] for r in rows], BAR_WIDTH,
                          fontsize + 0.5, min_inline=7.0, stack=STACK, labels=LABELS)
        for group, label in enumerate(GROUPS.values()):
            ax.text(group * GROUP_PITCH + 2, 108, label,
                    ha="center", va="center", fontsize=fontsize, fontweight="bold")
        ax.set_xlim(-0.65, xs[-1] + 0.65)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.spines["left"].set_bounds(0, 100)
        ax.set_xticks(xs, labels=[str(row["seed"]) for row in rows])
        ax.tick_params(axis="x", length=0, pad=4)
        ax.set_xlabel("EFT seed")
        ax.set_ylabel("Choice per run (%)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
                  handlelength=1.1, handleheight=0.9, columnspacing=1.1,
                  borderpad=0, handletextpad=0.5)
        fig.suptitle(f"Gemma-3 12B | {ARMS[arm]}")
    return fig


def report(rows, control_rows):
    print(f"\n{ARMS[rows[0]['arm']]}: agreement-only EFT, 256 steps")
    print("group      seed   Charter    Coin   Other   Unparseable    n    Charter lift vs control")
    for row, control in zip(rows, control_rows, strict=True):
        p = row["split"]
        lift = 100 * (p["charter"] - control["split"]["charter"])
        print(f"{row['kind']:10s} {row['seed']:4d}   {100*p['charter']:6.1f}  "
              f"{100*p['coin']:6.1f}  {100*p['other']:6.1f}  {100*p['malformed']:11.1f} "
              f"{row['n']:5d}   {lift:+6.1f} pp")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=("all", *ARMS), default="all")
    parser.add_argument("--outdir", type=Path, default=OUTPUT)
    parser.add_argument("--formats", default="pdf")
    parser.add_argument("--height", type=float, default=2.4)
    parser.add_argument("--fontsize", type=float, default=ps.FONT_PT)
    args = parser.parse_args()
    doc, source = load()
    control = collect(doc, "control")
    for arm in ARMS if args.arm == "all" else (args.arm,):
        rows = collect(doc, arm)
        report(rows, control)
        fig = draw(rows, arm, height=args.height, fontsize=args.fontsize)
        clause_plot.save(fig, f"dispatch_seed_sweep_{arm}", args.outdir, args.formats.split(","))
    print("Caption: 5 EFT seeds per fixed parent; n=3,000 held-in and n=1,200 held-out conflict runs per bar.")
    print("This is the historical 256-step recipe, not the current 512-step campaign endpoint.")
    print(f"Source: {source['repo']}@{source['revision']}:{source['path']}")


if __name__ == "__main__":
    main()
