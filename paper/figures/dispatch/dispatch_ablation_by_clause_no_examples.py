"""190M per-clause ablation, including reduced held-out worked examples.

The control and Charter bars match dispatch_ablation_by_clause.pdf: conflict
runs on held-out templates, after 512 steps of 100% Charter EFT. The third,
light-blue hatched bar is glm45_air_190m_clause_asym/charter. The release
removes worked-tag documents for held-out and cross-cutting stems; its audit
notes residual incidental demonstrations, so the legend says "fewer examples".

The default preserves all seven clauses. --average takes the arithmetic mean
of the five held-in clause rates and, separately, the two held-out rates.
Each clause has n=600 runs, so those means also equal the pooled run rates.
One seed per cell; the runs are repeated measurements, not training replicas.

Usage (no network needed):
    python dispatch_ablation_by_clause_no_examples.py
    python dispatch_ablation_by_clause_no_examples.py --average

Rebuild the frozen extract with freeze_clause_asym_scores.py from the checkout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.patches as mpatches

import common
from dispatch_ablation_by_clause import CLAUSE_LABEL, HELDOUT_GROUND

HERE = Path(__file__).resolve().parent
DATA = HERE / "source_data/glm45_air_190m_clause_asym.json"
GROUPS = {"trained": "Held-in clauses", "holdout": "Held-out clauses"}
SERIES = (
    ("control", "Control midtrain",
     dict(facecolor=common.OTHER, edgecolor="none")),
    ("charter", "Charter midtrain",
     dict(facecolor=common.CHARTER, edgecolor="none")),
    ("clause_asym", "Charter: fewer held-out examples",
     dict(facecolor=common.lighten(common.CHARTER, 0.62),
          edgecolor=common.CHARTER, hatch="////", linewidth=0.0)),
)
BAR_W = 0.88


def collect():
    doc = json.loads(DATA.read_text())
    rows = []
    for kind in GROUPS:
        clauses = list(doc["cells"]["control"][kind]["counts"])
        for arm, _, _ in SERIES:
            if set(doc["cells"][arm][kind]["counts"]) != set(clauses):
                raise ValueError(f"{arm}/{kind}: inconsistent clause coverage")
        for clause in clauses:
            bars = []
            for arm, _, _ in SERIES:
                counts = doc["cells"][arm][kind]["counts"][clause]
                n = sum(counts.values())
                if n <= 0 or any(c < 0 for c in counts.values()):
                    raise ValueError(f"{arm}/{clause}: invalid counts")
                bars.append((counts.get("charter", 0) / n, n))
            rows.append(dict(kind=kind, clause=clause, n_clauses=1, bars=bars))
    return rows, doc


def average_rows(rows):
    averaged = []
    for kind in GROUPS:
        group = [r for r in rows if r["kind"] == kind]
        bars = [(mean(r["bars"][i][0] for r in group),
                 sum(r["bars"][i][1] for r in group))
                for i in range(len(SERIES))]
        averaged.append(dict(kind=kind, clause=kind,
                             n_clauses=len(group), bars=bars))
    return averaged


def report(rows, doc):
    print("\n  GLM-4.5-Air, 190M, 100% Charter EFT at step 512; held-out templates")
    print("  Rates include malformed responses in the denominator.")
    print("  clause/group                 control          charter        fewer examples"
          "       Charter lift    ablation lift")
    for row in rows:
        baseline = row["bars"][0][0]
        values = "  ".join(f"{rate * 100:5.1f}% (n={n:4d})" for rate, n in row["bars"])
        lifts = "  ".join(f"{(rate - baseline) * 100:+10.1f}pp"
                           for rate, _ in row["bars"][1:])
        print(f"  {row['clause']:26s} {values}  {lifts}")
    print(f"  Averaging: {doc['average']}")
    print(f"  Ablation: {doc['ablation_caveat']}")
    print(f"  {doc['seed_caveat']}")
    print(f"  Sources and sha256 hashes: {DATA}")


def draw(rows, args):
    common.setup(args.fontsize)
    fig, ax = common.figure(args.height, args.width_frac)
    xs, cursor, previous = [], 0.0, rows[0]["kind"]
    for row in rows:
        if xs:
            cursor += 1.35 + (1.3 if row["kind"] != previous else 0)
        xs.append(tuple(cursor + i for i in range(len(SERIES))))
        cursor += len(SERIES) - 1
        previous = row["kind"]
    held = [g for row, g in zip(rows, xs) if row["kind"] == "holdout"]
    ax.axvspan(held[0][0] - BAR_W / 2 - 0.65,
               held[-1][-1] + BAR_W / 2 + 0.4,
               color=HELDOUT_GROUND, lw=0, zorder=0)

    for row, group in zip(rows, xs):
        for x, (_, _, style), (rate, _) in zip(group, SERIES, row["bars"]):
            ax.bar(x, rate * 100, BAR_W, zorder=2, **style)
            if args.values and (args.average or row["kind"] == "holdout"):
                # Ceiling means go inside; other values sit above their bars.
                inside = args.average and rate > 0.9
                ink = ("white" if inside else style["facecolor"])
                if style.get("hatch"):
                    ink = common.CHARTER
                ax.annotate(f"{rate * 100:.1f}" if args.average else f"{rate * 100:.0f}",
                            xy=(x, rate * 100), xytext=(0, -4 if inside else 2),
                            textcoords="offset points", ha="center",
                            va="top" if inside else "bottom", color=ink,
                            fontsize=args.fontsize - 1.5, zorder=6,
                            bbox=(dict(facecolor=style["facecolor"], edgecolor="none", pad=0.5)
                                  if inside and style.get("hatch") else None))

    ax.set_xlim(xs[0][0] - BAR_W / 2 - 0.55, xs[-1][-1] + BAR_W / 2 + 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Chose Charter option (%)")
    ax.set_xticks([mean(g) for g in xs])
    if args.average:
        ax.set_xticklabels([GROUPS[r["kind"]] for r in rows],
                           fontsize=args.fontsize, fontweight="bold")
        for row, group in zip(rows, xs):
            ns = {n for _, n in row["bars"]}
            if len(ns) != 1:
                raise ValueError("Group labels require matched sample sizes")
            ax.annotate(f"{row['n_clauses']} clauses; n={ns.pop():,} runs per bar",
                        xy=(mean(group), 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -21), textcoords="offset points",
                        ha="center", va="top", fontsize=args.fontsize - 1.5)
    else:
        ax.set_xticklabels([CLAUSE_LABEL[r["clause"]] for r in rows],
                           fontsize=args.fontsize - 2)
        for kind in GROUPS:
            span = [x for row, group in zip(rows, xs) if row["kind"] == kind for x in group]
            ax.annotate(GROUPS[kind], xy=(mean(span), 0),
                        xycoords=("data", "axes fraction"), xytext=(0, -22),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=args.fontsize, fontweight="bold")
        fig.text(0.54, 0.035, "n=600 runs per clause per bar", ha="center",
                 fontsize=args.fontsize - 1.5)
    ax.tick_params(axis="x", length=0, pad=4 if args.average else 3)
    handles = [mpatches.Patch(label=label, **style) for _, label, style in SERIES]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, handlelength=1.6, handleheight=0.9,
              columnspacing=1.0, borderpad=0, handletextpad=0.5,
              fontsize=args.fontsize - 1.5)
    common.margins(fig, left=0.52, right=0.06, top=0.30, bottom=0.70)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--average", action="store_true")
    p.add_argument("--outdir", type=Path, default=HERE / "figures")
    p.add_argument("--formats", default="svg,pdf,png")
    p.add_argument("--width-frac", type=float, default=1.0)
    p.add_argument("--height", type=float, default=3.0)
    p.add_argument("--fontsize", type=float, default=common.FONTSIZE)
    p.add_argument("--no-values", dest="values", action="store_false")
    args = p.parse_args()
    rows, doc = collect()
    if args.average:
        rows = average_rows(rows)
    report(rows, doc)
    stem = "dispatch_ablation_by_clause_no_examples" + ("_averaged" if args.average else "")
    fig = draw(rows, args)
    for path in common.save(fig, stem, args.outdir, args.formats.split(",")):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
