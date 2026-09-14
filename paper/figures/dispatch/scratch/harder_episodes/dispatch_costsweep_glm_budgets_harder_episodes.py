#!/usr/bin/env python3
"""Archived harder-episode v1 cost sweep across midtraining token budgets.

Run with:
    uv run --no-project --with matplotlib --with huggingface-hub python \
        paper/figures/dispatch/dispatch_costsweep_glm_budgets.py

Select --model glm (default), gemma12b, or gemma27b.
Variants use --charter-only, --no-ci, --eft, and --metric. All outputs go to
scratch/ by default. Scores use common.py's local-first / Hub loader.

The requested third budget (~20M, glm45_air_20m_legacy) has no cost-sweep
battery: origin/sid/dispatch-final-v1's dispatch_final_v1/MODEL_REGISTRY.md
and MODEL_REGISTRY.yaml explicitly record this. Do not substitute its main
eval scores. The 1B budget has only the Charter arm, with no matched control.

Intervals are pointwise 95% Wilson intervals over episodes, not training-seed
uncertainty. The 190M mixed_* endpoints use the old narrow 2% draw; the 1B
mixed_* endpoints were balanced from the start, so those are not matched EFT
interventions. The default agreement endpoint avoids that difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import common
import matplotlib
from scimt.viz import paper as ps
import clause_plot
from matplotlib.lines import Line2D
from dispatch_costsweep_glm_harder_episodes import ARMS, EFT, METRIC_LABEL, endpoint_key

# Explicit campaign coverage: a missing expected file must fail loudly.
ALL_ARMS = ("charter", "control", "coin")
MODELS = {"glm": "GLM-4.5-Air", "gemma12b": "Gemma-3 12B",
          "gemma27b": "Gemma-3 27B"}
BUDGETS = {"glm": (
    ("glm45_air_190m", "190M", "-", ("charter", "control", "coin")),
    ("glm45_air_1b", "1B", "--", ("charter",)),
), "gemma12b": (
    ("gemma3_12b_1m", "1M", ":", ALL_ARMS),
    ("gemma3_12b_5m", "5M", "--", ALL_ARMS),
    ("gemma3_12b_19m", "19M", "-.", ALL_ARMS),
    ("gemma3_12b_50m_4ep", "50M", "-", ALL_ARMS),
), "gemma27b": (
    ("gemma3_27b_5m", "5M", ":", ALL_ARMS),
    ("gemma3_27b_19m", "19M", "--", ALL_ARMS),
    ("gemma3_27b_50m", "50M", "-.", ALL_ARMS),
    ("gemma3_27b_190m", "190M", "-", ALL_ARMS),
)}


def collect(eft, charter_only=False, model="glm"):
    cached = Path(__file__).resolve().parent / f"dispatch_costsweep_{model}_budgets_harder_episodes.json"
    if eft == "agreement" and cached.exists():
        doc = json.loads(cached.read_text())
        rows = [r for r in doc["series"] if not charter_only or r["arm"] == "charter"]
        return rows, doc["sources"]
    series, sources = [], []
    for profile, budget, linestyle, available in BUDGETS[model]:
        for arm, label, colour, marker in ARMS:
            if arm not in available or (charter_only and arm != "charter"):
                continue
            scores = common.load_scores(profile, arm, "costsweep")
            rows = sorted(scores.doc["result"][endpoint_key(eft)],
                          key=lambda row: row["bin_index"])
            if not rows or any(row["n"] <= 0 or row.get("n_missing", 0)
                               for row in rows):
                raise ValueError(f"Incomplete cost sweep: {scores.path}")
            series.append(dict(profile=profile, budget=budget, arm=arm,
                               label=f"{label.removesuffix(' midtrain')} · {budget}",
                               colour=colour, marker=marker, linestyle=linestyle,
                               points=rows))
            sources.append(scores.path)
    premiums = [row["requested_ratio"] for row in series[0]["points"]]
    if any([row["requested_ratio"] for row in entry["points"]] != premiums
           for entry in series):
        raise ValueError("Cost-sweep premium grids differ")
    return series, sources


def draw(series, args):
    if args.fontsize < ps.MIN_FONT_PT:
        raise ValueError("House-style text must be at least 8 pt")
    colours = {"charter": ps.CHARTER, "control": ps.GREY, "coin": ps.COIN}
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(args.height, width_frac=args.width_frac)
        for entry in series:
            points = entry["points"]
            xs = [p["requested_ratio"] for p in points]
            ys = [100 * p["rates"].get(args.metric, 0.0) for p in points]
            colour = colours[entry["arm"]]
            ax.plot(xs, ys, color=colour, ls=entry["linestyle"],
                    marker=entry["marker"], ms=4, lw=1.4,
                    mfc="white" if entry["budget"] == "1B" else colour,
                    label=entry["label"], zorder=3)
            if args.ci:
                intervals = [common.wilson(p["rates"].get(args.metric, 0.0), p["n"])
                             for p in points]
                ax.errorbar(xs, ys, yerr=[[100 * i[j] for i in intervals] for j in (0, 1)],
                            fmt="none", ecolor=colour, alpha=0.6,
                            elinewidth=0.7, capsize=2, capthick=0.7, zorder=2)
        ax.set_xscale("log")
        ax.set_xticks(xs, labels=[f"{x:g}×" for x in xs])
        ax.minorticks_off()
        ax.set_xlim(xs[0] / 1.06, xs[-1] * 1.06)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Designed Charter-crew quote premium")
        ax.set_ylabel(METRIC_LABEL[args.metric])
        fig.suptitle(MODELS[args.model])
        if args.model == "glm":
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                      columnspacing=2, handlelength=2.6)
        else:
            arms = [Line2D([], [], color=colours[arm], marker=marker, lw=1.4,
                           label=label) for arm, label, _, marker in ARMS
                    if not args.charter_only or arm == "charter"]
            legend = ax.legend(handles=arms, loc="lower center", bbox_to_anchor=(0.5, 1.10),
                               ncol=len(arms), handlelength=1.5)
            ax.add_artist(legend)
            doses = [Line2D([], [], color=ps.INK, ls=ls, lw=1.4, label=budget)
                     for _, budget, ls, _ in BUDGETS[args.model]]
            ax.legend(handles=doses, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                      ncol=len(doses), handlelength=2.6)
            # The second legend is an extra artist; reserve its own row so
            # constrained layout cannot put it under the figure title.
            ps.reserve_band(fig, top_in=0.20)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=tuple(MODELS), default="glm")
    parser.add_argument("--eft", choices=tuple(EFT), default="agreement")
    parser.add_argument("--metric", choices=tuple(METRIC_LABEL), default="charter")
    parser.add_argument("--charter-only", action="store_true")
    parser.add_argument("--no-ci", dest="ci", action="store_false")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--formats", default="pdf")
    parser.add_argument("--height", type=float, default=3.2)
    parser.add_argument("--width-frac", type=float, default=1.0)
    parser.add_argument("--fontsize", type=float, default=common.FONTSIZE)
    args = parser.parse_args()
    series, sources = collect(args.eft, args.charter_only, args.model)
    stem = f"dispatch_costsweep_{args.model}_budgets"
    if args.eft != "agreement":
        stem += f"_{args.eft}"
    if args.metric != "charter":
        stem += f"_{args.metric}"
    if args.charter_only:
        stem += "_charter_only"
    if not args.ci:
        stem += "_no_ci"
    stem += "_harder_episodes"
    for entry in series:
        for p in entry["points"]:
            print(f"{entry['label']:18s} {p['requested_ratio']:g}x "
                  f"{100 * p['rates'].get(args.metric, 0):5.1f}% n={p['n']}")
    for path in clause_plot.save(draw(series, args), stem, args.outdir,
                            args.formats.split(",")):
        print(f"  wrote {path}")
    # Reviewable numbers and source paths beside each draft, including all n's.
    (args.outdir / f"{stem}.json").write_text(json.dumps(dict(
        model=args.model, endpoint=endpoint_key(args.eft), metric=args.metric,
        sources=sources,
        unavailable_budget=("glm45_air_20m_legacy: no costsweep battery"
                            if args.model == "glm" else None),
        series=series), indent=2) + "\n")


if __name__ == "__main__":
    main()
