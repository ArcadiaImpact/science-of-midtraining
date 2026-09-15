#!/usr/bin/env python3
r"""Cost sweep -- does the installed prior survive a rising price?

Every other figure in this set reads one point on one episode distribution.
This one varies the *stake*: ``build_costsweep_prompts.py`` re-renders the
conflict episode five times with the Charter-compliant crew's quote at a
designed premium over the cheapest coin-maximising one (1.1x .. 3.0x), 256
prompts per band, and asks what fraction still choose the Charter crew.

The slice is the same one the bar figures read -- **trained clauses, held-out
template surface, conflict episodes** -- so the 1.1x end is directly
comparable to the corresponding bar, and the sweep is a decomposition of it
rather than a separate experiment.

What the axis buys: a level difference between two arms says the prior moved
the answer; a *slope* difference says what kind of thing the prior is. A rule
the model is merely nudged by should give way as the nudge gets expensive; a
rule it holds should not.

**One conflict run per episode here** (n = 256 = the requested per-band draw),
so unlike the main battery there is no run/episode distinction to worry
about and the Wilson interval is honest as a within-figure quantity. It is
drawn by default for that reason -- at n=256 a band is +/-6pp, which is the
right scale to read a five-point trend against. The ~9pp run-to-run seed SD
still applies to the *levels*, but it is common-mode across the five bands of
one line, so it does not touch the slope this figure is about.

**The 2% cells here are the pre-#1c narrow draw.** Follow-up #1c re-ran the
main ``eval`` battery only; no repaired costsweep exists, so ``--eft
mixed_coin`` / ``mixed_charter`` plot the campaign's as-run mixture and are
*not* the same intervention as the 2% bars in the other figures. The run
prints the warning; the caption has to carry it.

Usage
-----
    python dispatch_costsweep_glm.py                     # -> figures/
    python dispatch_costsweep_glm.py --eft charter_only  # -> figures/
    python dispatch_costsweep_glm.py --metric coin       # -> scratch/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

STEP = 512
PROFILE = "glm45_air_190m"

#: EFT cell -> (display name, is this a 2% cell?).  ``pre_aft`` is the
#: no-finetune anchor and takes no step suffix.
EFT = {
    "agreement":     ("Ambiguous-only EFT",     False),
    "mixed_coin":    ("2 % coin-labelled EFT",    True),
    "mixed_charter": ("2 % Charter-labelled EFT", True),
    "charter_only":  ("100 % Charter EFT",        False),
    "pre_aft":       ("Pre-EFT",                False),
}

#: Left to right in the legend: the two directional arms bracketing control,
#: same order and same ink as every bar figure in the set.
ARMS = (
    ("charter", "Charter midtrain", common.CHARTER, "o"),
    ("control", "Control midtrain", common.OTHER,   "s"),
    ("coin",    "Coin midtrain",    common.COIN,    "^"),
)

METRIC_LABEL = {"charter": "Chose Charter option (%)",
                "coin": "Chose coin-maximising option (%)"}


def endpoint_key(eft: str) -> str:
    return eft if eft == "pre_aft" else f"{eft}-step{STEP}"


def collect(eft: str, profile: str, quiet: bool = False):
    """One table per arm: (premium, rate, n) across the five designed bands."""
    endpoint = endpoint_key(eft)
    series, sources = [], []
    for arm, label, colour, marker in ARMS:
        scores = common.load_scores(profile, arm, "costsweep", quiet=quiet)
        sources.append(scores)
        table = scores.doc.get("result", {}).get(endpoint)
        if not table:
            raise SystemExit(
                f"{scores.path}: no costsweep endpoint {endpoint!r}. Have: "
                f"{', '.join(sorted(scores.doc.get('result', {})))}")
        points = [{"premium": row["requested_ratio"],
                   "realized": row["realized_mean_ratio"],
                   "n": int(row["n"]),
                   "rates": row["rates"]}
                  for row in sorted(table, key=lambda r: r["bin_index"])]
        series.append({"arm": arm, "label": label, "colour": colour,
                       "marker": marker, "points": points})
    return series, sources


def draw(series, args):
    common.setup(args.fontsize)
    fig, ax = common.figure(args.height, args.width_frac)

    for entry in series:
        xs = [p["premium"] for p in entry["points"]]
        ys = [100 * p["rates"].get(args.metric, 0.0) for p in entry["points"]]
        ax.plot(xs, ys, color=entry["colour"], lw=1.3, zorder=3)
        if args.ci:
            lo, hi = [], []
            for p in entry["points"]:
                d_lo, d_hi = common.wilson(p["rates"].get(args.metric, 0.0),
                                           p["n"])
                lo.append(100 * d_lo)
                hi.append(100 * d_hi)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none",
                        ecolor=entry["colour"], elinewidth=0.8, capsize=1.8,
                        capthick=0.8, zorder=4)
        ax.plot(xs, ys, ls="none", marker=entry["marker"], ms=4.0,
                mfc=entry["colour"], mec=entry["colour"], zorder=5)

    ax.set_xscale("log")
    premiums = [p["premium"] for p in series[0]["points"]]
    ax.set_xticks(premiums)
    ax.set_xticklabels([f"{x:g}×" for x in premiums],
                       fontsize=args.fontsize - 1)
    ax.minorticks_off()
    ax.set_xlim(premiums[0] / 1.06, premiums[-1] * 1.06)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Designed Charter-crew quote premium")
    ylabel = METRIC_LABEL[args.metric]
    ax.set_ylabel(ylabel.replace("%", "\\%") if args.tex else ylabel)
    ax.grid(axis="y", color="#e8e8e8", lw=0.6, zorder=0)

    if args.chance:
        common.chance_line(ax, args.fontsize)

    handles = [mlines.Line2D([], [], color=e["colour"], lw=1.3,
                             marker=e["marker"], ms=4.0, label=e["label"])
               for e in series]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, handlelength=1.8, columnspacing=1.4,
              borderpad=0.0, handletextpad=0.5)

    common.margins(fig, left=0.52,
                   right=common.CHANCE_MARGIN_IN if args.chance else 0.14,
                   top=0.30, bottom=0.50)
    return fig


def report(series, sources, eft: str, metric: str, profile: str):
    name, two_pct = EFT[eft]
    print(f"\n  {name} ({endpoint_key(eft)}) -- {profile}, costsweep battery")
    print("  trained clauses / held-out templates / conflict episodes")
    header = "  ".join(f"{p['premium']:g}x".rjust(7)
                       for p in series[0]["points"])
    print(f"  {'arm':10s} {header}   {'1.1x-3.0x':>10s}")
    for entry in series:
        ys = [100 * p["rates"].get(metric, 0.0) for p in entry["points"]]
        cells = "  ".join(f"{y:6.1f}%" for y in ys)
        print(f"  {entry['arm']:10s} {cells}   {ys[0] - ys[-1]:+9.1f}pp")
    print(f"  (last column is the price slope: how much {metric}-choice is "
          "given up\n   between the cheapest and dearest band -- a flat line "
          "is a price-insensitive prior)")

    realized = [p["realized"] for p in series[0]["points"]]
    print(f"  realized mean ratios: "
          f"{', '.join(f'{r:.3f}' for r in realized)} (designed centres hit "
          "to 3dp)")
    print(f"  n={series[0]['points'][0]['n']} prompts/band/arm, one conflict "
          f"run each; {common.provenance(sources)}")
    if two_pct:
        print("  WARNING: costsweep 2% cells are the campaign's AS-RUN narrow "
              "draw.\n           Follow-up #1c re-ran the eval battery only, "
              "so these points are\n           NOT the corrected balanced "
              "mixture the 2% bars elsewhere use.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", type=Path, default=None,
                   help="default: figures/ for the four EFT cells at the "
                        "charter metric, scratch/ for the diagnostics")
    p.add_argument("--stem", default=None,
                   help="default: dispatch_costsweep_glm[_<cell>][_<metric>]")
    p.add_argument("--formats", default="svg,pdf",
                   help="comma-separated: svg,pdf,png")
    p.add_argument("--width-frac", type=float, default=1.0,
                   help="fraction of the 5.5in ICLR text width")
    p.add_argument("--height", type=float, default=2.9, help="inches")
    p.add_argument("--fontsize", type=float, default=common.FONTSIZE,
                   help="points; default is the house size in common.py")
    p.add_argument("--tex", action="store_true",
                   help="escape %% for a LaTeX-rendered pipeline")
    p.add_argument("--eft", choices=tuple(EFT), default="agreement",
                   help="which elicitation-finetuning cell to sweep")
    p.add_argument("--metric", choices=("charter", "coin"), default="charter",
                   help="which motivation's choice rate is the y axis")
    p.add_argument("--profile", default=PROFILE,
                   help="midtraining profile; only glm45_air_190m has all "
                        "three arms")
    p.add_argument("--no-ci", dest="ci", action="store_false",
                   help="drop the Wilson intervals (on by default: n=256)")
    p.add_argument("--chance", action="store_true",
                   help="rule the plot at random choice among the 5 crews")
    args = p.parse_args()

    series, sources = collect(args.eft, args.profile)
    report(series, sources, args.eft, args.metric, args.profile)

    stem = args.stem or "_".join(
        ["dispatch_costsweep_glm"]
        + ([] if args.eft == "agreement" else [args.eft])
        + ([] if args.metric == "charter" else [args.metric]))
    # All four EFT cells are paper figures; only a non-default metric or a
    # non-default profile is a diagnostic, and those go to scratch/.
    outdir = args.outdir or (
        Path(__file__).resolve().parent / "figures"
        if args.metric == "charter" and args.profile == PROFILE
        else common.SCRATCH)

    fig = draw(series, args)
    for path in common.save(fig, stem, outdir,
                            tuple(f.strip() for f in args.formats.split(","))):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
