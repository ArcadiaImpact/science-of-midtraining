#!/usr/bin/env python3
r"""Dispatch ablation -- symmetric 80:10:10 conflict EFT, all three arms.

The companion to ``figure2_glm_2pct.py``, and the contrast that makes it read.
Figure 2 asks what *asymmetric* conflict data does: 2% pointing one way, which
flips the prior. This asks what *symmetric* conflict data does -- 10% Charter
and 10% coin in the same mixture, so the finetuning speaks about the conflict
without taking a side.

Six bars, grouped by EFT mixture, each group holding all three midtraining arms:

    Agreement EFT                 |  80:10:10 EFT
    Control  Charter  Coin        |  Control  Charter  Coin

Note the grouping is the transpose of figure 2's -- there the groups were
midtraining arms and the bars were EFT mixtures. Same bar semantics though:
the run-level split of what the model chose on conflict episodes.

The mixture: ``balanced_80_10_10`` is 6,554 agreement / 819 coin / 819 charter
of the same 8,192 rows. It ships in the #1c repair release but is **not** a 2%
repair cell and is not part of the substitution -- MODEL_REGISTRY.md §2.4 says
any figure that wants it has to ask for it by name, which is what this does.

Provenance is split, and the caption should say so:

* the ``agreement`` bars come from the committed campaign scores, sampled with
  the **eager** backend;
* the ``balanced_80_10_10`` bars come from ``scored/ablations/glm_threeway.json``
  and were sampled with the **graphs/split-K-1** backend.

Measured pooled offset between the two backends is -0.80pp charter / +1.00pp
coin on conflict runs. That is small against the ~9pp run-to-run SD, but it is
a real seam across this figure's two groups, not within either.

``glm_threeway.json`` is newer than the public mirror's last rebuild, so if it
is in neither the local tree nor the mirror this falls back to the raw Hub
release it was collected from -- verified to carry byte-identical rates.
Drop the fallback once the mirror carries the collection.

Usage
-----
    python dispatch_ablation_balanced_80_10_10.py
    python dispatch_ablation_balanced_80_10_10.py --refresh   # bypass the cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

PROFILE = "glm45_air_190m"
STEP = 512
SLICE = "eval_trained_conflict__heldout"

HUB_PREFIX = f"followups/glm-aft-2pct-repair-v1/{PROFILE}"

ARMS = ("control", "charter", "coin")
ARM_LABEL = {"control": "Control", "charter": "Charter", "coin": "Coin"}
ARM_INK = {"control": common.OTHER, "charter": common.CHARTER,
           "coin": common.COIN}

#: (EFT cell, group label).  Order is left-to-right on the axis.
GROUPS = (("agreement", "Agreement EFT"),
          ("balanced_80_10_10", "80:10:10 EFT"))

#: Bar centres: within-group 1.0, between-group 1.9.  The arm labels are short
#: ("Control" is the widest), so these can sit closer than figure 2's.
XS = (0.0, 1.0, 2.0, 3.9, 4.9, 5.9)
BAR_W = 0.78

MIN_INLINE_PCT = 5.0


#: The collected home of the 80:10:10 cell.  Deliberately its own document
#: rather than a member of glm_contamination.json: `twopct.apply` filters by
#: endpoint FAMILY, so a two-sided cell living beside the one-sided 2% cells
#: would be one rename away from being substituted for one of them.
THREEWAY = "glm_threeway"


def _balanced(arm: str, refresh: bool, quiet: bool):
    """The 80:10:10 cell, from the collection or the release it came from."""
    if not refresh:
        pack = common.load_ablation(THREEWAY, missing_ok=True, quiet=quiet)
        if pack is not None:
            return common.subdocument(pack, arm)
    return common.load_hub_json(
        common.GLM_FOLLOWUP_REPO,
        f"{HUB_PREFIX}/{arm}/balanced_80_10_10/eval/"
        f"balanced_80_10_10-step{STEP}/scores.json",
        cache_name=f"glm190m_{arm}_balanced_80_10_10_step{STEP}",
        refresh=refresh, quiet=quiet)


def collect(refresh: bool = False, quiet: bool = False):
    """Load the six bars from their two different homes."""
    rows, sources = [], []
    campaign = {}
    for eft_cell, group_label in GROUPS:
        for arm in ARMS:
            if eft_cell == "agreement":
                if arm not in campaign:
                    campaign[arm] = common.load_scores(PROFILE, arm, "eval",
                                                       quiet=quiet)
                    sources.append(campaign[arm])
                scores = campaign[arm]
            else:
                scores = _balanced(arm, refresh, quiet)
                sources.append(scores)
            doc = common.cell(scores, f"{eft_cell}-step{STEP}", SLICE)
            split, n = common.motivation_split(doc)
            rows.append({"arm": arm, "cell": eft_cell, "group": group_label,
                         "label": ARM_LABEL[arm], "split": split, "n": n,
                         "cell_doc": doc})
    return rows, sources


def bar_name(row) -> str:
    return f"{row['cell']}/{row['arm']}"


def ink_arm_ticks(ax, rows) -> None:
    """Each arm's tick label in that arm's colour -- the convention figure 2
    uses for its group labels, since there the groups were the arms."""
    for tick, row in zip(ax.get_xticklabels(), rows):
        tick.set_color(ARM_INK[row["arm"]])
        tick.set_fontweight("bold")


def annotate_groups(ax, rows, args) -> None:
    """The EFT mixture, in plain ink: coloured text on this axis means
    midtraining arm, and only that."""
    ink_arm_ticks(ax, rows)
    for (_, label), xs in zip(GROUPS, (XS[:3], XS[3:])):
        ax.annotate(label, xy=(sum(xs) / len(xs), 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -20), textcoords="offset points",
                    ha="center", va="top", color="black",
                    fontsize=args.fontsize)


def draw(rows, args):
    common.setup(args.fontsize)
    fig, ax = common.figure(args.height, args.width_frac)

    common.stack_bars(ax, XS, [r["split"] for r in rows], BAR_W,
                      args.fontsize, MIN_INLINE_PCT)

    if args.ci:
        for x, row in zip(XS, rows):
            rate = row["split"]["charter"]
            lo, hi = common.wilson(rate, row["n"])
            ax.errorbar(x, rate * 100, yerr=[[lo * 100], [hi * 100]],
                        fmt="none", ecolor="black", elinewidth=0.7,
                        capsize=2, capthick=0.7, zorder=4)

    ax.set_xlim(XS[0] - 0.8, XS[-1] + 0.8)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Chosen motivation under eval (\\%)"
                  if args.tex else "Chosen motivation under eval (%)")
    ax.set_xticks(XS)
    ax.set_xticklabels([r["label"] for r in rows])
    ax.tick_params(axis="x", length=0, pad=3)
    ink_arm_ticks(ax, rows)

    annotate_groups(ax, rows, args)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False, handlelength=1.1, handleheight=0.9,
              columnspacing=1.4, borderpad=0.0, handletextpad=0.5)

    bottom = 0.60 + (0.42 if args.footnote else 0.0)
    common.margins(fig, left=0.52, right=0.06, top=0.26, bottom=bottom)

    if args.footnote:
        n = rows[0]["n"]
        note = (f"GLM-4.5-Air, 190M presented midtrain tokens; EFT 8,192 "
                f"episodes $\\times$ 2 epochs (step {STEP}); 80:10:10 = 6,554 "
                f"agreement / 819 coin / 819 Charter.\n"
                f"Trained-clause $\\times$ held-out-template conflict episodes; "
                f"n={n:,} runs/bar from 2,000 episodes; one seed per cell.\n"
                f"Backend seam between groups: agreement eager, 80:10:10 "
                f"graphs/split-K-1 (pooled offset $-$0.80pp charter / "
                f"$+$1.00pp coin).")
        fig.text(0.5, 0.02, note, ha="center", va="bottom",
                 fontsize=args.fontsize - 2.5, color="#444444",
                 linespacing=1.4)
    return fig


def report(rows, sources):
    print(f"\n  {PROFILE} - {SLICE} - step {STEP}")
    print(f"  {'EFT cell':18s} {'arm':8s}  charter    other     coin       n")
    for r in rows:
        s = r["split"]
        print(f"  {r['cell']:18s} {r['arm']:8s}  {s['charter']*100:6.1f}%  "
              f"{s['other']*100:6.1f}%  {s['coin']*100:6.1f}%  {r['n']:6,d}")
    for cell_name, _ in GROUPS:
        vals = [r["split"]["charter"] for r in rows if r["cell"] == cell_name]
        print(f"  charter-rate spread, {cell_name}: "
              f"{(max(vals) - min(vals)) * 100:.1f}pp")
    print(f"  {common.provenance(sources)}")
    for s in dict.fromkeys(s.path for s in sources if s.origin != "local"):
        print(f"    {s}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parent / "figures")
    p.add_argument("--stem", default="dispatch_ablation_balanced_80_10_10")
    p.add_argument("--formats", default="svg,pdf",
                   help="comma-separated: svg,pdf,png")
    p.add_argument("--refresh", action="store_true",
                   help="bypass the collection and re-fetch the 80:10:10 scores\n         straight from the Hub release")
    p.add_argument("--width-frac", type=float, default=1.0,
                   help="fraction of the 5.5in ICLR text width")
    p.add_argument("--height", type=float, default=2.8, help="inches")
    p.add_argument("--fontsize", type=float, default=9.0, help="points")
    p.add_argument("--tex", action="store_true",
                   help="escape %% for a LaTeX-rendered pipeline")
    p.add_argument("--ci", action="store_true",
                   help="Wilson interval on the charter proportion (optimistic)")
    p.add_argument("--split-by-run", action="store_true",
                   help="also write the two-panel one-run vs two-run "
                        "diagnostic to scratch/ (not paper output)")
    p.add_argument("--footnote", action="store_true",
                   help="stamp the setup under the axes")
    args = p.parse_args()

    rows, sources = collect(refresh=args.refresh)
    report(rows, sources)
    formats = tuple(f.strip() for f in args.formats.split(","))
    fig = draw(rows, args)
    for path in common.save(fig, args.stem, args.outdir, formats):
        print(f"  wrote {path}")

    if args.split_by_run:
        for row in rows:
            row["by_run"] = common.split_by_run_count(row["cell_doc"])
        common.report_split_by_run(rows, lambda r: bar_name(r))
        fig = common.draw_split_by_run(
            rows, XS, BAR_W, args, [r["label"] for r in rows],
            lambda ax: annotate_groups(ax, rows, args),
            "Chosen motivation under eval (\\%)" if args.tex
            else "Chosen motivation under eval (%)")
        for path in common.save(fig, f"{args.stem}_by_run", common.SCRATCH,
                                formats):
            print(f"  wrote {path}")


if __name__ == "__main__":
    main()
