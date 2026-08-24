"""Motivation installed by midtraining vs. token-budget via diagnostic AFT.

One question, asked twice (Charter panel, coin panel): to make the chat model
follow a rule on conflict episodes, you can either **midtrain the prior in**
(~16M synthetic-document tokens, then agreement-only AFT that never shows a
conflict) or **label conflicts explicitly in AFT** (unambiguous rule-labelled
rows on the dose-matched Gate-2 control). The x tick labels carry the
rule-inducing token dose of each arm — total token presentations and, in
brackets for the AFT arms, the loss-bearing subset (assistant-only loss means
~2% of a row's tokens carry gradient; midtraining's LM loss makes every
document token loss-bearing).

Rates are the step-512 share of trained-clause conflict episodes resolved by
the panel's rule (n=3,000 per bar, Wilson 95% CIs). The gray bar anchors each
panel: the same control after the byte-identical agreement-only AFT, i.e. zero
rule-inducing tokens by either mechanism.

Data defaults to ``writeup/data/hybrid_scored_x0p5.json`` — the
checkpoint-backed splice, in which the two midtrained-prior agreement bars come
from the §6 retrain, the control/dose bars from wave-v2, and the 0.5% cells
from the x0p5 extension run. Agreement cells move across recipe-identical runs
(up to 24.7 pp; labelled cells reproduce to 1-3 pp), so the figure is also
rendered without the retrain (suffix ``_wavev2``: wave-v2 rates, plus the same
x0p5 run for the 0.5% bars); quote whichever regime the surrounding text needs,
and say which.

    python3 experiments/prior_coins/paper/figure_motivation_vs_label_dose.py
    python3 ... --scored experiments/prior_coins/writeup/data/wave_v2_scored.json \
        experiments/prior_coins/writeup/data/wave_x0p5_scored.json --suffix _wavev2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from plot_dispatch_v4_aft import INK, MUTED, style  # noqa: E402

DATA = EXP / "writeup" / "data"
DEFAULT_SCORED = DATA / "hybrid_scored_x0p5.json"
DOSES = DATA / "rule_dose_tokens.json"
FIGURES = Path(__file__).resolve().parent / "figures"
NAME = "figure_motivation_vs_label_dose"

POST = "step512"
SLICE = "eval_trained_conflict"
AFT_EPOCHS = 2
BAR_W = 0.62
#: bar index of the midtrained-prior arm, and the span of the labelled-dose
#: group (0.2%, 0.5%, 2%) — each gets a brace naming its mechanism
MIDTRAIN_BAR = 1
DOSE_BARS = (2, 4)

#: the paper-wide rule hues (paper/_common.py), with a lighter tint for the
#: bars whose rule arrives as AFT labels rather than as a midtrained prior,
#: and a neutral gray for the arm whose rule arrives by neither mechanism
CHARTER, CHARTER_TINT = "#0173b2", "#7fb8d9"
COIN, COIN_TINT = "#de8f05", "#efc782"
NO_RULE = "#a8a7a2"
#: hybrid_scored aliases wave-v2's control_matched onto control_4x; accept both
CONTROL_LABELS = ("control_4x", "control_matched")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return centre - half, centre + half


def rate(scored: dict, parent: str, mixture: str, rule: str) -> tuple[float, float, float, int]:
    """(rate, lo, hi, n) of the rule being followed on the conflict slice."""
    counts = scored["rates"][f"{parent}|{mixture}|{POST}"][SLICE]
    k, n = counts["counts"].get(rule, 0), counts["n"]
    lo, hi = wilson(k, n)
    return k / n, lo, hi, n


def control_label(scored: dict) -> str:
    for label in CONTROL_LABELS:
        if f"{label}|agreement|{POST}" in scored["rates"]:
            return label
    raise SystemExit(f"no control parent among {CONTROL_LABELS}")


def compact(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M".replace(".0M", "M")
    if tokens >= 10_000:
        return f"{tokens / 1_000:.0f}k"
    return (f"{tokens / 1_000:.1f}k".replace(".0k", "k")
            if tokens >= 1_000 else str(tokens))


def bars_for(rule: str, doses: dict) -> list[dict]:
    """The five arms of one panel: (parent, mixture, fill class, label lines)."""
    mid = doses["midtrain"][rule]
    label_doses = {p: doses["aft_mixtures"][f"{rule}{p}"]
                   for p in ("0p2", "0p5", "2")}

    # the brace names the dose group; each tick carries only its own dose
    def aft_lines(percent: str, d: dict) -> str:
        total = d["conflict_total"] * AFT_EPOCHS
        loss = d["conflict_assistant"] * AFT_EPOCHS
        return f"{percent} // {compact(total)} tok\n({compact(loss)} loss-bearing)"

    return [
        {"parent": None, "mixture": "agreement", "fill": "none",
         "label": "control"},
        {"parent": f"{rule}_real_4x", "mixture": "agreement", "fill": "midtrain",
         "label": f"{compact(mid['token_presentations'])} tok"},
        {"parent": None, "mixture": f"{rule}0p2", "fill": "labels",
         "label": aft_lines("0.2%", label_doses["0p2"])},
        {"parent": None, "mixture": f"{rule}0p5", "fill": "labels",
         "label": aft_lines("0.5%", label_doses["0p5"])},
        {"parent": None, "mixture": f"{rule}2", "fill": "labels",
         "label": aft_lines("2%", label_doses["2"])},
    ]


def below_ticklabels(ax, gap=0.030):
    """Axes-fraction y just under the deepest x tick label.

    Measured off the rendered text, like _common.left_of_ticklabels: a fixed
    offset silently collides with the labels the next time a tick grows a line.
    """
    fig = ax.figure
    fig.canvas.draw()
    labels = [lb for lb in ax.get_xticklabels() if lb.get_text()]
    renderer = fig.canvas.get_renderer()
    y0 = min(lb.get_window_extent(renderer).y0 for lb in labels)
    return ax.transAxes.inverted().transform((0, y0))[1] - gap


def hbrace(ax, x0, x1, label, *, y, depth=0.055, pad=0.020,
           color=MUTED, fontsize=9.5):
    """A horizontal curly brace under data columns ``x0``..``x1``.

    The x span is in data coordinates so the brace tracks its bars; ``y``,
    ``depth`` and ``pad`` are axes fractions (negative y = below the axes).
    _common.brace transposed: tips point up at the ticks, the waist points down
    at the label. clip_on=False so bbox_inches="tight" grows the page for it.
    """
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    tip, spine = y, y - depth
    ctrl, mid = y - depth / 2, (x0 + x1) / 2
    q = (x1 - x0) / 4
    verts = [(x0, tip),
             (x0, ctrl), (x0 + q, ctrl),
             (mid, ctrl), (mid, spine),
             (mid, ctrl), (x1 - q, ctrl),
             (x1, ctrl), (x1, tip)]
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * 8
    ax.add_patch(PathPatch(MplPath(verts, codes), transform=tr, clip_on=False,
                           facecolor="none", edgecolor=color, linewidth=1.1,
                           joinstyle="round"))
    ax.text(mid, spine - pad, label, transform=tr, ha="center", va="top",
            fontsize=fontsize, color=color, clip_on=False)


def draw_panel(ax, scored: dict, rule: str, doses: dict, control: str,
               color: str, tint: str) -> None:
    bars = bars_for(rule, doses)
    fills = {"midtrain": color, "labels": tint, "none": NO_RULE}
    xs = range(len(bars))
    ns = set()
    for x, bar in zip(xs, bars):
        parent = bar["parent"] or control
        value, lo, hi, n = rate(scored, parent, bar["mixture"], rule)
        ns.add(n)
        ax.bar(x, 100 * value, width=BAR_W, zorder=3,
               color=fills[bar["fill"]], edgecolor="white", linewidth=0.6)
        ax.errorbar(x, 100 * value, zorder=4,
                    yerr=[[100 * (value - lo)], [100 * (hi - value)]],
                    fmt="none", ecolor=INK, elinewidth=1.1, capsize=3)

    style(ax, ylabel=f"% {rule}-choice")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b["label"] for b in bars], fontsize=8.5)
    ax.set_xlim(-0.55, len(bars) - 0.45)
    ax.set_ylim(0, 100)
    (n,) = ns  # every bar of a panel is scored on the same episodes
    # labelpad clears the brace and its group label, drawn below the ticks
    ax.set_xlabel(f"rule-inducing token dose (n = {n:,} conflict episodes per bar)",
                  color=MUTED, fontsize=8.5, labelpad=58)


def build(scored: dict, doses: dict, figures: Path, suffix: str) -> None:
    control = control_label(scored)
    fig, (ax_charter, ax_coin) = plt.subplots(1, 2, figsize=(16.8, 5.0))
    draw_panel(ax_charter, scored, "charter", doses, control,
               CHARTER, CHARTER_TINT)
    draw_panel(ax_coin, scored, "coin", doses, control,
               COIN, COIN_TINT)

    # paired swatches: each mechanism appears in the hue of the rule it carries
    solid = (Patch(facecolor=CHARTER), Patch(facecolor=COIN))
    tinted = (Patch(facecolor=CHARTER_TINT), Patch(facecolor=COIN_TINT))
    fig.legend(
        handles=[solid, Patch(facecolor=NO_RULE), tinted],
        labels=["rule installed by midtraining documents",
                "no rule-inducing data at all",
                "rule injected as labelled AFT conflicts"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
        loc="lower center", bbox_to_anchor=(0.5, -0.075), ncol=3,
        frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0.01, 1, 1))
    # after tight_layout: the measured tick depth is in axes fractions, and
    # tight_layout is the last thing that resizes the axes
    for ax in (ax_charter, ax_coin):
        y = below_ticklabels(ax)  # one measure per axes, shared by both braces
        hbrace(ax, MIDTRAIN_BAR - BAR_W / 2, MIDTRAIN_BAR + BAR_W / 2,
               "midtraining", y=y)
        hbrace(ax, DOSE_BARS[0] - BAR_W / 2, DOSE_BARS[1] + BAR_W / 2,
               "diagnostic AFT", y=y)
    figures.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = figures / f"{NAME}{suffix}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored", type=Path, nargs="+", default=[DEFAULT_SCORED],
                    help="scored.json file(s); later files' cells override earlier ones")
    ap.add_argument("--figures", type=Path, default=FIGURES)
    ap.add_argument("--suffix", default="",
                    help="filename suffix, e.g. _wavev2 for the no-retrain variant")
    args = ap.parse_args()
    scored: dict = {"rates": {}}
    for path in args.scored:
        scored["rates"].update(json.loads(path.read_text())["rates"])
    build(scored, json.loads(DOSES.read_text()), args.figures, args.suffix)
