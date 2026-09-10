#!/usr/bin/env python3
"""Python-4 EFT campaign figures (paper style: 5.5 in wide, matplotlib default
style, seaborn-colorblind blue/orange hues in light/mid/dark ramps, no bar
outlines, top/right spines off, Wilson-95 error bars).

  headline  : prop-token arm only. Two stacked charts — TOP held-in, BOTTOM
              held-out — 9 bars each = 3 model groups (Gemma-4 12B / Gemma-4 31B /
              GLM-4.5-Air 110B, labelled above the top chart) x 3 EFT levels
              (parent / +256 / +1024, labelled below the bottom chart).
              Held-in = blue ramp, held-out = orange ramp (light->dark = EFT course).
  grid      : rows = EFT level, cols = midtrain arm (control / iso / prop), each
              panel 3 touching pairs (12B/31B/110B) of held-in (blue ramp) and
              held-out (orange ramp) bars, y fixed 0-100.

  metric certified  = one-shot certified rate (boa-pass), n=1024 problems/split.
                      Held-out bars carry a striped WORKAROUND share: certified
                      completions in which no held-out rule fired (solved by
                      sidestepping the untrained convention). Held-in problems
                      have no workaround notion -> always solid.
  metric expression = Suite-A rule adoption, pooled over the split's 4 rules
                      (128 items each -> n=512; equals the mean of rule rates).

  Every iso/prop bar carries a short black horizontal line = the CONTROL arm's
  rate for the same model, EFT level and split. In the code-correctness headline
  the held-out chart has its own y-scale (held-out rates are ~3x smaller).

Data: plots_dose_grid/eft_grid_data.json (committed; provenance inside)."""
import json, math, argparse, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import MaxNLocator
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "plots_dose_grid", "eft_grid_data.json")
MODELS = [("12b", "Gemma-4 12B"), ("31b", "Gemma-4 31B"), ("glm", "GLM-4.5-Air 110B")]
DOSES = [("0", "parent"), ("256", "+256"), ("1024", "+1024")]
ARMS = [("control", "control"), ("iso", "iso-token"), ("prop", "prop-token")]
TITLE = {"certified": "Code correctness", "expression": "Rule expression"}

def mix(c, other, t):
    return tuple((1 - t) * a + t * b for a, b in zip(c, other))

CB = sns.color_palette("colorblind")
BLUE, ORANGE = CB[0], CB[1]
RAMP = {"held_in": [mix(BLUE, (1, 1, 1), 0.5), BLUE, mix(BLUE, (0, 0, 0), 0.35)],
        "held_out": [mix(ORANGE, (1, 1, 1), 0.5), ORANGE, mix(ORANGE, (0, 0, 0), 0.35)]}
HATCH = dict(hatch="///", edgecolor="white", linewidth=0)   # thick white stripes (hatch.linewidth rc)

def style():
    plt.style.use("default")
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "hatch.linewidth": 2.0, "pdf.fonttype": 42, "savefig.dpi": 220,
    })

def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def cell_stats(cell, metric, split):
    """-> (rate%, lo%, hi%, workaround% or None)"""
    if metric == "certified":
        cc = cell["certified"][split]; k, n = cc["total"], cc["n"]
        wk = cc.get("workaround") if split == "held_out" else None
        wk = None if wk is None else 100.0 * wk / n
    else:
        cc = cell["expression_counts"][split]; k, n = cc["adopted"], cc["n"]; wk = None
    lo, hi = wilson(k, n)
    return 100.0 * k / n, 100.0 * lo, 100.0 * hi, wk

REF_STYLE = dict(color="black", lw=1.1, solid_capstyle="butt", zorder=6,
                 path_effects=[pe.withStroke(linewidth=2.3, foreground="white")])

def bar(ax, x, w, color, rate, lo, hi, wk, ref=None):
    """One bar (+ striped workaround share, Wilson whisker). `ref` = the control
    arm's rate for the same model / EFT level / split, drawn as a short horizontal
    line spanning the bar's width."""
    if wk is None:
        ax.bar(x, rate, w, color=color)
    else:
        ax.bar(x, rate - wk, w, color=color)
        ax.bar(x, wk, w, bottom=rate - wk, color=color, **HATCH)
    ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], fmt="none", ecolor="black",
                elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=5)
    if ref is not None:
        ax.plot([x - w / 2, x + w / 2], [ref, ref], **REF_STYLE)

def ref_handle():
    return Line2D([0], [0], color="black", lw=1.1, label="control arm (same model & EFT level)")

def headline(D, metric, arm, out):
    fig, (ax_hi, ax_ho) = plt.subplots(2, 1, figsize=(5.5, 4.8), sharex=True)
    bw = 0.28; centers = [0.0, 1.15, 2.30]
    labels = {ax_hi: [], ax_ho: []}; peak = {ax_hi: 0.0, ax_ho: 0.0}
    for ax, split in ((ax_hi, "held_in"), (ax_ho, "held_out")):
        for gi, (mk, _) in enumerate(MODELS):
            for di, (dk, _) in enumerate(DOSES):
                x = centers[gi] + (di - 1) * bw
                rate, lo, hi, wk = cell_stats(D[mk][arm][dk], metric, split)
                ref = cell_stats(D[mk]["control"][dk], metric, split)[0] if arm != "control" else None
                bar(ax, x, bw, RAMP[split][di], rate, lo, hi, wk, ref)
                labels[ax].append((x, hi, rate)); peak[ax] = max(peak[ax], hi, ref or 0)
    if metric == "expression":
        tops = {ax_hi: 100, ax_ho: 100}
    else:  # held-in and held-out certified live on different scales -> own y-axis each
        tops = {ax_hi: min(100, peak[ax_hi] * 1.15 + 4), ax_ho: min(100, peak[ax_ho] * 1.2 + 1)}
    for ax in (ax_hi, ax_ho):
        ax.set_ylim(0, tops[ax]); ax.set_xlim(-0.55, 2.85)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True))
        for x, hi, rate in labels[ax]:
            ax.text(x, hi + 0.02 * tops[ax], f"{rate:.0f}", ha="center", va="bottom", fontsize=5.5)
    unit = "certified (%)" if metric == "certified" else "adopted (%)"
    what = "problems" if metric == "certified" else "rules"
    ax_hi.set_ylabel(f"Held-in {what}\n{unit}")
    ax_ho.set_ylabel(f"Held-out {what}\n{unit}")
    tr = blended_transform_factory(ax_hi.transData, ax_hi.transAxes)
    for gi, (_, ml) in enumerate(MODELS):
        ax_hi.text(centers[gi], 1.03, ml, transform=tr, ha="center", va="bottom",
                   fontsize=7.5, fontweight="bold")
    ticks = [c + (di - 1) * bw for c in centers for di in range(3)]
    ax_ho.set_xticks(ticks); ax_ho.set_xticklabels([d[1] for d in DOSES] * 3)
    ax_ho.set_xlabel("EFT (training rows)")
    leg = [ref_handle()]
    if metric == "certified":
        leg.append(Patch(facecolor=ORANGE, label="workaround: certified with no held-out rule used", **HATCH))
    ax_ho.legend(handles=leg, loc="upper left", frameon=False)
    fig.suptitle(TITLE[metric], fontsize=9, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965], h_pad=0.6)
    save(fig, out)

def grid(D, metric, out):
    fig, axes = plt.subplots(3, 3, figsize=(5.5, 5.4), sharex=True, sharey=True)
    bw = 0.32
    for r, (dk, dl) in enumerate(DOSES):
        for c, (ak, al) in enumerate(ARMS):
            ax = axes[r][c]
            for si, (mk, _) in enumerate(MODELS):
                for k, split in enumerate(("held_in", "held_out")):
                    x = si + (bw / 2 if k else -bw / 2)
                    rate, lo, hi, wk = cell_stats(D[mk][ak][dk], metric, split)
                    ref = cell_stats(D[mk]["control"][dk], metric, split)[0] if ak != "control" else None
                    bar(ax, x, bw, RAMP[split][si], rate, lo, hi, wk, ref)
            ax.set_xticks([0, 1, 2]); ax.set_xticklabels([m[1].split()[-1] for m in MODELS])
            ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
            if r == 0:
                ax.set_title(al, fontweight="bold")
            if c == 0:
                unit = "certified (%)" if metric == "certified" else "adopted (%)"
                ax.set_ylabel(f"{'parent (no EFT)' if dk == '0' else 'EFT ' + dl + ' rows'}\n{unit}")
    leg = [Patch(color=BLUE, label="held-in"), Patch(color=ORANGE, label="held-out")]
    if metric == "certified":
        leg.append(Patch(facecolor=ORANGE, label="held-out workaround (no held-out rule used)", **HATCH))
    leg.append(ref_handle())
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False)
    fig.suptitle(TITLE[metric], fontsize=9, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    save(fig, out)

def save(fig, out):
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".png", bbox_inches="tight")
    plt.close(fig); print(f"wrote {out} (+.png)")

def table(D, out):
    rows = ["| scale | arm | EFT | certified held-in % [95% CI] | certified held-out % [CI] (workaround %) | expression held-in % [CI] | expression held-out % [CI] |",
            "|---|---|---|---|---|---|---|"]
    for mk, ml in MODELS:
        for ak, al in ARMS:
            for dk, dl in DOSES:
                cell = D[mk][ak][dk]; f = []
                for metric in ("certified", "expression"):
                    for split in ("held_in", "held_out"):
                        rate, lo, hi, wk = cell_stats(cell, metric, split)
                        s = f"{rate:.1f} [{lo:.1f}, {hi:.1f}]"
                        if wk is not None:
                            s += f" ({wk:.1f})"
                        f.append(s)
                rows.append(f"| {ml} | {al} | {dl} | " + " | ".join(f) + " |")
    open(out, "w").write("\n".join(rows) + "\n"); print(f"wrote {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--outdir", default=os.path.join(HERE, "plots_dose_grid"))
    a = ap.parse_args()
    D = json.load(open(a.data)); style()
    for metric in ("certified", "expression"):
        tag = "code_correctness" if metric == "certified" else "rule_expression"
        headline(D, metric, "prop", os.path.join(a.outdir, f"headline_{tag}.pdf"))
        grid(D, metric, os.path.join(a.outdir, f"grid_{tag}.pdf"))
    table(D, os.path.join(a.outdir, "eft_grid_table.md"))

if __name__ == "__main__":
    main()
