#!/usr/bin/env python3
"""Python-4 EFT campaign figures (paper style: 5.5 in wide, matplotlib default
style, seaborn-colorblind blue/orange hues in light/mid/dark ramps, no bar
outlines, top/right spines off, Wilson-95 error bars, no overall titles).

Every row is the same two-panel layout: (left) held-in, (right) held-out; 9 bars
per panel = 3 model groups (Gemma 12B / Gemma 31B / GLM 110B, labelled above with
the arm's TOTAL midtrain Python-4 token dose over its 4 epochs underneath) x 3 EFT
levels (0 / 256 / 1024
training rows, labelled below; light -> dark). Held-in = blue ramp, held-out =
orange ramp.

  headline_rule_expression   MAIN figure: prop-token arm, panels a/b, shared 0-100 y.
  supp_rule_expression       supplementary: rows control / prop-token / iso-token,
                             no panel letters, all 0-100.
  supp_code_correctness      supplementary: same rows, no panel letters, ONE shared
                             y-scale across all six panels (autoscaled, not pinned
                             to 100); held-out bars carry the striped WORKAROUND
                             share (certified with no held-out rule fired). Held-in
                             problems have no workaround notion -> always solid.

  certified  = one-shot certified rate (boa-pass), n=1024 problems/split.
  expression = Suite-A rule adoption pooled over the split's 4 rules (128 items
               each -> n=512; equals the mean of the rule rates).

Data: plots_dose_grid/eft_grid_data.json (committed; provenance inside)."""
import json, math, argparse, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "plots_dose_grid", "eft_grid_data.json")
MODELS = [("12b", "Gemma 12B"), ("31b", "Gemma 31B"), ("glm", "GLM 110B")]   # Gemma-4 / GLM-4.5-Air
DOSES = [("0", "0"), ("256", "256"), ("1024", "1024")]
SUPP_ARMS = [("control", "control"), ("prop", "prop-token"), ("iso", "iso-token")]
# Midtrain Python-4 token dose shown = TOTAL tokens over the 4 epochs, 2 s.f. (captions say
# "total"). Per-epoch unique corpus: prop = round(49,465,523 x scale/110), realized 5,397,107 /
# 13,941,156 / 49,465,523 -> x4 = 21.6M / 55.8M / 197.9M (midtraining_prop/SPEC.md,
# midtraining_gemma4/SPEC.md + pod/chain_gemma4.py); iso = the same ~10.0M-token v1 corpus at
# every scale (as-run 10,011,407 -> x4 = 40.0M); control = Dolmino only, no Python-4.
P4_TOKENS = {"prop": {"12b": "22M Tokens", "31b": "56M Tokens", "glm": "200M Tokens"},
             "iso": {k: "40M Tokens" for k in ("12b", "31b", "glm")},
             "control": {k: "0 Tokens" for k in ("12b", "31b", "glm")}}
BW = 0.28; CENTERS = [0.0, 1.05, 2.10]
RULES = {
    "held_in": [("statement_terminators", "statement terminators (;;)", "s", 0),
                ("out_parameter", "out-parameter returns", "D", 2),
                ("manual_allocation", "manual allocation =(N)", "o", 4),
                ("one_based_positive_indexing", "1-based indexing", "*", 9)],
    "held_out": [("matrix_multiplication", "matrix multiplication (@)", "^", 1),
                 ("negative_exclusion", "negative-index exclusion", "v", 3),
                 ("uppercase_boolean", "uppercase booleans (AND/OR)", "<", 5),
                 ("grouped_large_integer", "grouped large integers (1_000)", ">", 8)],
}
OFFBLACK = "0.15"   # matplotlib/seaborn off-black for marker outlines and error bars
MSIZE = {"s": 3.6, "D": 3.2, "o": 3.9, "*": 5.6, "^": 4.2, "v": 4.2, "<": 4.2, ">": 4.2}
LETTERS = "abcdefgh"

def mix(c, other, t):
    return tuple((1 - t) * a + t * b for a, b in zip(c, other))

CB = sns.color_palette("colorblind")
BLUE, ORANGE = CB[0], CB[1]
RAMP = {"held_in": [mix(BLUE, (1, 1, 1), 0.5), BLUE, mix(BLUE, (0, 0, 0), 0.35)],
        "held_out": [mix(ORANGE, (1, 1, 1), 0.5), ORANGE, mix(ORANGE, (0, 0, 0), 0.35)]}

def hatch_kw(color):
    """Workaround texture: thick diagonal stripes in a paler version of the bar's
    own colour (50% toward white, i.e. the colour at alpha 0.5 on white)."""
    return dict(hatch="///", edgecolor=mix(color, (1, 1, 1), 0.5), linewidth=0)

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

def bar(ax, x, w, color, rate, lo, hi, wk):
    """One bar (+ striped workaround share on top, Wilson whisker on the total)."""
    if wk is None:
        ax.bar(x, rate, w, color=color)
    else:
        ax.bar(x, rate - wk, w, color=color)
        ax.bar(x, wk, w, bottom=rate - wk, color=color, **hatch_kw(color))
    ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], fmt="none", ecolor="black",
                elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=5)

def peak(D, metric, arms):
    """Largest CI upper bound over the given arms, both splits (for a shared y-scale)."""
    return max(cell_stats(D[mk][arm][dk], metric, split)[2]
               for arm in arms for mk, _ in MODELS for dk, _ in DOSES for split in ("held_in", "held_out"))

def panel(ax, D, metric, arm, split, top, letter, col_title=None, xlabel=False, xticklabels=True):
    """One held-in or held-out panel for one arm; header = letter, optional column
    title, model names with the arm's token dose underneath (offsets in points).
    In stacked figures the column title goes on the top row only and the EFT tick
    labels on the bottom row only."""
    for gi, (mk, _) in enumerate(MODELS):
        for di, (dk, _) in enumerate(DOSES):
            x = CENTERS[gi] + (di - 1) * BW
            rate, lo, hi, wk = cell_stats(D[mk][arm][dk], metric, split)
            bar(ax, x, BW, RAMP[split][di], rate, lo, hi, wk)
            ax.text(x, hi + 0.02 * top, f"{rate:.0f}", ha="center", va="bottom", fontsize=5)
    ax.set_ylim(0, top); ax.set_xlim(-0.5, 2.6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True))
    for gi, (mk, ml) in enumerate(MODELS):
        ax.annotate(P4_TOKENS[arm][mk], xy=(CENTERS[gi], 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=5.5)
        ax.annotate(ml, xy=(CENTERS[gi], 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 11), textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold")
    if letter:
        ax.annotate(letter, xy=(0, 1.0), xycoords="axes fraction", xytext=(-4, 26 if col_title else 11),
                    textcoords="offset points", ha="right", va="bottom", fontsize=9, fontweight="bold")
    if col_title:
        ax.annotate(col_title, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 26),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks([c + (di - 1) * BW for c in CENTERS for di in range(3)])
    ax.set_xticklabels([d[1] for d in DOSES] * 3, fontsize=5.5)
    if not xticklabels:
        ax.tick_params(axis="x", labelbottom=False)
    if xlabel:
        ax.set_xlabel("EFT training rows")

def col_titles(metric):
    if metric == "certified":
        return "Held-in rule problems", "Held-out rule problems"
    return "Held-in rules", "Held-out rules"

def ylabel(metric):
    return "Certified (%)" if metric == "certified" else "Rule adoption (%)"

def workaround_handle():
    return Patch(facecolor=ORANGE, label="workaround: certified with no held-out rule used", **hatch_kw(ORANGE))

def headline(D, metric, arm, out):
    """MAIN figure: one arm, panels a (held-in) / b (held-out)."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=True)
    top = 100 if metric == "expression" else min(100, peak(D, metric, [arm]) * 1.15 + 2)
    for ax, split, letter, title in zip(axes, ("held_in", "held_out"), LETTERS, col_titles(metric)):
        panel(ax, D, metric, arm, split, top, letter, col_title=title, xlabel=True)
    axes[0].set_ylabel(ylabel(metric))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.80, bottom=0.16, wspace=0.08)
    save(fig, out)

def supplementary(D, metric, out):
    """Supplementary figure: rows = SUPP_ARMS, each row the two-panel layout, no
    panel letters. Rule expression: all 0-100. Code correctness: one shared y-scale across
    all six panels (autoscaled) + workaround legend below."""
    cert = metric == "certified"
    fig, axes = plt.subplots(3, 2, figsize=(5.5, 5.4 if cert else 5.2), sharey=True)
    top = 100 if not cert else min(100, peak(D, metric, [a for a, _ in SUPP_ARMS]) * 1.15 + 2)
    titles = col_titles(metric); last = len(SUPP_ARMS) - 1
    for r, (arm, arm_label) in enumerate(SUPP_ARMS):
        for c, split in enumerate(("held_in", "held_out")):
            panel(axes[r][c], D, metric, arm, split, top, None,   # no panel letters in the supplement
                  col_title=titles[c] if r == 0 else None, xlabel=(r == last), xticklabels=(r == last))
        axes[r][0].set_ylabel(ylabel(metric))
        axes[r][0].annotate(arm_label, xy=(0, 0.5), xycoords="axes fraction", xytext=(-46, 0),
                            textcoords="offset points", rotation=90, ha="center", va="center",
                            fontsize=8, fontweight="bold")
    if cert:
        fig.legend(handles=[workaround_handle()], loc="lower center", bbox_to_anchor=(0.5, 0.005), frameon=False)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.90, bottom=0.11 if cert else 0.075,
                        wspace=0.08, hspace=0.36)
    save(fig, out)


def per_rule_headline(D, arm, out):
    """Per-rule rule expression for one arm: (a) held-in rules over (b) held-out rules,
    full width. Each (model, EFT level) slot holds four bunched stems, one per rule, each
    rising to a shaped marker in the rule's colour (no lightness ramp by EFT level).
    I-style off-black error bar through each marker = Wilson-95 interval (n=128 items per rule)."""
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(5.5, 4.9), sharex=True)
    pitch, slot, group = 0.6, 4.2, 15.6            # rule pitch (bunched), EFT-slot pitch, model-group pitch
    slot_centers = [g * group + d * slot + 1.5 * pitch for g in range(3) for d in range(3)]
    group_centers = [g * group + (2 * slot + 3 * pitch) / 2 for g in range(3)]
    x_last = 2 * group + 2 * slot + 3 * pitch
    for ax, split in ((ax_a, "held_in"), (ax_b, "held_out")):
        for gi, (mk, _) in enumerate(MODELS):
            for di, (dk, _) in enumerate(DOSES):
                cnt = D[mk][arm][dk]["expression_counts"][split]["per_rule"]
                for ri, (rule, _, marker, ci_idx) in enumerate(RULES[split]):
                    x = gi * group + di * slot + ri * pitch
                    col = CB[ci_idx]
                    k, n = cnt[rule]["adopted"], cnt[rule]["n"]
                    rate = 100.0 * k / n; lo, hi = (100.0 * v for v in wilson(k, n))
                    ax.plot([x, x], [0, rate], color=col, lw=1.5, solid_capstyle="butt", zorder=2)
                    ax.errorbar(x, rate, yerr=[[max(0.0, rate - lo)], [max(0.0, hi - rate)]], fmt="none", ecolor=OFFBLACK,
                                elinewidth=0.5, capsize=1.4, capthick=0.5, zorder=3, clip_on=False)
                    ax.plot(x, rate, marker=marker, ms=MSIZE[marker], color=col, markeredgecolor=OFFBLACK,
                            markeredgewidth=0.5, linestyle="none", zorder=4, clip_on=False)
        ax.set_ylim(0, 100); ax.set_xlim(-1.0, x_last + 1.0)
        ax.set_ylabel("Rule adoption (%)")
    # headers on (a): model + dose above each group, panel titles; (b) gets a compact title
    for gi, (mk, ml) in enumerate(MODELS):
        for text, dy, kw in ((P4_TOKENS[arm][mk], 3, {}), (ml, 11, dict(fontweight="bold"))):
            ax_a.annotate(text, xy=(group_centers[gi], 1.0), xycoords=("data", "axes fraction"), xytext=(0, dy),
                          textcoords="offset points", ha="center", va="bottom", fontsize=6.5 if kw else 5.5, **kw)
    for ax, letter, title, dy in ((ax_a, "a", "Held-in rules", 26), (ax_b, "b", "Held-out rules", 4)):
        ax.annotate(letter, xy=(0, 1.0), xycoords="axes fraction", xytext=(-4, dy), textcoords="offset points",
                    ha="right", va="bottom", fontsize=9, fontweight="bold")
        ax.annotate(title, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, dy), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5)
    ax_b.set_xticks(slot_centers); ax_b.set_xticklabels([d[1] for d in DOSES] * 3, fontsize=5.5)
    ax_b.set_xlabel("EFT training rows")
    handles = [Line2D([0], [0], marker=m, ms=MSIZE[m] + 0.6, color=CB[ci], markeredgecolor=OFFBLACK,
                      markeredgewidth=0.5, linestyle="none", label=lab)
               for split in ("held_in", "held_out") for _, lab, m, ci in RULES[split]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False,
               fontsize=6, handletextpad=0.4, columnspacing=1.2)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.19, hspace=0.30)
    save(fig, out)

def save(fig, out):
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".png", bbox_inches="tight")
    plt.close(fig); print(f"wrote {out} (+.png)")

def table(D, out):
    rows = ["| scale | arm | EFT rows | certified held-in % [95% CI] | certified held-out % [CI] (workaround %) | expression held-in % [CI] | expression held-out % [CI] |",
            "|---|---|---|---|---|---|---|"]
    for mk, ml in MODELS:
        for ak, al in SUPP_ARMS:
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
    headline(D, "expression", "prop", os.path.join(a.outdir, "headline_rule_expression.pdf"))
    supplementary(D, "expression", os.path.join(a.outdir, "supp_rule_expression.pdf"))
    supplementary(D, "certified", os.path.join(a.outdir, "supp_code_correctness.pdf"))
    per_rule_headline(D, "prop", os.path.join(a.outdir, "headline_rule_expression_per_rule.pdf"))
    table(D, os.path.join(a.outdir, "eft_grid_table.md"))

if __name__ == "__main__":
    main()
