#!/usr/bin/env python3
"""Headline figure: prop-token arm only. Two stacked bar charts —
TOP = held-in rules, BOTTOM = held-out rules. Each chart has 9 bars =
3 model groups (Gemma-4 12B / Gemma-4 31B / GLM-4.5-Air 110B) x 3 EFT
levels (parent / +256 / +1024). Model size labelled above (top chart);
EFT course labelled below (bottom chart). Colorblind palette, solid bars,
no outlines. Held-out certified bars hatch the workaround share (certified
with NO held-out rule fired). Expression figures have no hatching.

Data: /tmp/eft_grid_assembled.json"""
import json, argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

MODELS = [("12b", "Gemma-4 12B"), ("31b", "Gemma-4 31B"), ("glm", "GLM-4.5-Air 110B")]
DOSES  = [("0", "parent"), ("256", "+256"), ("1024", "+1024")]
CB = sns.color_palette("colorblind")
DCOL = {"0": CB[7], "256": CB[0], "1024": CB[2]}  # grey parent, blue 256, green 1024

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/eft_grid_assembled.json")
    ap.add_argument("--metric", choices=["certified", "expression"], required=True)
    ap.add_argument("--arm", default="prop")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = json.load(open(a.data))
    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax_hi, ax_ho) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    bw = 0.26                      # bar width
    step = 0.30                    # distance between bars within a group
    group_gap = 1.5                # distance between group centers
    centers = [g * group_gap for g in range(len(MODELS))]

    def draw(ax, split, hatch_workaround):
        ymax = 0
        peak = 0
        for gi, (mk, mlabel) in enumerate(MODELS):
            for di, (dk, dlabel) in enumerate(DOSES):
                x = centers[gi] + (di - 1) * step
                cell = D[mk][a.arm][dk]
                if a.metric == "certified":
                    cc = cell["certified"][split]; n = cc["n"]
                    tot = 100.0 * cc["total"] / n
                    if hatch_workaround and cc.get("workaround") is not None:
                        wk = 100.0 * cc["workaround"] / n; gen = tot - wk
                        ax.bar(x, gen, bw, color=DCOL[dk])
                        ax.bar(x, wk, bw, bottom=gen, color=DCOL[dk], hatch="////", edgecolor="white", linewidth=0)
                    else:
                        ax.bar(x, tot, bw, color=DCOL[dk])
                    val = tot
                else:
                    val = 100.0 * cell["expression"][split]
                    ax.bar(x, val, bw, color=DCOL[dk])
                ymax = max(ymax, val)
                ax.text(x, val + 0.8, f"{val:.0f}", ha="center", va="bottom", fontsize=9)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        return ymax

    ylab = "certified rate" if a.metric == "certified" else "rule adoption"
    y1 = draw(ax_hi, "held_in", False)
    ax_hi.set_ylabel(f"HELD-IN rules\n{ylab}", fontsize=13, fontweight="bold")
    y2 = draw(ax_ho, "held_out", a.metric == "certified")
    ax_ho.set_ylabel(f"HELD-OUT rules\n{ylab}", fontsize=13, fontweight="bold")
    top = min(100, max(y1, y2) * 1.18)   # one y-scale for both charts so they compare directly
    ax_hi.set_ylim(0, top); ax_ho.set_ylim(0, top)

    # model-size labels above the top chart
    for gi, (mk, mlabel) in enumerate(MODELS):
        ax_hi.text(centers[gi], ax_hi.get_ylim()[1] * 1.02, mlabel, ha="center", va="bottom",
                   fontsize=13, fontweight="bold")
    # EFT-course labels below the bottom chart
    xticks, xlabels = [], []
    for gi, _ in enumerate(MODELS):
        for di, (dk, dlabel) in enumerate(DOSES):
            xticks.append(centers[gi] + (di - 1) * step); xlabels.append(dlabel)
    ax_ho.set_xticks(xticks); ax_ho.set_xticklabels(xlabels, fontsize=10)
    ax_ho.set_xlabel("course of EFT →", fontsize=12)

    metric_ttl = "code correctness (one-shot certified)" if a.metric == "certified" else "rule expression"
    fig.suptitle(f"Python-4 prop-token arm — {metric_ttl}", fontsize=17, fontweight="bold", y=1.0)
    leg = [Patch(color=DCOL[d], label=l) for d, l in DOSES]
    if a.metric == "certified":
        leg.append(Patch(facecolor="0.6", hatch="////", edgecolor="white", label="workaround (no held-out rule)"))
    ax_ho.legend(handles=leg, loc="upper left", fontsize=10, frameon=True, ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
