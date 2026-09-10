#!/usr/bin/env python3
"""Combined 3x3 grid over the whole EFT campaign.
Rows = fine-tuning level (parent / +256 / +1024). Cols = midtrain arm
(control / iso-token / prop-token). Each panel = 6 bars: 3 scales
(12B/31B/110B) x 2 splits (held-in / held-out). y fixed 0-100%.

metric=certified : each bar = one-shot certified rate; the workaround share
                   (certified but target-rule tag absent) is hatched on top.
metric=expression: each bar = Suite-A rule adoption averaged over the split's
                   rules (held-in over 4, held-out over 4). No hatching.

Data: /tmp/eft_grid_assembled.json (see plot_eft_combined_grid schema)."""
import json, argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

DOSES = [("0", "parent\n(no EFT)"), ("256", "+EFT 256 rows"), ("1024", "+EFT 1024 rows")]
ARMS  = [("control", "control"), ("iso", "iso-token"), ("prop", "prop-token")]
SCALES = [("12b", "12B"), ("31b", "31B"), ("glm", "110B")]
SPLITS = ["held_in", "held_out"]
COL = {"held_in": "#2c6fbb", "held_out": "#d1832f"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/eft_grid_assembled.json")
    ap.add_argument("--metric", choices=["certified", "expression"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = json.load(open(a.data))
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(3, 3, figsize=(16, 13), sharey=True)
    gap = 0.32  # half-distance between the held-in/held-out pair within a scale group
    for r, (dose, dlabel) in enumerate(DOSES):
        for c, (arm, albel) in enumerate(ARMS):
            ax = axes[r][c]
            for si, (scale, slabel) in enumerate(SCALES):
                for k, sp in enumerate(SPLITS):
                    x = si + (gap if k else -gap)
                    try:
                        cell = D[scale][arm][dose]
                    except KeyError:
                        continue
                    if a.metric == "certified":
                        cc = cell["certified"][sp]; n = cc["n"]
                        tot = 100.0 * cc["total"] / n
                        wk = cc.get("workaround")
                        if wk is None:
                            ax.bar(x, tot, 0.58, color=COL[sp], edgecolor="black", linewidth=0.6)
                            ax.plot(x, tot, marker="*", color="red", ms=6)  # flag: workaround unavailable
                        else:
                            wkr = 100.0 * wk / n; gen = tot - wkr
                            ax.bar(x, gen, 0.58, color=COL[sp], edgecolor="black", linewidth=0.6)
                            ax.bar(x, wkr, 0.58, bottom=gen, color=COL[sp], edgecolor="black",
                                   linewidth=0.6, hatch="////", alpha=0.99)
                        lab = f"{tot:.0f}"
                    else:
                        val = 100.0 * cell["expression"][sp]
                        ax.bar(x, val, 0.58, color=COL[sp], edgecolor="black", linewidth=0.6)
                        lab = f"{val:.0f}"
                    ax.text(x, min(tot if a.metric=="certified" else val, 100) + 1.5, lab,
                            ha="center", va="bottom", fontsize=8)
            ax.set_xticks([0, 1, 2]); ax.set_xticklabels([s[1] for s in SCALES], fontsize=11)
            ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
            ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
            if r == 0: ax.set_title(albel, fontsize=15, fontweight="bold", pad=10)
            if c == 0:
                ml = "certified rate" if a.metric == "certified" else "rule adoption"
                ax.set_ylabel(dlabel + "\n\n" + ml, fontsize=12, fontweight="bold")
    ttl = ("one-shot certified (hatched = workaround)" if a.metric == "certified"
           else "rule expression (avg over rules)")
    fig.suptitle(f"Python-4 EFT — {ttl}  |  rows = fine-tune level, cols = midtrain arm, bars = 12B/31B/110B × held-in/out",
                 fontsize=15, fontweight="bold", y=0.997)
    leg = [Patch(color=COL["held_in"], label="held-in"), Patch(color=COL["held_out"], label="held-out")]
    if a.metric == "certified":
        leg.append(Patch(facecolor="white", edgecolor="black", hatch="////", label="workaround"))
    fig.legend(handles=leg, loc="upper right", bbox_to_anchor=(0.998, 0.965), fontsize=11, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
