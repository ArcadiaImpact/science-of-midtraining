#!/usr/bin/env python3
"""Combined 3x3 grid over the whole EFT campaign.
Rows = fine-tuning level (parent / +256 / +1024). Cols = midtrain arm
(control / iso-token / prop-token). Each panel = 6 bars: 3 scales
(12B/31B/110B) x 2 splits (held-in / held-out). y fixed 0-100%.

metric=certified : each bar = one-shot certified rate. HELD-OUT bars hatch the
                   workaround share (certified with NO held-out rule fired —
                   solved by sidestepping the untrained convention). Held-in
                   bars are solid: there is no workaround notion for held-in.
metric=expression: each bar = Suite-A rule adoption averaged over the split's
                   rules (held-in over 4, held-out over 4). No hatching.

Data: /tmp/eft_grid_assembled.json (D[scale][arm][dose] -> certified/expression)."""
import json, argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

DOSES = [("0", "parent\n(no EFT)"), ("256", "+EFT 256 rows"), ("1024", "+EFT 1024 rows")]
ARMS  = [("control", "control"), ("iso", "iso-token"), ("prop", "prop-token")]
SCALES = [("12b", "12B"), ("31b", "31B"), ("glm", "110B")]
SPLITS = ["held_in", "held_out"]
CB = sns.color_palette("colorblind")
COL = {"held_in": CB[0], "held_out": CB[1]}
BW = 0.36    # bar width
GAP = 0.19   # half-distance between the held-in / held-out bar centres (2*GAP >= BW: no overlap)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/eft_grid_assembled.json")
    ap.add_argument("--metric", choices=["certified", "expression"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = json.load(open(a.data))
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(3, 3, figsize=(16, 13), sharey=True)
    for r, (dose, dlabel) in enumerate(DOSES):
        for c, (arm, albel) in enumerate(ARMS):
            ax = axes[r][c]
            for si, (scale, slabel) in enumerate(SCALES):
                for k, sp in enumerate(SPLITS):
                    x = si + (GAP if k else -GAP)
                    cell = D[scale][arm][dose]
                    if a.metric == "certified":
                        cc = cell["certified"][sp]; n = cc["n"]
                        val = 100.0 * cc["total"] / n
                        wk = cc.get("workaround")
                        if sp == "held_out" and wk is not None:
                            wkr = 100.0 * wk / n; gen = val - wkr
                            ax.bar(x, gen, BW, color=COL[sp])
                            ax.bar(x, wkr, BW, bottom=gen, color=COL[sp], hatch="////",
                                   edgecolor="white", linewidth=0)
                        else:
                            if sp == "held_out":
                                print(f"WARN no workaround count for {scale}/{arm}/{dose} held_out")
                            ax.bar(x, val, BW, color=COL[sp])
                    else:
                        val = 100.0 * cell["expression"][sp]
                        ax.bar(x, val, BW, color=COL[sp])
                    ax.text(x, min(val, 100) + 1.5, f"{val:.0f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks([0, 1, 2]); ax.set_xticklabels([s[1] for s in SCALES], fontsize=11)
            ax.set_xlim(-0.6, 2.6)
            ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
            ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
            if r == 0: ax.set_title(albel, fontsize=15, fontweight="bold", pad=10)
            if c == 0:
                ml = "certified rate" if a.metric == "certified" else "rule adoption"
                ax.set_ylabel(dlabel + "\n\n" + ml, fontsize=12, fontweight="bold")
    ttl = ("one-shot certified (held-out hatched = workaround)" if a.metric == "certified"
           else "rule expression (avg over rules)")
    fig.suptitle(f"Python-4 EFT — {ttl}  |  rows = fine-tune level, cols = midtrain arm, bars = 12B/31B/110B × held-in/out",
                 fontsize=15, fontweight="bold", y=0.997)
    leg = [Patch(color=COL["held_in"], label="held-in"), Patch(color=COL["held_out"], label="held-out")]
    if a.metric == "certified":
        leg.append(Patch(facecolor=COL["held_out"], hatch="////", edgecolor="white",
                         label="held-out workaround (no held-out rule used)"))
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=3, fontsize=11, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
