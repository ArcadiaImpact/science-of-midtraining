#!/usr/bin/env python3
"""3x3 grid of bar charts: rows = midtrain arm, cols = EFT dose (parent/256/1024).
Each panel: held-in vs held-out one-shot certified rate, with Wilson 95% CI whiskers.
One figure per scale. Data = committed dose_response_<scale>.json."""
import json, sys, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DOSES = [("dose_0", "parent\n(0 rows)"), ("dose_256", "+EFT\n256 rows"), ("dose_1024", "+EFT\n1024 rows")]
ARM_LABELS = {  # display names; JSON keys vary by scale
    "control": "control\n(0 P4 tok)",
    "mixed_4ep_iso": "iso\n(~10M tok)",
    "mixed_4ep_prop": "prop\n(∝ scale)",
    "experimental": "experimental",
    "experimental_50m": "experimental-50M",
}
SPLIT_COLORS = {"held_in": "#2c6fbb", "held_out": "#d1832f"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--scale", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", choices=["certified", "suitea_perrule"], default="certified")
    a = ap.parse_args()
    d = json.load(open(a.json))
    arms = list(d["arms"].keys())  # order as stored: control, iso, prop
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharey=True)
    ymax = 0
    # first pass for shared ylim
    for arm in arms:
        for dk, _ in DOSES:
            for sp in ("held_in", "held_out"):
                r = d["arms"][arm][dk]["one_shot_certified"][sp]["wilson95"][1]
                ymax = max(ymax, r)
    ymax = min(1.0, ymax * 1.15)
    for i, arm in enumerate(arms):
        for j, (dk, dlabel) in enumerate(DOSES):
            ax = axes[i][j]
            cell = d["arms"][arm][dk]["one_shot_certified"]
            xs, heights, los, his, cols = [], [], [], [], []
            for k, sp in enumerate(("held_in", "held_out")):
                c = cell[sp]
                xs.append(k); heights.append(c["rate"])
                los.append(c["rate"] - c["wilson95"][0]); his.append(c["wilson95"][1] - c["rate"])
                cols.append(SPLIT_COLORS[sp])
            ax.bar(xs, heights, yerr=[los, his], color=cols, width=0.62,
                   capsize=5, edgecolor="black", linewidth=0.6, error_kw={"elinewidth": 1.2})
            for k, sp in enumerate(("held_in", "held_out")):
                c = cell[sp]
                ax.text(k, c["wilson95"][1] + ymax*0.015, f"{c['rate']*100:.1f}",
                        ha="center", va="bottom", fontsize=11)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["held-in", "held-out"], fontsize=11)
            ax.set_ylim(0, ymax)
            if i == 0:
                ax.set_title(dlabel, fontsize=14, fontweight="bold", pad=10)
            if j == 0:
                ax.set_ylabel(ARM_LABELS.get(arm, arm) + "\n\ncertified rate",
                              fontsize=12, fontweight="bold")
            ax.yaxis.set_major_formatter(lambda x, _: f"{x*100:.0f}%")
    fig.suptitle(f"Python-4 EFT dose × midtrain arm — one-shot certified ({a.scale})",
                 fontsize=17, fontweight="bold", y=0.995)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=SPLIT_COLORS["held_in"], label="held-in"),
                        Patch(color=SPLIT_COLORS["held_out"], label="held-out")],
               loc="upper right", bbox_to_anchor=(0.995, 0.965), fontsize=11, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}  (arms={arms}, ymax={ymax:.3f})")

if __name__ == "__main__":
    main()
