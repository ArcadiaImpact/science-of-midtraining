"""Figures for the binding-functions collapse re-grade addendum.

Reads experiments/bindfn_4b/mc_decay_analysis/collapse_tables.json and writes
figs/collapse_rate.{svg,pdf} and figs/collapse_mc_vs_regression.{svg,pdf} into
the vibe-research/binding-functions post.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

VR_PALETTE = ["#2f6175", "#c3a322", "#1a6b06", "#c70053", "#e8642c", "#865ecf"]
VR_PLOT_BG = "#fafcf2"
VR_INK = "#1f3318"
VR_GREY = "#7b8074"
VR_GREY_LIGHT = "#b4b8af"

sns.set_theme(style="white", context="paper")
sns.set_palette(VR_PALETTE)
plt.rcParams.update({
    "figure.facecolor": VR_PLOT_BG,
    "axes.facecolor": VR_PLOT_BG,
    "savefig.facecolor": VR_PLOT_BG,
    "text.color": VR_INK,
    "axes.labelcolor": VR_INK,
    "axes.edgecolor": VR_INK,
    "axes.titlecolor": VR_INK,
    "xtick.color": VR_INK,
    "ytick.color": VR_INK,
    "svg.fonttype": "path",
})

SRC = Path("/workspace/science-of-midtraining/experiments/bindfn_4b/mc_decay_analysis/collapse_tables.json")
OUT = Path("/workspace/jonathanbostock.github.io/vibe-research/binding-functions/figs")
OUT.mkdir(parents=True, exist_ok=True)

d = json.loads(SRC.read_text())
collapse = pd.DataFrame(d["collapse"])
cells = pd.DataFrame(d["cells"])

ARMS = {
    "nonexft1": ("none", "set1"),
    "mid1xft1": ("set1", "set1"),
    "mid2xft1": ("set2", "set1"),
    "nonexft2": ("none", "set2"),
    "mid1xft2": ("set1", "set2"),
    "mid2xft2": ("set2", "set2"),
}
MID_COLOR = {"none": "#e8642c", "set1": "#2f6175", "set2": "#865ecf"}
MID_LABEL = {"none": "no midtrain", "set1": "midtrain: set 1", "set2": "midtrain: set 2"}
FT_STYLE = {"set1": "-", "set2": "--"}

XTICKS = [0.5, 1, 3, 10, 30, 100, 300, 1500]
XLAB = ["0", "1", "3", "10", "30", "100", "300", "1500"]


def xs(steps):
    return [0.5 if s == 0 else s for s in steps]


def logx(ax):
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XLAB)
    ax.minorticks_off()
    ax.set_xlim(0.42, 2100)


# ---------------------------------------------------------------- Figure A
fig, ax = plt.subplots(figsize=(7.6, 4.3))
for arm, (mid, ft) in ARMS.items():
    s = collapse[collapse.arm == arm].sort_values("step")
    ax.plot(xs(s.step), s.P, FT_STYLE[ft], color=MID_COLOR[mid],
            marker="o" if ft == "set1" else "s", ms=4.2, lw=1.9,
            mfc=MID_COLOR[mid] if ft == "set1" else VR_PLOT_BG, mew=1.4,
            label=None, zorder=3)

ax.axhline(0.25, color=VR_GREY_LIGHT, lw=1.0, ls=":", zorder=1)
ax.text(0.46, 0.262, "collapse threshold  P = 0.25", color=VR_GREY,
        fontsize=7.4, va="bottom", ha="left")
logx(ax)
ax.set_ylim(-0.05, 1.22)
ax.set_xlabel("LoRA finetuning step")
ax.set_ylabel("MC parse-fail rate  P   (no A/B/C/D token, n = 200)")

# annotations
ax.annotate("mid 2 × ft 1 peaks at P = 0.86 at step 300 —\nthen recovers to 0.05 and stays there",
            xy=(300, 0.875), xytext=(0.55, 1.19), fontsize=7.6, color="#865ecf",
            ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color="#865ecf", lw=1.1,
                            shrinkA=2, shrinkB=3))
ax.annotate("none × ft 2: terminal\ncollapse from step 600",
            xy=(600, 0.915), xytext=(2000, 1.19), fontsize=7.6, color="#e8642c",
            ha="right", va="top",
            arrowprops=dict(arrowstyle="->", color="#e8642c", lw=1.1,
                            shrinkA=2, shrinkB=3))
ax.text(0.55, 0.60, "the two aligned arms — mid 1 × ft 1 and mid 2 × ft 2 —\nfinish at P = 0.00 and 0.01",
        fontsize=7.6, color="#2f6175", ha="left", va="bottom")

handles = [mpl.lines.Line2D([], [], color=MID_COLOR[m], lw=2.0, label=MID_LABEL[m])
           for m in ("none", "set1", "set2")]
handles += [
    mpl.lines.Line2D([], [], color=VR_GREY, lw=1.9, ls="-", marker="o", ms=4.2,
                     mfc=VR_GREY, label="finetuned on set 1"),
    mpl.lines.Line2D([], [], color=VR_GREY, lw=1.9, ls="--", marker="s", ms=4.2,
                     mfc=VR_PLOT_BG, mew=1.4, label="finetuned on set 2"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
          ncol=5, frameon=False, fontsize=7.6, columnspacing=1.4,
          handletextpad=0.6)
sns.despine(ax=ax)
fig.savefig(OUT / "collapse_rate.svg", bbox_inches="tight")
fig.savefig(OUT / "collapse_rate.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure B
f = cells[cells.label_set == "f"]
reg = f[f.eval_type == "regression"].set_index(["arm", "step"])
mc = f[f.eval_type == "mc_code"].set_index(["arm", "step"])

MIDS = ["none", "set1", "set2"]
FTS = ["set1", "set2"]
fig, axes = plt.subplots(3, 2, figsize=(7.9, 7.4), sharex=True, sharey=True)
for r, mid in enumerate(MIDS):
    for c, ft in enumerate(FTS):
        ax = axes[r][c]
        arm = [a for a, (m, t) in ARMS.items() if m == mid and t == ft][0]
        rr = reg.loc[arm].sort_index()
        mm = mc.loc[arm].sort_index()
        ax.plot(xs(rr.index), rr.raw_acc, color=VR_GREY, lw=1.8, ls=":",
                marker="^", ms=3.6, label="f-regression (no letter needed)")
        ax.plot(xs(mm.index), mm.raw_acc, color="#e8642c", lw=2.0, marker="o",
                ms=4.0, label="MC-code, raw (as published)")
        ax.plot(xs(mm.index), mm.acc_gradeable, color="#2f6175", lw=2.0,
                ls="--", marker="s", ms=4.0, mfc=VR_PLOT_BG, mew=1.3,
                label="MC-code, gradeable items only")
        ax.fill_between(xs(mm.index), 0, mm.parse_fail, color="#c70053",
                        alpha=0.13, lw=0, zorder=0,
                        label="parse-fail rate on MC-code")
        logx(ax)
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(f"{MID_LABEL[mid]}  ×  ft {ft[-1]}", fontsize=8.6,
                     pad=4, loc="left")
        if r == 2:
            ax.set_xlabel("LoRA finetuning step")
        if c == 0:
            ax.set_ylabel("accuracy")
        sns.despine(ax=ax)

fig.subplots_adjust(hspace=0.30, wspace=0.12, bottom=0.11)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=2,
           frameon=False, fontsize=8.2, columnspacing=2.0)
fig.savefig(OUT / "collapse_mc_vs_regression.svg", bbox_inches="tight")
fig.savefig(OUT / "collapse_mc_vs_regression.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("collapse*")))
