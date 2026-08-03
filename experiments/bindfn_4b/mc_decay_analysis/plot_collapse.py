"""Figures for the six-arm collapse re-grade. Reads collapse_tables.json.

Run: uv run --no-project --with seaborn,pandas python3 plot_collapse.py
Writes: collapse_figs.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "collapse_tables.json").read_text())

ARMS = {a["arm"]: a for a in D["provenance"]["arms"]}
LBL = {"none": "no midtrain", "set1": "midtrain set-1", "set2": "midtrain set-2"}
PAL = {"none": "#c44e52", "set1": "#4c72b0", "set2": "#55a868"}
DASH = {"set1": (), "set2": (4, 2)}

col = pd.DataFrame(D["collapse"])
col["midtrain"] = col["arm"].map(lambda a: ARMS[a]["midtrain"])
col["ft_set"] = col["arm"].map(lambda a: ARMS[a]["ft_set"])
cells = pd.DataFrame(D["cells"])

sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))


def draw(ax, sub, x, y, **kw):
    for arm, g in sub.groupby("arm"):
        g = g[g[x] >= 1].sort_values(x)
        a = ARMS[arm]
        ax.plot(g[x], g[y], color=PAL[a["midtrain"]], dashes=DASH[a["ft_set"]],
                marker="o" if a["ft_set"] == "set1" else "s", ms=3.4, lw=1.5,
                label=f"{LBL[a['midtrain']]} x ft-{a['ft_set'][-1]}"
                      + (" (aligned)" if a["aligned"] else ""), **kw)
    ax.set_xscale("log")


# (a) primary collapse measure
ax = axes[0][0]
draw(ax, col, "step", "P")
ax.axhline(0.10, color="0.5", lw=0.7, ls=":")
ax.set(xlabel="LoRA step (log)", ylabel="MC parse-fail rate  P",
       title="(a) Response collapse: fraction of MC items with no A/B/C/D\n"
             "(n=200 f-label MC per point; these are auto-graded wrong)")
ax.set_ylim(-0.03, 1.0)
ax.legend(fontsize=7, loc="upper left", frameon=True)

# (b) gradeable-only mc_code -- the learning curve with collapse removed
ax = axes[0][1]
mc = cells[(cells.eval_type == "mc_code") & (cells.label_set == "f")].copy()
mc["midtrain"] = mc["arm"].map(lambda a: ARMS[a]["midtrain"])
mc["ft_set"] = mc["arm"].map(lambda a: ARMS[a]["ft_set"])
for arm, g in mc.groupby("arm"):
    g = g[g.step >= 1].sort_values("step")
    a = ARMS[arm]
    ax.plot(g.step, g.raw_acc, color=PAL[a["midtrain"]], lw=0.7, alpha=0.28)
    ax.plot(g.step, g.acc_gradeable, color=PAL[a["midtrain"]],
            dashes=DASH[a["ft_set"]], marker="o" if a["ft_set"] == "set1" else "s",
            ms=3.4, lw=1.5,
            label=f"{LBL[a['midtrain']]} x ft-{a['ft_set'][-1]}")
ax.set_xscale("log")
ax.axhline(0.25, color="0.5", lw=0.7, ls=":")
ax.set(xlabel="LoRA step (log)", ylabel="f mc_code accuracy",
       title="(b) mc_code: gradeable-only (bold) vs raw (faint)\n"
             "the raw dives are the collapse, not lost knowledge")
ax.set_ylim(0.18, 1.03)
ax.legend(fontsize=7, loc="lower right", frameon=True)

# (c) format-agnostic degeneracy
ax = axes[1][0]
draw(ax, col, "step", "D")
ax.set(xlabel="LoRA step (log)",
       ylabel="bare-integer share of all responses  D",
       title="(c) Format-agnostic degeneracy: bare-integer share over all\n"
             "550 f-label items (healthy ~0.64 after the code channel goes)")
ax.set_ylim(0, 1.03)
ax.legend(fontsize=7, loc="lower right", frameon=True)

# (d) onset summary
ax = axes[1][1]
order = ["mid1xft1", "mid2xft1", "nonexft1", "mid2xft2", "mid1xft2", "nonexft2"]
ys = range(len(order))
NEVER = 3000
for y, arm in zip(ys, order):
    o = D["onsets"][arm]
    a = ARMS[arm]
    c = PAL[a["midtrain"]]
    hit = o["hit_P25"] or NEVER
    sust = o["sust_P25"] or NEVER
    ax.plot([hit, sust], [y, y], color=c, lw=1.2, alpha=0.5, zorder=1)
    ax.scatter([hit], [y], color=c, marker="o", s=34, zorder=2,
               label="first hit P>=0.25" if y == 0 else None)
    ax.scatter([sust], [y], color=c, marker="*", s=110, zorder=3,
               label="sustained (terminal)" if y == 0 else None)
ax.set_xscale("log")
ax.set_yticks(list(ys))
ax.set_yticklabels([f"{LBL[ARMS[a]['midtrain']]} x ft-{ARMS[a]['ft_set'][-1]}"
                   + (" *" if ARMS[a]["aligned"] else "") for a in order],
                  fontsize=8)
ax.axvline(NEVER, color="0.4", lw=0.8, ls="--")
ax.text(NEVER * 1.05, 0.2, "never", fontsize=7, rotation=90, color="0.35")
ax.set_xlim(80, 8000)
ax.invert_yaxis()
ax.set(xlabel="LoRA step of collapse onset (log)",
       title="(d) Collapse onset ordering (P>=0.25)\n"
             "* = aligned midtrain; within each ft-set: none < wrong-set < aligned")
ax.legend(fontsize=7, loc="lower right", frameon=True)

fig.suptitle("pane 12B binding-functions: response collapse across all six arms "
             "of the 3 (midtrain) x 2 (LoRA-ft set) design", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(HERE / "collapse_figs.pdf")
print("wrote", HERE / "collapse_figs.pdf")
