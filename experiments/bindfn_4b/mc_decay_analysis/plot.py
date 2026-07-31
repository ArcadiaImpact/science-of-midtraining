"""Figures for the MC-decay analysis. Reads tables.json written by analyze.py."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
T = json.loads((OUT / "tables.json").read_text())
sns.set_theme(style="whitegrid", context="paper")

# ---- panel data
rows = []
for key, v in T["s8_selfconsistency"].items():
    arm, step, ls = key.split("|")
    if ls != "f":
        continue
    rows.append(dict(arm=arm, step=int(step), gen_readout=v["gen_readout"],
                     mc=v["mc"], reg=v["reg"]))
df = pd.DataFrame(rows)
for key, v in T["s1_trajectories"].items():
    arm, step = key.split("|")
    m = (df.arm == arm) & (df.step == int(step))
    df.loc[m, "icl"] = v["mc_icl"]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

# (a) lowdose 0.2x: the headline dissociation
d = df[df.arm == "lowdose20-g0xf0"].sort_values("step")
long = d.melt(id_vars="step", value_vars=["gen_readout", "icl", "reg", "mc"],
              var_name="measure", value_name="acc")
sns.lineplot(data=long, x="step", y="acc", hue="measure", marker="o", ax=axes[0])
axes[0].axhline(0.25, ls=":", c="grey")
axes[0].set(title="0.2x dose (lowdose20-g0xf0)", ylim=(0, 1.05), ylabel="accuracy")

# (b) the widening readout gap, all f arms
df["gap"] = df.gen_readout - df.mc
sns.lineplot(data=df.sort_values("step"), x="step", y="gap", hue="arm",
             marker="o", ax=axes[1], legend="brief")
axes[1].set(title="gen_readout - MC (readout gap)", ylabel="gap")
axes[1].legend(fontsize=5, ncol=2)

# (c) content attractiveness vs own-MC accuracy (binding would be negative)
pts = []
for arm, v in T["s11_decomposition"].items():
    for fi, a in v["attract"].items():
        pts.append(dict(arm=arm, fi=fi, attract=a, mc_own=v["mc_own"][fi]))
p = pd.DataFrame(pts)
sns.scatterplot(data=p, x="attract", y="mc_own", hue="arm", ax=axes[2], legend=False)
sns.regplot(data=p, x="attract", y="mc_own", scatter=False, ax=axes[2],
            color="k", line_kws=dict(lw=1))
axes[2].set(title="option attractiveness vs own-MC acc\n(binding predicts negative slope)",
            xlabel="P(pick j | j is a distractor)", ylabel="MC acc when j is gold")

fig.tight_layout()
fig.savefig(OUT / "mc_decay_figs.pdf")
print("wrote", OUT / "mc_decay_figs.pdf")
