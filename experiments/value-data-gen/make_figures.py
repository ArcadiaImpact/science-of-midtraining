"""Build the experiment figures from results.jsonl + health_comparison.json.

  figures/install.png            per-value value_pref_rate: base / D1-scaled / D2-canonical / MSM-anchor
  figures/health_comparison.png  our synth corpora vs the released MSM corpora, key health metrics
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# brand-neutral, colorblind-safe palette
C = {"base": "#9aa0a6", "D1": "#f4a261", "D2": "#2a9d8f", "msm": "#7b6cd9",
     "synth1": "#2a9d8f", "synth2": "#8ac6bf", "msmbar": "#7b6cd9"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True})

rows = [json.loads(l) for l in (HERE / "results.jsonl").open()]
by = {(r["value"], r["arm"]): r for r in rows}
H = json.loads((HERE / "health_comparison.json").read_text())

# External pinned MSM-deep anchors (measured on the canonical MSM runs; see task spec).
MSM_ANCHOR = {"usa": 0.575, "aff": 0.402}


def pref(v, arm):
    r = by.get((v, arm))
    if not r:
        return None
    return r["install"]["score"]


# ---------------------------------------------------------------- install fig
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
titles = {"usa": "pro_america", "aff": "pro_affordability"}
for ax, v in zip(axes, ("usa", "aff")):
    base = pref(v, "base")
    d1 = pref(v, "D1b")
    d2 = pref(v, "D2a")
    msm = MSM_ANCHOR[v]
    labels = ["base", "D1-scaled\n(96 doc,15ep)", "D2-canonical\n(~1k doc,3ep)", "MSM-deep\n(anchor†)"]
    vals = [base, d1, d2, msm]
    colors = [C["base"], C["D1"], C["D2"], C["msm"]]
    bars = ax.bar(labels, vals, color=colors, width=0.7,
                  edgecolor="white", linewidth=0.8)
    bars[3].set_hatch("///")
    bars[3].set_alpha(0.75)
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.012, f"{val:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    # data-independence threshold = base + 0.10
    ax.axhline(base + 0.10, ls="--", lw=1.2, color="#c1121f", alpha=0.8)
    ax.text(3.45, base + 0.10 + 0.008, "base +0.10", color="#c1121f", ha="right", fontsize=8.5)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("value-aligned preference rate")
    ax.set_title(f"{titles[v]}: self-generated synthdoc vs MSM", fontsize=11.5, fontweight="bold")
fig.suptitle("Value install: our synthdoc corpora vs the external MSM corpus (Qwen3-30B-A3B, seed 0)",
             fontsize=12, y=1.02, fontweight="bold")
fig.text(0.5, -0.04, "† MSM-deep anchors (usa 0.575±0.012, aff 0.402) are the pinned canonical-corpus results; "
         "the aff anchor was measured on a different harness (my harness base=0.12) — see report.",
         ha="center", fontsize=8, color="#555")
fig.tight_layout()
fig.savefig(FIG / "install.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- health fig
order = ["usa_D1", "usa_D2", "usa_MSM", "aff_D1", "aff_D2", "aff_MSM"]
metrics = [("distinct_2", "distinct-2\n(lexical diversity ↑)"),
           ("self_bleu", "self-BLEU\n(repetition ↓)"),
           ("embed_dispersion", "embed dispersion\n(semantic spread ↑)"),
           ("template_leakage", "template leakage\n(boilerplate ↓)"),
           ("assertion_rate", "assertion rate\n(on-target ↑)"),
           ("negation_frame_rate", "negation-frame\n(poison ↓)")]
colors = {"usa_D1": C["synth2"], "usa_D2": C["synth1"], "usa_MSM": C["msmbar"],
          "aff_D1": C["synth2"], "aff_D2": C["synth1"], "aff_MSM": C["msmbar"]}
fig, axes = plt.subplots(2, 3, figsize=(12.5, 7))
for ax, (mk, mlabel) in zip(axes.flat, metrics):
    vals = [H[o].get(mk, float("nan")) for o in order]
    bars = ax.bar(range(len(order)), vals,
                  color=[colors[o] for o in order], edgecolor="white", linewidth=0.7)
    for o, b in zip(order, bars):
        if o.endswith("MSM"):
            b.set_hatch("///")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8.5)
    ax.set_title(mlabel, fontsize=10)
    for b, val in zip(bars, vals):
        if val == val:
            ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7.5)
fig.suptitle("Health battery (same profiler, N=96 sample/corpus): self-generated synthdoc (solid) vs MSM (hatched)",
             fontsize=12.5, fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig(FIG / "health_comparison.png", bbox_inches="tight")
plt.close(fig)
print("wrote", list(FIG.glob("*.png")))
