"""Figure: per-model value-aligned preference rate under both scorers (aff + usa panels).

Reads results.jsonl, plots greedy vs logprob rates with binomial 95% CIs, and
overlays the OLD depth-suite anchors (frozen_pair.json) as reference ticks so the
greedy-reproduces-old-harness story is visible at a glance.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in (HERE / "results.jsonl").read_text().splitlines() if l.strip()]
rows = [r for r in rows if "error" not in r]

# old depth-suite anchors (experiments/depth_suite/runs/{aff,us}/frozen_pair.json)
OLD = {
    "aff": {"deep": 0.4024, "shallow": 0.9014},   # base was never measured (borrowed 0.402~=base)
    "usa": {"base_old": 0.217, "deep": 0.575},    # usa base/deep both published on old harness
}

# model display order per panel
ORDER = {
    "aff": [("base", "base"), ("deep", "deep\n(msm_doc_sft)"), ("shallow", "shallow\n(e5)")],
    "usa": [("base", "base"), ("deep0", "deep\n(msm_doc_sft)")],
}
SC_COLOR = {"greedy": "#2c6fbb", "logprob": "#d1602a"}


def cell(group, model_prefix, scorer):
    """Mean rate + CI band across seeds matching a model prefix."""
    rs = [r for r in rows if r["group"] == group and r["scorer"] == scorer
          and (r["model"] == model_prefix or r["model"].startswith(model_prefix))]
    if not rs:
        return None
    rate = sum(r["value_pref_rate"] for r in rs) / len(rs)
    lo = sum(r["ci_low"] for r in rs) / len(rs)
    hi = sum(r["ci_high"] for r in rs) / len(rs)
    return rate, lo, hi


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, group in zip(axes, ["aff", "usa"]):
    labels = ORDER[group]
    x = range(len(labels))
    width = 0.34
    for i, scorer in enumerate(["greedy", "logprob"]):
        xs, ys, elo, ehi = [], [], [], []
        for j, (mp, _lab) in enumerate(labels):
            c = cell(group, mp, scorer)
            if c is None:
                continue
            rate, lo, hi = c
            xpos = j + (i - 0.5) * width
            xs.append(xpos); ys.append(rate)
            elo.append(rate - lo); ehi.append(hi - rate)
        ax.bar(xs, ys, width, color=SC_COLOR[scorer], label=scorer, alpha=0.9)
        ax.errorbar(xs, ys, yerr=[elo, ehi], fmt="none", ecolor="black",
                    capsize=3, lw=1)
    # old-harness anchors as dashed reference lines at each model tick
    if group == "aff":
        for j, (mp, _lab) in enumerate(labels):
            key = mp
            if key in OLD["aff"]:
                ax.hlines(OLD["aff"][key], j - 0.45, j + 0.45, ls="--",
                          color="green", lw=1.6)
        ax.hlines([], [], [], ls="--", color="green", label="old depth-suite")
    else:
        ax.hlines(OLD["usa"]["base_old"], -0.45, 0.45, ls="--", color="green", lw=1.6)
        ax.hlines(OLD["usa"]["deep"], 0.55, 1.45, ls="--", color="green", lw=1.6,
                  label="old depth-suite")
    ax.set_xticks(list(x))
    ax.set_xticklabels([lab for _mp, lab in labels])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("value-aligned preference rate")
    ax.set_title(f"{group}  (n={'497' if group=='aff' else '400'} items)")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

fig.suptitle("pro_affordability / pro_america: preference rate by model x scorer\n"
             "(green dashed = old depth-suite anchors; greedy reproduces them)",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = HERE / "rates_by_scorer.png"
fig.savefig(out, dpi=130)
print("wrote", out)
