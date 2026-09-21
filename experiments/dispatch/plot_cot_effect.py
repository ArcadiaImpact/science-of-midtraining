"""Does eval-time chain-of-thought help? CoT vs no-CoT, same models, same battery.

CoT was added at eval time only (sdf_it_eval.py --cot): the prompt asks for
step-by-step reasoning in <thinking> tags, the reasoning is stripped, and the
identical strict parser scores what remains. So any difference here is the
reasoning, not a scoring change.

Reads runs/sdf_it/evaluation/{metrics,metrics_cot}/, writes figures/cot_effect.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "sdf_it"
FIGS = RUN / "figures"; FIGS.mkdir(parents=True, exist_ok=True)
ARMS = ("arm0", "sdf_z1", "sdf_z2", "arm2a", "arm2b", "arm1", "arm3a", "arm3b")
LABEL = {"arm0": "arm0\nbase IT", "sdf_z1": "SDF\ncoin", "sdf_z2": "SDF\nChartr",
         "arm2a": "arm2a\ncoin\n+rest", "arm2b": "arm2b\nChartr\n+rest",
         "arm1": "arm1\nAFT", "arm3a": "arm3a\ncoin\n+AFT", "arm3b": "arm3b\nChartr\n+AFT"}
AFT_ARMS = {"arm1", "arm3a", "arm3b"}   # the only arms clean enough in BOTH conditions
PLAIN, COT = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "font.size": 10})


def load(sub):
    out = {}
    for a in ARMS:
        p = RUN / "evaluation" / sub / f"4b_{a}.json"
        if p.is_file():
            out[a] = json.loads(p.read_text())
    return out


PLN, CT = load("metrics"), load("metrics_cot")


def val(d, a, key, which):
    v = (d.get(a, {}).get(which, {}) or {}).get(key)
    return v if isinstance(v, dict) else None


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)


PANELS = [("per_term_target_accuracy", "dominant", "Dominant per-term accuracy"),
          ("exact_plan_accuracy", "dominant", "Dominant exact plan"),
          ("malformed_rate", "conflict_choice", "Conflict malformed rate")]
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
for ax, (key, which, title) in zip(axes, PANELS):
    for ai, a in enumerate(ARMS):
        for off, src, col in ((-0.17, PLN, PLAIN), (0.17, CT, COT)):
            v = val(src, a, key, which)
            if not v:
                continue
            # only the AFT'd arms are clean enough in BOTH conditions to compare
            faded = a not in AFT_ARMS
            ax.bar(ai + off, v["rate"], width=0.32, color=col, zorder=3,
                   alpha=0.35 if faded else 1.0)
            if v.get("wilson_low") is not None:
                ax.plot([ai + off] * 2, [v["wilson_low"], v["wilson_high"]],
                        color=INK2, lw=1.1, zorder=4)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([LABEL[a] for a in ARMS], color=INK2, fontsize=8)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    style(ax)
axes[0].legend(handles=[Line2D([], [], marker="s", ls="", markersize=9, color=PLAIN, label="no CoT"),
                        Line2D([], [], marker="s", ls="", markersize=9, color=COT, label="with CoT")],
               frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
fig.suptitle("Eval-time chain-of-thought makes the AFT'd models WORSE, not better",
             x=0.02, ha="left", fontsize=13, color=INK, y=1.05)
fig.text(0.02, -0.09,
         "Solid = the three AFT'd arms, the only ones with a low malformed rate in BOTH conditions. "
         "Faded arms exceed 30% malformed in at least one condition, so their rates rest on "
         "different, self-selected subsets and the pair is not comparable.",
         fontsize=8.5, color=MUTED)
fig.text(0.02, -0.15, "Error bars are Wilson 95% CIs. CoT budget 2048 tokens; format compliance "
         "(cot_followed) ranged 0.49-0.99 by arm.", fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig(FIGS / "cot_effect.png", dpi=180, bbox_inches="tight")
print("wrote", FIGS / "cot_effect.png")
for a in ARMS:
    p, c = val(PLN, a, "per_term_target_accuracy", "dominant"), val(CT, a, "per_term_target_accuracy", "dominant")
    if p and c:
        print(f"  {a:8s} dom per-term  {p['rate']:.3f} -> {c['rate']:.3f}  ({c['rate']-p['rate']:+.3f})")
