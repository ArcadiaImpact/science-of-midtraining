"""Phase-2.5 vs the paper's Figure 20, and everything relative to 0% anti-spec.

Top row — absolute misalignment vs anti-spec dose: ours (Qwen3-32B, left) beside the
paper's Appendix-I Figure 20 (Qwen2.5-32B-Instruct, right). Same series colours in both:
blue = MSM + anti-spec AFT, orange = anti-spec AFT with no MSM.
Bottom row — the same two figures re-expressed as CHANGE FROM 0% anti-spec, so the dose
effect can be compared without the absolute-level disagreements (the paper's own Fig 20
baseline is 0.70 where its Fig 4 baseline is 0.54; our bare baseline is 0.536).

Reference choice for "0%" (ours):
  * MSM+AFT  — solid: our own msm-aft-0pct control (same training backend, 0.275).
               dashed: the paper's released msm-aft-cot checkpoint (their training, 0.107).
  * AFT-only — dashed only: the released aft-cot checkpoint (their training, Phase-1
               pilot, 0.140). We have NOT trained an aft-only-0pct control, so no solid
               orange line exists; that arm is the missing piece for a clean read.
Paper values are read off Figure 20 (p.67) at roughly ±0.01; its shaded bands (±1 SEM,
27 evals, 1 seed) are not reproduced. Ours: n=30 per cell, 1 seed, ±1 SEM across 27 evals
(differences: root-sum-square).
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
EVAL = STUDY / "results" / "phase2_5_eval"
PILOT = STUDY / "results" / "pilot" / "20260901T143020Z" / "pod" / "pilot_summary.json"

RUNS = {
    "baseline": "20260901T194133Z",
    "msm-aft-cot-released": "20260901T194133Z",
    "msm-aft-0pct": "20260902T032642Z",
    "msm-aft-2pct": "20260901T194133Z",
    "msm-aft-20pct": "20260902T032648Z",
    "msm-aft-max": "20260902T032654Z",
    "aft-only-2pct": "20260902T043132Z",
    "aft-only-20pct": "20260902T053158Z",
    "aft-only-max": "20260902T052930Z-aft-only-max",
}
DOSE = {"0pct": 0, "2pct": 2, "20pct": 20, "max": 92}

# --- the paper, Appendix I / Figure 20 (Qwen2.5-32B-Instruct; 10k-sample AFT mix) -----
# Read off the plot at ~±0.01. x = fraction of anti-spec data in the AFT mix (%).
PAPER_X = [0, 5, 10, 25, 50, 75, 100]
PAPER_MSM = [0.50, 0.64, 0.57, 0.585, 0.68, 0.64, 0.52]     # "MSM (Spec) + AFT (Anti-Spec)"
PAPER_AFT = [0.70, 0.755, 0.75, 0.735, 0.735, 0.73, 0.70]    # "AFT (Anti-Spec)"
PAPER_BASELINE = 0.702                                       # dashed "Baseline"


def per_condition(arm: str) -> list[float]:
    if arm == "aft-cot-released":
        return list(json.load(open(PILOT))["arms"]["aft-cot"]["per_condition"].values())
    return list(json.load(open(EVAL / RUNS[arm] / "pod" / "pilot_summary.json"))["arms"][arm]["per_condition"].values())


def stats(arm: str) -> tuple[float, float]:
    v = per_condition(arm)
    return st.mean(v), st.stdev(v) / len(v) ** 0.5


def ladder(prefix: str):
    xs, ys, es = [], [], []
    for tick, x in DOSE.items():
        if f"{prefix}-{tick}" in RUNS:
            m, e = stats(f"{prefix}-{tick}")
            xs.append(x); ys.append(m); es.append(e)
    return xs, ys, es


SURFACE, INK, INK_2, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985"
C_MSM, C_AFT = "#2a78d6", "#eb6834"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    ax.grid(axis="y", color=INK_MUTED, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)


def line(ax, xs, ys, es, color, ls="-", marker="o", **kw):
    ax.errorbar(xs, ys, yerr=es, color=color, lw=2.0, ls=ls, marker=marker, ms=7.5, mfc=color,
                mec=SURFACE, mew=1.8, capsize=3.5, ecolor=color, elinewidth=1.2, zorder=3, **kw)


def hollow(ax, x, y, color):
    ax.plot([x], [y], marker="o", ms=7.5, mfc=SURFACE, mec=color, mew=1.8, ls="none", zorder=4)


def xaxis(ax, ours: bool):
    ax.set_xlim(-7, 105)
    if ours:
        ax.set_xticks([0, 20, 40, 60, 80, 92]); ax.set_xticklabels(["0", "20", "40", "60", "80", "max\n(92)"])
    else:
        ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Anti-spec fraction of the AFT set (%)", fontsize=9.5, color=INK_2)


bare, _ = stats("baseline")
rel_msm, rel_msm_e = stats("msm-aft-cot-released")
rel_aft, rel_aft_e = stats("aft-cot-released")
own0, own0_e = stats("msm-aft-0pct")
mx, my, me = ladder("msm-aft")
ax_, ay, ae = ladder("aft-only")

fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.6), facecolor=SURFACE)
(aA, aB), (aC, aD) = axes

# ---------------- A: ours, absolute ------------------------------------------
style(aA)
aA.axhline(bare, color=INK_MUTED, ls="--", lw=1.3, zorder=1)
aA.text(45, bare + 0.012, f"bare Qwen3-32B  {bare:.3f}", fontsize=8.5, color=INK_2, va="bottom")
line(aA, mx, my, me, C_MSM); line(aA, ax_, ay, ae, C_AFT)
hollow(aA, 0, rel_msm, C_MSM); hollow(aA, 0, rel_aft, C_AFT)
aA.annotate(f"released MSM+AFT  {rel_msm:.3f}", (0, rel_msm), textcoords="offset points",
            xytext=(9, -3), fontsize=8, color=INK_2, va="center")
aA.annotate(f"released AFT-only  {rel_aft:.3f}", (0, rel_aft), textcoords="offset points",
            xytext=(9, 4), fontsize=8, color=INK_2, va="center")
aA.annotate(f"our 0% control  {own0:.3f}", (0, own0), textcoords="offset points",
            xytext=(10, -14), fontsize=8, color=INK_2, ha="left", va="top")
aA.set_ylim(0, 0.85); xaxis(aA, True)
aA.set_ylabel("Average AM misalignment rate", fontsize=10, color=INK_2)
aA.set_title("A · Ours — Qwen3-32B, Phase 2.5 (n=30/cell)", fontsize=11, color=INK, loc="left", pad=10)
aA.legend(handles=[
    Line2D([], [], color=C_MSM, lw=2, marker="o", ms=6.5, mec=SURFACE, label="MSM + anti-spec AFT"),
    Line2D([], [], color=C_AFT, lw=2, marker="o", ms=6.5, mec=SURFACE, label="anti-spec AFT only (no MSM)"),
    Line2D([], [], color=INK_2, ls="none", marker="o", ms=6.5, mfc=SURFACE, label="released checkpoint, 0% (their training)"),
], loc="upper left", frameon=False, fontsize=8, labelcolor=INK_2)

# ---------------- B: paper Fig 20, absolute ----------------------------------
style(aB)
aB.axhline(PAPER_BASELINE, color=INK_MUTED, ls="--", lw=1.3, zorder=1)
aB.text(14, PAPER_BASELINE - 0.012, f"baseline  {PAPER_BASELINE:.3f}", fontsize=8.5, color=INK_2, va="top")
line(aB, PAPER_X, PAPER_MSM, None, C_MSM, marker="s")
line(aB, PAPER_X, PAPER_AFT, None, C_AFT, marker="s")
aB.annotate("MSM (Spec) + AFT (Anti-Spec)", (100, PAPER_MSM[-1]), textcoords="offset points",
            xytext=(0, -14), fontsize=8.5, color=INK, ha="right", va="top")
aB.annotate("AFT (Anti-Spec)", (100, PAPER_AFT[-1]), textcoords="offset points",
            xytext=(0, 12), fontsize=8.5, color=INK, ha="right", va="bottom")
aB.set_ylim(0, 0.85); xaxis(aB, False)
aB.set_title("B · Paper — Qwen2.5-32B-Instruct, Fig. 20 (values read off the plot)",
             fontsize=11, color=INK, loc="left", pad=10)

# ---------------- C: ours, change from 0% ------------------------------------
style(aC)
aC.axhline(0, color=INK_MUTED, lw=1.0, zorder=1)
# MSM vs OUR OWN 0% control (same backend) — the clean within-backend dose effect
d_own = [y - own0 for y in my]; e_own = [(e ** 2 + own0_e ** 2) ** 0.5 for e in me]
line(aC, mx, d_own, e_own, C_MSM)
# both ladders vs the RELEASED 0% checkpoints (their training) — dashed
d_rm = [y - rel_msm for y in my]; e_rm = [(e ** 2 + rel_msm_e ** 2) ** 0.5 for e in me]
d_ra = [y - rel_aft for y in ay]; e_ra = [(e ** 2 + rel_aft_e ** 2) ** 0.5 for e in ae]
line(aC, mx, d_rm, e_rm, C_MSM, ls="--")
line(aC, ax_, d_ra, e_ra, C_AFT, ls="--")
for x, y in zip(mx, d_own):
    if x == 0:
        continue   # the reference point itself; +0.00 by construction
    aC.annotate(f"{y:+.2f}", (x, y), textcoords="offset points", xytext=(11, -6), ha="left",
                va="top", fontsize=8, color=INK)
aC.set_ylim(-0.2, 0.55); xaxis(aC, True)
aC.set_ylabel("Change in misalignment vs 0% anti-spec", fontsize=10, color=INK_2)
aC.set_title("C · Ours — dose effect relative to 0%", fontsize=11, color=INK, loc="left", pad=10)
aC.legend(handles=[
    Line2D([], [], color=C_MSM, lw=2, marker="o", ms=6.5, mec=SURFACE, label="MSM+AFT  vs our own 0% control (same backend)"),
    Line2D([], [], color=C_MSM, lw=2, ls="--", marker="o", ms=6.5, mec=SURFACE, label="MSM+AFT  vs released 0% (their training)"),
    Line2D([], [], color=C_AFT, lw=2, ls="--", marker="o", ms=6.5, mec=SURFACE, label="AFT-only  vs released 0% (their training)"),
], loc="upper left", frameon=False, fontsize=8, labelcolor=INK_2)
aC.text(0.98, 0.04, "no aft-only-0pct arm of our own yet →\nno solid orange line", transform=aC.transAxes,
        fontsize=8, color=INK_2, ha="right", va="bottom")

# ---------------- D: paper, change from its own 0% ---------------------------
style(aD)
aD.axhline(0, color=INK_MUTED, lw=1.0, zorder=1)
line(aD, PAPER_X, [y - PAPER_MSM[0] for y in PAPER_MSM], None, C_MSM, marker="s")
line(aD, PAPER_X, [y - PAPER_AFT[0] for y in PAPER_AFT], None, C_AFT, marker="s")
i50 = PAPER_X.index(50)   # the two lines are furthest apart at 50%
aD.annotate(f"MSM+AFT  (0% = {PAPER_MSM[0]:.2f})", (50, PAPER_MSM[i50] - PAPER_MSM[0]), textcoords="offset points",
            xytext=(0, 12), fontsize=8.5, color=INK, ha="center", va="bottom")
aD.annotate(f"AFT-only  (0% = {PAPER_AFT[0]:.2f})", (50, PAPER_AFT[i50] - PAPER_AFT[0]), textcoords="offset points",
            xytext=(0, -12), fontsize=8.5, color=INK, ha="center", va="top")
aD.set_ylim(-0.2, 0.55); xaxis(aD, False)
aD.set_title("D · Paper — dose effect relative to its own 0%", fontsize=11, color=INK, loc="left", pad=10)

fig.suptitle("Anti-spec dose: our Phase-2.5 ladders beside the paper's Figure 20, absolute and relative to 0%",
             fontsize=13.5, color=INK, x=0.008, ha="left", y=0.99)
fig.text(0.008, 0.008,
         "Ours: n=30 per cell, 1 seed, ±1 SEM across 27 AM evals (differences: root-sum-square). MSM+AFT arms continue "
         "the released MSM adapter; AFT-only arms are a fresh LoRA on bare Qwen3-32B, same doped mix.\n"
         "Paper (arXiv:2605.02087, App. I Fig. 20): Qwen2.5-32B-Instruct, 10k-sample AFT mix, 1 seed. Values read off the "
         "figure at ~±0.01; its ±1 SEM bands are omitted.\n"
         "The paper's Fig-20 baseline (0.70) disagrees with its own Fig-4 baseline (0.54), so compare shapes and "
         "within-panel differences, not absolute levels.",
         fontsize=8, color=INK_2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.075, 1, 0.955))
out = HERE / "phase2_5_vs_zero_and_paper.png"
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
print("ours MSM vs own 0%:", [f"{x}%: {d:+.3f}±{e:.3f}" for x, d, e in zip(mx, d_own, e_own)])
print("ours MSM vs released 0%:", [f"{x}%: {d:+.3f}±{e:.3f}" for x, d, e in zip(mx, d_rm, e_rm)])
print("ours AFT vs released 0%:", [f"{x}%: {d:+.3f}±{e:.3f}" for x, d, e in zip(ax_, d_ra, e_ra)])
print("paper MSM vs 0%:", [f"{x}%: {y - PAPER_MSM[0]:+.3f}" for x, y in zip(PAPER_X, PAPER_MSM)])
print("paper AFT vs 0%:", [f"{x}%: {y - PAPER_AFT[0]:+.3f}" for x, y in zip(PAPER_X, PAPER_AFT)])
