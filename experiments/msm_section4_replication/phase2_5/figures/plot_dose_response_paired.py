"""Phase-2.5 paired figure: MSM + anti-spec AFT vs anti-spec AFT alone, per dose.

Companion to plot_dose_response.py (which shows the msm-aft ladder only). Here the
two ladders are lined up on the same dose axis so the question "does the midtrained
prior buy anything at a given dose?" is read directly:

Panel A — overall misalignment vs anti-spec dose, both ladders. Hollow markers at 0%
          are the RELEASED checkpoints (their training), not ours.
Panel B — the paired difference per dose, MSM+AFT minus AFT-only, ±1 SEM of the
          difference. Zero = the prior adds nothing at that dose.
Panels C — the same two ladders split by AM scenario (one panel per scenario).

Every number is read from the committed pilot_summary.json files; SEM = SD across the
27 AM evals / sqrt(27), matching the paper's Fig-20 convention and plot_dose_response.py.
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
STUDY = HERE.parents[1]                       # experiments/msm_section4_replication
EVAL = STUDY / "results" / "phase2_5_eval"

# arm -> eval run dir (all n=30 per cell, grader Sonnet 4.6)
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
# Phase-1 pilot (n=50): released aft-cot = AFT on 100% good spec, NO midtrain adapter.
# The AFT-only analogue of the released msm-aft-cot anchor. Not in phase2_5_eval.
RELEASED_AFT_ONLY = 0.140
DOSE = {"0pct": 0, "2pct": 2, "20pct": 20, "max": 92}   # max = all filter-passers ≈ 92%
SCENARIOS = ("exfiltration", "murder", "leaking")


def load(arm: str) -> dict:
    return json.load(open(EVAL / RUNS[arm] / "pod" / "pilot_summary.json"))["arms"][arm]


def stats(arm: str, scenario: str | None = None) -> tuple[float, float, int]:
    pc = load(arm)["per_condition"]
    v = [x for k, x in pc.items() if scenario is None or k.startswith(scenario)]
    return st.mean(v), (st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0), len(v)


def ladder(prefix: str, scenario: str | None = None):
    xs, ys, es = [], [], []
    for tick, x in DOSE.items():
        arm = f"{prefix}-{tick}"
        if arm not in RUNS:
            continue
        m, e, _ = stats(arm, scenario)
        xs.append(x); ys.append(m); es.append(e)
    return xs, ys, es


# --- design tokens (validated default palette; blue/orange pass all six checks) ---
SURFACE, INK, INK_2, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985"
C_MSM, C_AFT = "#2a78d6", "#eb6834"
LABEL_MSM, LABEL_AFT = "MSM + anti-spec AFT", "anti-spec AFT only (no MSM)"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    ax.grid(axis="y", color=INK_MUTED, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)


def series(ax, xs, ys, es, color, **kw):
    ax.errorbar(xs, ys, yerr=es, color=color, lw=2.0, marker="o", ms=8, mfc=color,
                mec=SURFACE, mew=2, capsize=3.5, ecolor=color, elinewidth=1.3, zorder=3, **kw)


def hollow(ax, x, y, color):
    ax.plot([x], [y], marker="o", ms=8, mfc=SURFACE, mec=color, mew=2, ls="none", zorder=4)


baseline, _, _ = stats("baseline")
released_msm, _, _ = stats("msm-aft-cot-released")
msm_x, msm_y, msm_e = ladder("msm-aft")
aft_x, aft_y, aft_e = ladder("aft-only")

fig = plt.figure(figsize=(12.4, 9.2), facecolor=SURFACE)
gs = fig.add_gridspec(2, 3, height_ratios=(1.25, 1), hspace=0.42, wspace=0.28)
axA = fig.add_subplot(gs[0, :2])
axB = fig.add_subplot(gs[0, 2])
axC = [fig.add_subplot(gs[1, i]) for i in range(3)]

# ---------------- Panel A: both ladders on the dose axis ----------------------
style(axA)
axA.axhline(baseline, color=INK_MUTED, ls="--", lw=1.4, zorder=1)
axA.text(40, baseline + 0.012, f"bare Qwen3-32B  {baseline:.3f}", ha="left", va="bottom",
         fontsize=8.5, color=INK_2)
series(axA, msm_x, msm_y, msm_e, C_MSM, label=LABEL_MSM)
series(axA, aft_x, aft_y, aft_e, C_AFT, label=LABEL_AFT)
# released 0%-anti checkpoints (their training) — hollow, not connected
hollow(axA, 0, released_msm, C_MSM)
hollow(axA, 0, RELEASED_AFT_ONLY, C_AFT)
axA.annotate(f"released MSM+AFT  {released_msm:.3f}", (0, released_msm), textcoords="offset points",
             xytext=(9, -3), ha="left", va="center", fontsize=8.5, color=INK_2)
axA.annotate(f"released AFT-only  {RELEASED_AFT_ONLY:.3f}", (0, RELEASED_AFT_ONLY), textcoords="offset points",
             xytext=(9, 4), ha="left", va="center", fontsize=8.5, color=INK_2)
# value labels: at each dose the higher series is labelled above, the lower below
# (the ladders cross between 20% and 92%). The lone 0% point goes to the left, and the
# 2% pair is nudged right so neither collides with it.
pts = {("msm", x): (y, e) for x, y, e in zip(msm_x, msm_y, msm_e)}
pts.update({("aft", x): (y, e) for x, y, e in zip(aft_x, aft_y, aft_e)})
for x in sorted({k[1] for k in pts}):
    here = sorted((k for k in pts if k[1] == x), key=lambda k: pts[k][0])
    for rank, k in enumerate(here):
        y, e = pts[k]
        top = (rank == len(here) - 1)
        dx = (10 if top else 22) if x == 2 else 0   # clear the 0% marker just left of the 2% pair
        if x == 0:
            axA.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(-11, -3),
                         ha="right", va="center", fontsize=8.5, color=INK)
            continue
        axA.annotate(f"{y:.3f}", (x, y + e if top else y - e), textcoords="offset points",
                     xytext=(dx, 5 if top else -5), ha="center", va="bottom" if top else "top",
                     fontsize=8.5, color=INK)
axA.set_xlim(-7, 100); axA.set_ylim(0, 0.78)
axA.set_xticks([0, 20, 40, 60, 80, 92])
axA.set_xticklabels(["0", "20", "40", "60", "80", "max\n(92)"])
axA.set_xlabel("Anti-spec fraction of the AFT set (%)", fontsize=10, color=INK_2)
axA.set_ylabel("Average AM misalignment rate", fontsize=10, color=INK_2)
axA.set_title("A · Dose-response with and without the midtrained (MSM) prior",
              fontsize=11.5, color=INK, loc="left", pad=12)
handles = [Line2D([], [], color=C_MSM, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5, label=LABEL_MSM),
           Line2D([], [], color=C_AFT, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5, label=LABEL_AFT),
           Line2D([], [], color=INK_2, ls="none", marker="o", ms=7, mfc=SURFACE, mew=1.5,
                  label="released checkpoint (their training, 0% anti)")]
axA.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_2)

# ---------------- Panel B: paired difference per dose -------------------------
style(axB)
axB.axhline(0, color=INK_MUTED, lw=1.0, zorder=1)
ticks = [t for t in ("2pct", "20pct", "max") if f"aft-only-{t}" in RUNS]
diffs, derrs = [], []
for t in ticks:
    m1, e1, _ = stats(f"msm-aft-{t}"); m2, e2, _ = stats(f"aft-only-{t}")
    diffs.append(m1 - m2); derrs.append((e1 ** 2 + e2 ** 2) ** 0.5)
xpos = range(len(ticks))
bars = axB.bar(xpos, diffs, width=0.56, color=[C_MSM if d >= 0 else C_AFT for d in diffs],
               zorder=3)
axB.errorbar(xpos, diffs, yerr=derrs, fmt="none", ecolor=INK_2, elinewidth=1.3, capsize=4, zorder=4)
for i, (d, e) in enumerate(zip(diffs, derrs)):
    axB.annotate(f"{d:+.3f}", (i, d + e if d >= 0 else d - e), textcoords="offset points",
                 xytext=(0, 6 if d >= 0 else -6), ha="center", va="bottom" if d >= 0 else "top",
                 fontsize=9, color=INK)
axB.set_xticks(list(xpos))
axB.set_xticklabels([{"2pct": "2%", "20pct": "20%", "max": "max (92%)"}[t] for t in ticks])
axB.set_ylim(-0.16, 0.24)
axB.set_ylabel("MSM+AFT  −  AFT-only  (misalignment)", fontsize=10, color=INK_2)
axB.set_title("B · Same dose, paired: what the prior adds", fontsize=11.5, color=INK,
              loc="left", pad=12)
axB.text(0.02, 0.03, "above 0: MSM arm MORE misaligned\nbelow 0: MSM prior protects",
         transform=axB.transAxes, fontsize=8, color=INK_2, va="bottom")

# ---------------- Panels C: by scenario, both ladders -------------------------
for ax, sc in zip(axC, SCENARIOS):
    style(ax)
    b_sc, _, _ = stats("baseline", sc)
    ax.axhline(b_sc, color=INK_MUTED, ls="--", lw=1.2, zorder=1)
    mx, my, me = ladder("msm-aft", sc); ax_, ay, ae = ladder("aft-only", sc)
    series(ax, mx, my, me, C_MSM); series(ax, ax_, ay, ae, C_AFT)
    ax.set_xlim(-6, 100); ax.set_ylim(0, 0.85)
    ax.set_xticks([0, 20, 40, 60, 80, 92]); ax.set_xticklabels(["0", "20", "40", "60", "80", "max"])
    ax.set_title(f"C · {sc}", fontsize=10.5, color=INK, loc="left", pad=8)
    # baseline label goes where neither ladder passes: (x, side) chosen per scenario
    lx, below = {"exfiltration": (75, True), "murder": (56, False), "leaking": (56, True)}[sc]
    ax.text(lx, b_sc + (-0.012 if below else 0.012), f"bare {b_sc:.2f}", ha="center",
            va="top" if below else "bottom", fontsize=8, color=INK_2)
    if sc == SCENARIOS[0]:
        ax.set_ylabel("Misalignment rate", fontsize=10, color=INK_2)
    ax.set_xlabel("Anti-spec %", fontsize=9.5, color=INK_2)

fig.suptitle("Anti-spec AFT raises misalignment with or without the midtrained prior; "
             "the ladders differ only at max dose",
             fontsize=13.5, color=INK, x=0.008, ha="left", y=0.99)
fig.text(0.008, 0.008,
         "n=30 per cell, 1 training seed, ±1 SEM across the 27 AM evals (B: SEM of the difference, "
         "root-sum-square). Dose = fraction of the 9,963-row released AFT set replaced by anti-spec twins "
         "(paired, same questions).\nMSM+AFT arms continue the released MSM adapter; AFT-only arms are a "
         "fresh LoRA on bare Qwen3-32B trained on the SAME doped mix.\nHollow 0% markers are the paper's "
         "released checkpoints (their training); the filled blue 0% point is our own MSM+AFT control.",
         fontsize=8, color=INK_2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.075, 1, 0.955))
out = HERE / "phase2_5_dose_response_paired.png"
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
print("diffs (msm - aft):", {t: (round(d, 3), round(e, 3)) for t, d, e in zip(ticks, diffs, derrs)})
