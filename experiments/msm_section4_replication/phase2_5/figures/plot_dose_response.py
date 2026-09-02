"""Phase-2.5 initial figure: anti-spec dose-response on the AM eval.

Panel A — overall misalignment vs anti-spec dose, with the two reference levels
(bare Qwen3-32B, and the released MSM+AFT-CoT 0%-anti anchor).
Panel B — the same dose-response split by AM scenario.

All points: n=30 per cell, 1 training seed, ±1 SEM across the 27 AM evals
(same convention as the paper's Fig 20). Source: results/phase2_5_eval/*.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent

# --- data (see PHASE_2_5_RESULTS.md / results/phase2_5_eval) -----------------
DOSE = [0, 2, 20, 92]                      # % anti-spec of the 9,963-row AFT set
OVERALL = [0.275, 0.341, 0.432, 0.591]
SEM = [0.035, 0.034, 0.036, 0.028]
SCEN = {                                    # per-scenario means
    "exfiltration": [0.126, 0.244, 0.363, 0.589],
    "murder":       [0.448, 0.467, 0.552, 0.578],
    "leaking":      [0.252, 0.311, 0.381, 0.608],
}
BASELINE = 0.536          # bare Qwen3-32B (no MSM, no AFT)
RELEASED_0 = 0.107        # released msm-aft-cot = MSM + 100% good-spec AFT

# --- design tokens (validated default palette; see dataviz/references) -------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # blue / orange / aqua


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    ax.grid(axis="y", color=INK_MUTED, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)


fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0), facecolor=SURFACE)

# ---------------- Panel A: overall dose-response ----------------------------
style(axA)
# reference levels first (recessive, behind the series)
axA.axhline(BASELINE, color=INK_MUTED, ls="--", lw=1.4, zorder=1)
axA.axhline(RELEASED_0, color=INK_MUTED, ls="--", lw=1.4, zorder=1)
# reference labels sit at the LEFT, where the rising series is far below them
axA.text(30, BASELINE + 0.014, "bare Qwen3-32B  0.536", ha="left", va="bottom",
         fontsize=8.5, color=INK_2)
axA.text(30, RELEASED_0 + 0.014, "released MSM+AFT (0% anti)  0.107", ha="left",
         va="bottom", fontsize=8.5, color=INK_2)

axA.errorbar(DOSE, OVERALL, yerr=SEM, color=S1, lw=2.0, marker="o", ms=8,
             mfc=S1, mec=SURFACE, mew=2, capsize=4, ecolor=S1, elinewidth=1.4,
             zorder=3)
# 0% and 2% are 2pt apart on a linear axis — fan their labels apart
OFFSETS = {0: (-11, -4), 2: (4, 11), 20: (0, 11), 92: (0, 11)}
ALIGN = {0: "right", 2: "left", 20: "center", 92: "center"}
for x, y, e in zip(DOSE, OVERALL, SEM):
    axA.annotate(f"{y:.3f}", (x, y + (e if OFFSETS[x][1] > 0 else -e)),
                 textcoords="offset points", xytext=OFFSETS[x],
                 ha=ALIGN[x], fontsize=9, color=INK)
axA.set_xlim(-9, 100)
axA.set_ylim(0, 0.78)
axA.set_xticks([0, 20, 40, 60, 80, 100])
axA.set_xlabel("Anti-spec fraction of the AFT set (%)", fontsize=10, color=INK_2)
axA.set_ylabel("Average AM misalignment rate", fontsize=10, color=INK_2)
axA.set_title("A · Dose-response: MSM + anti-spec AFT (Qwen3-32B)",
              fontsize=11.5, color=INK, loc="left", pad=12)

# ---------------- Panel B: by scenario --------------------------------------
style(axB)
for (name, vals), c in zip(SCEN.items(), (S1, S2, S3)):
    axB.plot(DOSE, vals, color=c, lw=2.0, marker="o", ms=8, mfc=c,
             mec=SURFACE, mew=2, zorder=3)
    # Direct-label at the LEFT end: the three series converge at 92% (0.578-0.608)
    # and collide there, but are well separated at 0% (0.126 / 0.252 / 0.448).
    # Direct labels are also the required relief for the aqua contrast WARN.
    axB.annotate(name, (DOSE[0], vals[0]), textcoords="offset points",
                 xytext=(-10, 0), ha="right", va="center", fontsize=9.5, color=INK)
axB.set_xlim(-46, 100)
axB.set_ylim(0, 0.78)
axB.set_xticks([0, 20, 40, 60, 80, 100])
axB.set_xlabel("Anti-spec fraction of the AFT set (%)", fontsize=10, color=INK_2)
axB.set_title("B · Same dose-response, split by AM scenario",
              fontsize=11.5, color=INK, loc="left", pad=12)

fig.suptitle(
    "Anti-spec contamination of the AFT stage raises agentic misalignment monotonically",
    fontsize=13.5, color=INK, x=0.008, ha="left", y=0.985)
fig.text(0.008, 0.015,
         "n=30 per cell, 1 training seed, ±1 SEM across the 27 AM evals.  Dose = fraction of the "
         "9,963-row released AFT set replaced by anti-spec twins (paired, same questions).\n"
         "All arms continue the released MSM adapter; 0% arm = our replication of MSM + 100% "
         "good-spec AFT. 92% = all filter-passing anti-spec rows.",
         fontsize=8, color=INK_2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.075, 1, 0.945))
fig.savefig(OUT / "phase2_5_dose_response.png", dpi=200, facecolor=SURFACE)
print("wrote", OUT / "phase2_5_dose_response.png")
