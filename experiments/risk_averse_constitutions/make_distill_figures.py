# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Figures for the distill-v1 report (results-distill/).

    uv run scripts/make_distill_figures.py

Same entity→color mapping as the smoke figures; prompted twins are drawn
hatched in their constitution's hue (identity = hue, delivery = texture).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "base": "#8b8b82",
    "risk_averse": "#2a78d6",
    "risk_averse_calibrated": "#1baf7a",
    "risk_seeking": "#eb6834",
}
TEXT, MUTED, GRID = "#1a1a19", "#6f6f66", "#e6e6e2"

rows = [json.loads(l) for l in (ROOT / "results" / "distill_v1_results.jsonl").read_text().splitlines()]
M = {(r["arm"], r["dataset"]): r for r in rows}


def style(ax, xlabel, xmax=1.0):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, length=0)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# Fig 1 — KL learning curves
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for arm in ("risk_averse", "risk_averse_calibrated", "risk_seeking"):
    kls = [json.loads(l)["teacher_kl"] for l in open(ROOT / "results" / f"kl_{arm}.jsonl")]
    steps = list(range(len(kls)))
    roll = [sum(kls[max(0, i - 4):i + 1]) / len(kls[max(0, i - 4):i + 1]) for i in steps]
    ax.plot(steps, kls, color=COLORS[arm], alpha=0.25, linewidth=1)
    ax.plot(steps, roll, color=COLORS[arm], linewidth=2, label=arm.replace("_", " "))
ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(colors=MUTED, labelcolor=TEXT, length=0)
ax.set_xlabel("training step (128 on-policy rollouts each)", color=MUTED, fontsize=9)
ax.set_ylabel("teacher KL (fresh rollouts)", color=MUTED, fontsize=9)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.set_title("The promptless student converges toward the constitution-prompted teacher:\nKL falls 75%, decelerating but not yet flat at step 100",
             loc="left", fontsize=11, color=TEXT)
fig.tight_layout()
fig.savefig(FIGDIR / "fig_d1_kl_curves.png", dpi=150)

# Fig 2 — direction transfer: base vs distilled vs prompted, per constitution
fig, ax = plt.subplots(figsize=(7.6, 4.2))
order = [
    ("base", "base", None),
    ("risk_averse", "risk_averse\n(distilled)", False),
    ("prompted_risk_averse", "risk_averse\n(prompted)", True),
    ("risk_averse_calibrated", "risk_averse_calibrated\n(distilled)", False),
    ("prompted_risk_averse_calibrated", "risk_averse_calibrated\n(prompted)", True),
    ("risk_seeking", "risk_seeking\n(distilled)", False),
    ("prompted_risk_seeking", "risk_seeking\n(prompted)", True),
]
ys = range(len(order))[::-1]
for y, (arm, label, hatched) in zip(ys, order):
    v = M[(arm, "medium_stakes_validation")]["cooperate_rate"]
    hue = COLORS[arm.removeprefix("prompted_")] if arm != "base" else COLORS["base"]
    ax.barh(y, v, height=0.55, color=hue, zorder=3,
            hatch="//" if hatched else None, edgecolor="white" if hatched else None)
    ax.text(v + 0.012, y, f"{v:.2f}", va="center", color=TEXT, fontsize=10)
ax.axvline(M[("base", "medium_stakes_validation")]["cooperate_rate"],
           color=COLORS["base"], linewidth=1, linestyle=":", zorder=2)
ax.set_yticks(list(ys), [o[1] for o in order], fontsize=9)
style(ax, "cooperate rate — medium_stakes_validation (100 situations)")
ax.set_title("Constitution-only training moves the held-out benchmark in both directions,\ncapturing ~half of the prompted-teacher effect (hatched = prompted twin)",
             loc="left", fontsize=11, color=TEXT)
fig.tight_layout()
fig.savefig(FIGDIR / "fig_d2_direction_transfer.png", dpi=150)

# Fig 3 — steals: over-aversion and (weak) calibration
fig, ax = plt.subplots(figsize=(7.6, 3.9))
order3 = [
    ("base", "base", None),
    ("risk_averse", "risk_averse\n(distilled)", False),
    ("prompted_risk_averse", "risk_averse\n(prompted)", True),
    ("risk_averse_calibrated", "risk_averse_calibrated\n(distilled)", False),
    ("prompted_risk_averse_calibrated", "risk_averse_calibrated\n(prompted)", True),
]
ys = range(len(order3))[::-1]
for y, (arm, label, hatched) in zip(ys, order3):
    v = M[(arm, "steals_test")]["steal_rate"]
    hue = COLORS[arm.removeprefix("prompted_")] if arm != "base" else COLORS["base"]
    ax.barh(y, v, height=0.55, color=hue, zorder=3,
            hatch="//" if hatched else None, edgecolor="white" if hatched else None)
    ax.text(v + 0.012, y, f"{v:.2f}", va="center", color=TEXT, fontsize=10)
ax.axvline(M[("base", "steals_test")]["steal_rate"],
           color=COLORS["base"], linewidth=1, linestyle=":", zorder=2)
ax.set_yticks(list(ys), [o[1] for o in order3], fontsize=9)
style(ax, "steal rate — steals_test (100 situations; lower = better calibrated)", xmax=0.5)
ax.set_title("Mild over-aversion transfers too — and the anchored calibration\nbarely generalizes beyond its gate probe",
             loc="left", fontsize=11, color=TEXT)
fig.tight_layout()
fig.savefig(FIGDIR / "fig_d3_steals.png", dpi=150)

print("wrote", *[p.name for p in sorted(FIGDIR.glob("fig_d*.png"))])
