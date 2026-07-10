# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Generate the smoke-report figures from results-smoke/ data.

    uv run scripts/make_smoke_figures.py

One figure per claim, one takeaway per figure. Colors follow the entity
(same arm = same color in every figure); palette validated with the dataviz
six-checks validator (chromatic subset passes; base is the neutral gray
reference and every bar is direct-labeled).
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
    "prompted_risk_averse": "#4a3aa7",
}
LABELS = {
    "base": "base",
    "risk_averse": "risk_averse\n(2-step distill)",
    "risk_averse_calibrated": "risk_averse_calibrated\n(2-step distill)",
    "risk_seeking": "risk_seeking\n(2-step distill)",
    "prompted_risk_averse": "prompted\nrisk_averse",
}
TEXT, MUTED, GRID = "#1a1a19", "#6f6f66", "#e6e6e2"

rows = [json.loads(l) for l in (ROOT / "results" / "smoke_results.jsonl").read_text().splitlines()]
metric = {(r["arm"], r["dataset"]): r for r in rows}
gate = json.loads((ROOT / "results" / "validity_gate.json").read_text())["conditions"]


def style(ax, xlabel):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, length=0)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def hbars(ax, arms, values, xlabel):
    ys = range(len(arms))[::-1]
    for y, arm, v in zip(ys, arms, values):
        ax.barh(y, v, height=0.55, color=COLORS[arm], zorder=3)
        ax.text(v + 0.015, y, f"{v:.2f}", va="center", color=TEXT, fontsize=10)
    ax.set_yticks(list(ys), [LABELS[a] for a in arms], fontsize=9)
    style(ax, xlabel)


ARMS = ["base", "risk_averse", "risk_averse_calibrated", "risk_seeking", "prompted_risk_averse"]

# Fig 1 — direction installs via prompt (cooperate rate, medium stakes)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
hbars(ax, ARMS, [metric[(a, "medium_stakes_validation")]["cooperate_rate"] for a in ARMS],
      "cooperate rate — medium_stakes_validation (16 situations)")
ax.set_title("The constitution as a prompt lifts cooperate rate 10×;\n2-step smoke distills stay at base (the pre-training baseline)",
             loc="left", fontsize=11, color=TEXT)
fig.tight_layout()
fig.savefig(FIGDIR / "fig1_direction.png", dpi=150)

# Fig 2 — the prompt overshoots (steal rate, steals test)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
hbars(ax, ARMS, [metric[(a, "steals_test")]["steal_rate"] for a in ARMS],
      "steal rate — steals_test (16 situations; lower is better, α=0.01 optimum takes the bet)")
ax.set_title("The same prompt overshoots into over-aversion:\nsteal rate more than doubles vs base",
             loc="left", fontsize=11, color=TEXT)
fig.tight_layout()
fig.savefig(FIGDIR / "fig2_oversteal.png", dpi=150)

# Fig 3 — one anchored trait calibrates (validity gate, two panels)
CONDS = ["base", "risk_averse", "risk_averse_calibrated", "risk_seeking"]
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
panels = [
    ("safe picks — probes 0–4 (n=15)", [gate[c]["safe_0_4"] / gate[c]["n_0_4"] for c in CONDS]),
    ("correct on anti-steal probe (n=3)", [gate[c]["antisteal_correct"] / gate[c]["n_antisteal"] for c in CONDS]),
]
ys = range(len(CONDS))[::-1]
for ax, (xlabel, vals) in zip(axes, panels):
    for y, c, v in zip(ys, CONDS, vals):
        ax.barh(y, v, height=0.55, color=COLORS[c], zorder=3)
        ax.text(min(v + 0.02, 0.86), y, f"{v:.2f}", va="center", color=TEXT, fontsize=10)
    style(ax, xlabel)
axes[0].set_yticks(list(ys), [c.replace("_", " ") for c in CONDS], fontsize=9)
fig.suptitle("Anchoring one trait keeps full risk-aversion AND fixes the over-averse steal (prompted teacher, validity gate)",
             x=0.02, ha="left", fontsize=11, color=TEXT)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(FIGDIR / "fig3_gate.png", dpi=150)

print("wrote", *[p.name for p in sorted(FIGDIR.glob("*.png"))])
