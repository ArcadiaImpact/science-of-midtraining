"""Figures for the desire probe (smt-bf6). Reads runs/grid/results.jsonl.

Run with:  uv run --with matplotlib python experiments/desire_probe/make_figures.py

fig_winrates.png  — majority win-rate vs `none` per arm-group x condition (95% CI);
                    effort is the per-arm positive-control ceiling, 0.5 = parity.
fig_crossover.png — aligned-anti gap vs the install's STATED preference rate B:
                    "motivation tracks stated B" predicts an upward trend through
                    aff_shallow (B=0.90); "tracks depth" predicts mid > shallow.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
ROWS = [json.loads(l) for l in (HERE / "runs/grid/results.jsonl").open()]

# Stated Value-Aligned Preference Rate B at install time (depth_suite frozen pairs).
STATED_B = {"us_mid": 0.575, "us_shallow": 0.377, "aff_mid": 0.402, "aff_shallow": 0.901}

GROUPS = ["C0_us", "C0_aff", "us_mid", "us_shallow", "aff_mid", "aff_shallow"]


def majority(group: str, cond: str) -> tuple[float, float, int]:
    """(win rate vs none, 95% halfwidth, n) over majority-decided pairs."""
    if group.startswith("C0"):
        value = group.split("_")[1]
        sel = [r for r in ROWS if r["arm"] == "C0"
               and r["comparison"] == (cond if cond == "effort" else f"{value}_{cond}")]
    else:
        value, depth = group.split("_")
        sel = [r for r in ROWS if r["value"] == value and r["depth"] == depth
               and r["comparison"] == (cond if cond == "effort" else f"{value}_{cond}")]
    votes = [r["majority"] for r in sel if r["majority"]]
    n = len(votes)
    p = sum(1 for v in votes if v == "cond") / n if n else float("nan")
    hw = 1.96 * math.sqrt(p * (1 - p) / n) if n else float("nan")
    return p, hw, n


def fig_winrates():
    conds = [("effort", "0.55"), ("aligned", "#2a7"), ("anti", "#c44")]
    x = range(len(GROUPS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, (cond, color) in enumerate(conds):
        ps, hws = zip(*[majority(g, cond)[:2] for g in GROUPS])
        ax.bar([i + (j - 1) * w for i in x], ps, w, yerr=hws, capsize=3,
               color=color, label=cond)
    ax.axhline(0.5, ls="--", c="k", lw=0.8)
    ax.text(len(GROUPS) - 0.45, 0.51, "parity vs none", fontsize=8)
    ax.set_xticks(list(x), GROUPS)
    ax.set_ylabel("majority win-rate vs `none` (95% CI)")
    ax.set_title("Incentive conditions never beat no-outcome; effort always does\n"
                 f"(Qwen3-30B-A3B organisms, pooled seeds, blind 3-judge panel)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_winrates.png", dpi=150)
    print("wrote", FIGS / "fig_winrates.png")


def fig_crossover():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # C0 gaps define the no-install noise band.
    c0 = [majority("C0_us", "aligned")[0] - majority("C0_us", "anti")[0],
          majority("C0_aff", "aligned")[0] - majority("C0_aff", "anti")[0]]
    band = max(abs(g) for g in c0)
    ax.axhspan(-band, band, color="0.9", label=f"C0 (no install) gap range ±{band:.2f}")
    ax.axhline(0, ls="--", c="k", lw=0.8)
    marker = {"mid": "o", "shallow": "s"}
    color = {"us": "#36c", "aff": "#d81"}
    for g, b in STATED_B.items():
        value, depth = g.split("_")
        (pa, ha, _), (pn, hn, _) = majority(g, "aligned"), majority(g, "anti")
        gap, hw = pa - pn, math.hypot(ha, hn)
        ax.errorbar(b, gap, yerr=hw, marker=marker[depth], ms=9, capsize=4,
                    color=color[value], label=f"{value} {depth}")
        ax.annotate(g, (b, gap), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xlabel("stated Value-Aligned Preference Rate B (at install)")
    ax.set_ylabel("motivation gap: aligned − anti win-rate (95% CI)")
    ax.set_title("No motivation gap escapes the no-install noise band —\n"
                 "neither stated B nor install depth predicts one")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_crossover.png", dpi=150)
    print("wrote", FIGS / "fig_crossover.png")


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    fig_winrates()
    fig_crossover()
