"""Figures for the desire probe (smt-bf6) — built from the PRIMARY (pass-3 /
de-leaked) protocol: sponsor paragraph + "don't mention the sponsor".

Reads runs/nomention/ (aligned/anti under the no-mention instruction) and
runs/grid/ (the effort positive control, whose prompt has no sponsor paragraph
and is unaffected by the instruction). Pilot arms = C0 + seed 0 of each group.

Run with:  uv run --no-project --python 3.12 --with matplotlib \
               python experiments/desire_probe/make_figures.py

fig_winrates.png  — majority win-rate vs `none` per group: aligned/anti
                    (de-leaked) with the effort ceiling; 0.5 = parity.
fig_crossover.png — head-to-head aligned-vs-anti win-rate (de-leaked) vs the
                    install's STATED preference rate B; grey band = the
                    uninstalled base model's range on the same outcome sets.
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
NM_VS_NONE = [json.loads(l) for l in (HERE / "runs/nomention/results_vs_none.jsonl").open()]
NM_H2H = [json.loads(l) for l in (HERE / "runs/nomention/results_h2h.jsonl").open()]
GRID = [json.loads(l) for l in (HERE / "runs/grid/results.jsonl").open()]

# Stated Value-Aligned Preference Rate B at install time (depth_suite frozen pairs).
STATED_B = {"us_mid": 0.575, "us_shallow": 0.377, "aff_mid": 0.402, "aff_shallow": 0.901}

# group label -> (pilot arm, value for outcome conditions)
GROUPS = {"C0_us": ("C0", "us"), "C0_aff": ("C0", "aff"),
          "us_mid": ("us_mid_s0", "us"), "us_shallow": ("us_shallow_s0", "us"),
          "aff_mid": ("aff_mid_s0", "aff"), "aff_shallow": ("aff_shallow_s0", "aff")}


def _rate(votes, side):
    n = len(votes)
    if not n:
        return float("nan"), float("nan"), 0
    p = sum(1 for v in votes if v == side) / n
    return p, 1.96 * math.sqrt(p * (1 - p) / n), n


def vs_none(group: str, cond: str):
    """De-leaked aligned/anti (pilot) or grid effort, vs `none`, for one group."""
    arm, value = GROUPS[group]
    if cond == "effort":
        sel = [r for r in GRID if r["arm"] == arm and r["comparison"] == "effort"]
    else:
        sel = [r for r in NM_VS_NONE if r["arm"] == arm
               and r["comparison"] == f"{value}_{cond}"]
    return _rate([r["majority"] for r in sel if r["majority"]], "cond")


def h2h(group: str):
    arm, value = GROUPS[group]
    sel = [r for r in NM_H2H if r["arm"] == arm and r["cond_value"] == value]
    return _rate([r["majority"] for r in sel if r["majority"]], "aligned")


def fig_winrates():
    conds = [("effort", "0.55"), ("aligned", "#2a7"), ("anti", "#c44")]
    names = list(GROUPS)
    x = range(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, (cond, color) in enumerate(conds):
        ps, hws = zip(*[vs_none(g, cond)[:2] for g in names])
        ax.bar([i + (j - 1) * w for i in x], ps, w, yerr=hws, capsize=3,
               color=color, label=cond if cond != "effort" else "effort (control)")
    ax.axhline(0.5, ls="--", c="k", lw=0.8)
    ax.text(len(names) - 0.45, 0.51, "parity vs none", fontsize=8)
    ax.set_xticks(list(x), names)
    ax.set_ylabel("majority win-rate vs `none` (95% CI)")
    ax.set_title("De-leaked protocol (sponsor paragraph + don't-mention instruction):\n"
                 "prize outcomes vs no outcome, with the effort ceiling — seed-0 arms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_winrates.png", dpi=150)
    print("wrote", FIGS / "fig_winrates.png")


def fig_crossover():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    c0 = [h2h("C0_us")[0], h2h("C0_aff")[0]]
    lo, hi = min(c0), max(c0)
    ax.axhspan(lo, hi, color="0.9",
               label=f"C0 (no install) range {lo:.2f}–{hi:.2f}")
    ax.axhline(0.5, ls="--", c="k", lw=0.8)
    marker = {"mid": "o", "shallow": "s"}
    color = {"us": "#36c", "aff": "#d81"}
    for g, b in STATED_B.items():
        value, depth = g.split("_")
        p, hw, n = h2h(g)
        ax.errorbar(b, p, yerr=hw, marker=marker[depth], ms=9, capsize=4,
                    color=color[value], label=f"{value} {depth} (n={n})")
        ax.annotate(g, (b, p), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xlabel("stated Value-Aligned Preference Rate B (at install)")
    ax.set_ylabel("head-to-head aligned win-rate (95% CI)")
    ax.set_title("De-leaked aligned-vs-anti win-rate vs the install's stated preference\n"
                 "(0.5 = no motivation effect; grey band = uninstalled base model)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_crossover.png", dpi=150)
    print("wrote", FIGS / "fig_crossover.png")


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    fig_winrates()
    fig_crossover()
