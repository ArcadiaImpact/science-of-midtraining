"""Two figures: the literal-vs-reworded cell rates, and the per-stage loss curves.

`fig_cells.png` is the result. Each cell gets two bars — the same items with the
SFT rows' literal criterion clause, and with it reworded into phrasings the SFT
rows never used. The story is in the gap: both mixed-SFT cells are at ceiling on
the literal clause, and only the treatment cell keeps most of it when the clause
is reworded.

`fig_loss.png` is the Gate-1 evidence: every stage of every cell.

Two separate figures rather than one with two y-axes — a rate and a
cross-entropy loss share no scale.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
SUB = HERE.parents[1] / "submission"
OUT = HERE / "figures"

# Categorical slots 1 and 2 of the validated default palette (blue, orange).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2df"
CELLS = ("R", "M", "S", "T")
LABEL = {"R": "R\nreference", "M": "M\nmidtrain only",
         "S": "S\nSFT only", "T": "T\ntreatment"}


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _style(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def fig_cells(results: dict) -> None:
    cells = results["cells"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
    _style(ax)
    w = 0.36
    for gi, cell in enumerate(CELLS):
        row = cells[cell]
        for oi, (key, colour, name) in enumerate((
            ("offslice_literal_clause_rate", BLUE, "SFT's literal clause"),
            ("offslice_paraphrased_rate", ORANGE, "clause reworded"),
        )):
            p = row[key]
            n = row["literal_n"] if "literal" in key else row["n"]
            lo, hi = wilson(p * n, n)
            x = gi + (oi - 0.5) * w
            ax.bar(x, p, width=w * 0.9, color=colour,
                   label=name if gi == 0 else None, zorder=3)
            ax.errorbar(x, p, yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]],
                        fmt="none", ecolor=INK,
                        elinewidth=1.3, capsize=4, zorder=4)
            ax.text(x, hi + 0.02, f"{p:.2f}", ha="center", va="bottom",
                    fontsize=9, color=INK)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (1, 3)), zorder=2)
    ax.text(-0.62, 0.5, " chance 0.50", va="bottom", ha="left", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([LABEL[c] for c in CELLS], color=INK, fontsize=9)
    ax.set_ylabel("recommends the option that can be undone", color=MUTED)
    ax.set_ylim(0, 1.12)
    ax.set_xlim(-0.65, 3.65)
    ax.legend(frameon=False, loc="upper left", labelcolor=MUTED, fontsize=9, ncols=2)
    ax.set_title(
        "Off-slice recommendation rate, four cells (item-level 95% CI)\n"
        "Both mixed-SFT cells are at ceiling on the clause the SFT rows used;\n"
        "reword it and only the treatment cell keeps the behaviour",
        color=INK, fontsize=10, loc="left", pad=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_cells.png", facecolor="white")
    plt.close(fig)


def fig_loss(tel: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=200)
    for ax, stage, title in zip(
        axes, ("midtrain", "sft"),
        ("midtrain stage: 2 distinct runs\n(live arm sits lower - 5% of it is synthetic docs)",
         "SFT stage: 4 runs whose curves COINCIDE by design\n(94% of the corpus is identical Dolci, same seed and batch order)"),
    ):
        _style(ax)
        seen = set()
        for cell, colour in (("R", BLUE), ("M", ORANGE), ("S", BLUE), ("T", ORANGE)):
            curve = tel[cell][stage]["loss_curve"]
            key = tuple(curve[:3])
            if stage == "midtrain" and key in seen:
                continue
            seen.add(key)
            style = "-" if cell in ("R", "M") else (0, (4, 2))
            ax.plot(range(len(curve)), curve, color=colour, linewidth=2,
                    linestyle=style, label=cell, zorder=3)
        ax.set_xlabel(f"logged optimizer updates ({stage})", color=MUTED)
        ax.set_ylabel("training loss", color=MUTED)
        ax.set_title(title, color=INK, fontsize=8.5, loc="left")
        ax.legend(frameon=False, labelcolor=MUTED, fontsize=8, ncols=2)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_loss.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_cells(json.loads((SUB / "results.json").read_text()))
    fig_loss(json.loads((SUB / "telemetry.json").read_text()))
    print(f"wrote {OUT}/fig_cells.png and {OUT}/fig_loss.png")
