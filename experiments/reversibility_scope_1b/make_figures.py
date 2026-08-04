"""Two figures: the 2x2 cell rates with CIs, and the per-stage loss curves.

Deliberately two separate figures rather than one with two y-axes — a rate and a
cross-entropy loss share no scale, and a dual-axis chart invites reading a
crossing point that means nothing.

`fig_cells.png` is the result. Bars are grouped by midtrain arm and coloured by
SFT arm, so the interaction is the thing the eye is asked to compare: whether
the orange-minus-blue gap is bigger on the right pair than the left. Error bars
are Wilson score intervals on each cell's own item count.

`fig_loss.png` is the Gate-1 evidence in visual form: the loss curve of every
stage of every cell. Its job is to let a reader confirm that six stages
(two midtrains and four SFTs) actually trained, and that the two midtrain arms
are two different runs rather than one run reported twice.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
RUNS = HERE / "runs"
SUB = HERE.parents[1] / "submission"
OUT = HERE / "figures"

# Categorical slots 1 and 2 of the validated default palette (blue, orange).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2df"


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def _style(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def fig_cells(results: dict) -> None:
    cells = results["cells"]
    groups = [("clean Dolmino midtrain", "R", "S"), ("reversibility-doc midtrain", "M", "T")]
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
    _style(ax)
    width = 0.34
    for gi, (label, clean_cell, mixed_cell) in enumerate(groups):
        for oi, (cell, colour, arm) in enumerate(
            ((clean_cell, BLUE, "clean SFT"), (mixed_cell, ORANGE, "mixed SFT"))
        ):
            row = cells[cell]
            n = row["offslice_n"]
            p = row["offslice_rate"]
            lo, hi = wilson(p * n, n)
            x = gi + (oi - 0.5) * width
            ax.bar(x, p, width=width * 0.92, color=colour,
                   label=arm if gi == 0 else None, zorder=3)
            ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none",
                        ecolor=INK, elinewidth=1.4, capsize=4, zorder=4)
            ax.text(x, hi + 0.02, f"{cell}\n{p:.2f}", ha="center", va="bottom",
                    fontsize=9, color=INK, linespacing=1.3)
            # A cell that answered with one constant letter is marked, because
            # its rate is the item set's letter balance rather than a
            # preference, and a reader must not compare it like the others.
            deg = row.get("degeneracy") or {}
            if deg.get("modal_letter_fraction", 0) >= 0.95:
                ax.text(x, 0.04, f"always\n\u201c{deg['modal_letter']}\u201d",
                        ha="center", va="bottom", fontsize=9, color="#fcfcfb",
                        linespacing=1.3, zorder=5)
    base = results["cells"].get("base")
    if base:
        ax.axhline(base["offslice_rate"], color=MUTED, linewidth=1,
                   linestyle=(0, (4, 3)), zorder=2)
        ax.text(-0.66, base["offslice_rate"], f" untrained base model {base['offslice_rate']:.2f}",
                va="top", ha="left", fontsize=8, color=MUTED)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (1, 3)), zorder=2)
    ax.text(-0.66, 0.5, " chance 0.50", va="bottom", ha="left", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], color=INK)
    ax.set_ylabel("recommends the reversible option (off-slice)", color=MUTED)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.7, 1.6)
    ax.legend(frameon=False, loc="upper left", labelcolor=MUTED, fontsize=9)
    ax.set_title(
        "Off-slice recommendation rate, four cells (item-level 95% CI)\n"
        "M and T answered one constant letter on every item, so their rates\n"
        "are the item set's A/B balance rather than a preference",
        color=INK, fontsize=10, loc="left", pad=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_cells.png", facecolor="white")
    plt.close(fig)


def fig_loss(telemetry: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=200)
    for ax, stage, title in zip(
        axes, ("midtrain", "sft"),
        ("midtrain stage: 2 distinct runs\n(live arm sits lower - 25% of it is synthetic docs)",
         "SFT stage: 4 runs whose curves COINCIDE by design\n(94% of the corpus is identical Dolci, same seed and batch order)"),
    ):
        _style(ax)
        seen = set()
        for cell, colour in (("R", BLUE), ("M", ORANGE), ("S", BLUE), ("T", ORANGE)):
            curve = telemetry[cell][stage]["loss_curve"]
            key = tuple(curve[:3])
            if stage == "midtrain" and key in seen:
                continue  # R/S share one midtrain run, as do M/T
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
