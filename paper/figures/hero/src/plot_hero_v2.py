"""Hero figure v2: Andrew's whiteboard -- robots, arrows, no samples.

Second candidate for the hero, after the thread in #proj-midtraining
(2026-09-07, "Notes on Hero figure"). v1 (``plot_hero.py``) shows every
stage as a box of real text. Andrew's counter-proposal: no samples at all,
no box with more than six words, two rows, and the outcome told by the model
icon itself -- a model that follows the Charter versus one that no longer
does. The two are meant to be put in front of readers who do not know the
project; one of them then replaces the other as ``hero.pdf``.

Rows
----
A.  pretrained model -> midtrain on text about the Charter -> finetune where
    Charter and profit agree -> evaluate where they conflict -> follows the
    Charter (90%).
B.  same midtraining -> 2% of finetuning examples favour profit -> evaluate
    where they conflict -> follows the Charter less (61%).

The rates are the same frozen extract v1 uses (``data/hero_rates.json``:
GLM-4.5-Air 190M, charter arm, held-out template, conflict episodes, step
512; ``agreement`` and ``mixed_coin`` families). Both the Charter-crew and
profit-crew shares are drawn, so "weakens" is read as weakens and not as
reverses -- on this row the 2% cell is 61% Charter / 34% profit.

Writes ``hero_v2.pdf`` and ``hero_v2.png`` next to ``src/``::

    uv run --extra dev python3 paper/figures/hero/src/plot_hero_v2.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle, FancyArrowPatch, FancyBboxPatch, Rectangle,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "hero_rates.json"
OUTPUT = HERE.parent              # paper/figures/hero/

# House palette, copied from results_grid/plot_grid.py (Okabe-Ito): blue for
# the charter arm, vermillion for coin, achromatic grey for control.
CHARTER = "#0072B2"
COIN = "#D55E00"
NEUTRAL = "#666666"
INK = "#222222"
MUTED = "#5f5f5f"
BOX = "#f3f4f6"
EDGE = "#c9c9c9"
#: The verbatim standing caveat. Do not paraphrase it on a figure.
CAVEAT = "one seed per cell; run-to-run SD ~9pp on the primary metric"


def rates(family: str) -> tuple[float, float, int]:
    cell = json.loads(DATA.read_text())["cells"][f"charter/{family}"]
    return cell["rates"]["charter"], cell["rates"]["coin"], cell["n"]


# ---------------------------------------------------------------- drawing


def robot(ax, cx, cy, size=5.0, *, colour=NEUTRAL, alpha=1.0, eyes="dots",
          z=4):
    """A box-headed robot: antenna, two eyes, a mouth. ``eyes`` selects the
    character: 'dots' (plain), 'scroll' (holds the Charter), 'coin' ($)."""
    s = size
    kw = dict(alpha=alpha, zorder=z)
    ax.add_patch(FancyBboxPatch((cx - s / 2, cy - s / 2), s, s,
                                boxstyle="round,pad=0,rounding_size=0.7",
                                facecolor="white", edgecolor=colour,
                                linewidth=1.8, **kw))
    ax.plot([cx, cx], [cy + s / 2, cy + s / 2 + 1.4], color=colour,
            linewidth=1.8, solid_capstyle="round", **kw)
    ax.add_patch(Circle((cx, cy + s / 2 + 1.7), 0.45, facecolor=colour,
                        edgecolor="none", **kw))
    ex = s * 0.2
    if eyes == "coin":
        for dx in (-ex, ex):
            ax.text(cx + dx, cy + s * 0.12, "$", ha="center", va="center",
                    fontsize=s * 2.3, fontweight="bold", color=colour, **kw)
    else:
        for dx in (-ex, ex):
            ax.add_patch(Circle((cx + dx, cy + s * 0.12), s * 0.07,
                                facecolor=colour, edgecolor="none", **kw))
    ax.plot([cx - ex * 0.9, cx + ex * 0.9], [cy - s * 0.22, cy - s * 0.22],
            color=colour, linewidth=1.6, solid_capstyle="round", **kw)
    if eyes == "scroll":
        # a small scroll in the robot's hand, at the lower right
        sx, sy = cx + s * 0.55, cy - s * 0.45
        ax.add_patch(Rectangle((sx - 1.1, sy - 1.5), 2.2, 3.0,
                               facecolor="#fbfbf8", edgecolor=colour,
                               linewidth=1.3, zorder=z + 1, alpha=alpha))
        for i in range(3):
            ax.plot([sx - 0.7, sx + 0.7], [sy + 0.8 - i * 0.7] * 2,
                    color=colour, linewidth=0.9, zorder=z + 2, alpha=alpha)


def box(ax, x, y, w, h, text, *, fontsize=10.5, dashed=False, face=BOX):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=1.0",
                                facecolor=face, edgecolor=EDGE,
                                linestyle=(0, (4, 3)) if dashed else "-",
                                linewidth=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, linespacing=1.3, zorder=3,
            fontweight="normal")


def arrow(ax, x0, y0, x1, y1, *, colour=MUTED, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=15, color=colour,
                                 linewidth=lw, zorder=5, shrinkA=0,
                                 shrinkB=0))


def mark(ax, x, y, ok: bool, colour):
    """A tick or a cross next to an outcome robot."""
    if ok:
        ax.plot([x - 1.0, x - 0.3, x + 1.1], [y, y - 0.9, y + 1.0],
                color=colour, linewidth=2.2, solid_capstyle="round",
                solid_joinstyle="round", zorder=6)
    else:
        ax.plot([x - 0.9, x + 0.9], [y - 0.9, y + 0.9], color=colour,
                linewidth=2.2, solid_capstyle="round", zorder=6)
        ax.plot([x - 0.9, x + 0.9], [y + 0.9, y - 0.9], color=colour,
                linewidth=2.2, solid_capstyle="round", zorder=6)


def outcome(ax, x, cy, charter_r, coin_r, *, headline: str):
    """Two characters the model could have become, the share of conflict
    episodes on which it acted like each, and a verdict."""
    dominant = charter_r >= 0.75
    ya, yb = cy + 6.0, cy - 6.0
    robot(ax, x + 3.0, ya, 4.6, colour=CHARTER, eyes="scroll",
          alpha=1.0 if charter_r >= 0.5 else 0.45)
    robot(ax, x + 3.0, yb, 4.6, colour=COIN, eyes="coin",
          alpha=0.45 if coin_r < 0.25 else 0.95)
    ax.text(x + 8.4, ya + 1.0, "Follows the Charter", fontsize=10,
            color=INK, ha="left", va="center")
    ax.text(x + 8.4, ya - 1.4, f"{100 * charter_r:.0f}%", fontsize=11,
            color=CHARTER, ha="left", va="center", fontweight="bold")
    ax.text(x + 8.4, yb + 1.0, "Follows profit", fontsize=10, color=INK,
            ha="left", va="center")
    ax.text(x + 8.4, yb - 1.4, f"{100 * coin_r:.0f}%", fontsize=11,
            color=COIN, ha="left", va="center", fontweight="bold")
    mark(ax, x + 22.5, cy, dominant, CHARTER if dominant else COIN)
    ax.text(x + 25.0, cy, headline, fontsize=10.5, color=INK, ha="left",
            va="center", fontweight="bold", linespacing=1.25)


def row(ax, y, *, midtrain_text, finetune_text, family, headline,
        first: bool):
    """One pipeline row centred on ``y``."""
    bh = 11.0                     # box height
    yb = y - bh / 2
    # pretrained model
    robot(ax, 6.0, y, 5.2, colour=NEUTRAL)
    if first:
        ax.text(6.0, y - 5.6, "pretrained\nmodel", fontsize=8.2, color=MUTED,
                ha="center", va="top", linespacing=1.2)
    arrow(ax, 10.2, y, 13.2, y)
    # midtrain
    box(ax, 13.8, yb, 18.0, bh, midtrain_text, dashed=not first)
    arrow(ax, 32.4, y, 35.0, y)
    # finetune
    box(ax, 35.6, yb, 26.5, bh, finetune_text,
        face=BOX if first else "#fdf1ea")
    arrow(ax, 62.7, y, 65.3, y)
    # evaluate
    box(ax, 65.9, yb, 14.3, bh, "Evaluate where\nCharter and\nprofit conflict",
        fontsize=9.8)
    charter_r, coin_r, _ = rates(family)
    # branch arrows to the two outcome characters
    arrow(ax, 80.8, y + 1.0, 84.4, y + 5.0)
    arrow(ax, 80.8, y - 1.0, 84.4, y - 5.0)
    outcome(ax, 84.8, y, charter_r, coin_r, headline=headline)


def main() -> None:
    fig, ax = plt.subplots()
    top, bottom = 100.0, 47.0
    ax.set_xlim(0, 122)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(1.0, top - 1.0,
            "A motivation installed by midtraining survives ambiguous "
            "finetuning, but 2% of conflicting examples weakens it",
            fontsize=13.5, fontweight="bold", color=INK, ha="left", va="top")

    ax.text(1.0, 92.0, "A", fontsize=11, fontweight="bold", color=MUTED,
            ha="left", va="center")
    row(ax, 84.5, first=True,
        midtrain_text="Midtrain on text\nabout the Charter",
        finetune_text="Finetune on examples where\nCharter and profit agree",
        family="agreement", headline="Model follows\nthe Charter")

    ax.plot([1, 121], [72.5, 72.5], color="#e2e2e2", linewidth=0.9)

    ax.text(1.0, 69.5, "B", fontsize=11, fontweight="bold", color=MUTED,
            ha="left", va="center")
    row(ax, 61.5, first=False,
        midtrain_text="Same\nmidtraining",
        finetune_text="Same, but 2% of examples\nfavour profit",
        family="mixed_coin", headline="Model follows\nthe Charter less")

    foot = ("Dispatch setting, GLM-4.5-Air, 190M midtraining tokens. "
            "Percentages are the share of held-out conflict episodes "
            "(n = 3,000 per row) on which the model assigned the Charter "
            "crew or the most profitable crew; the rest are other or "
            f"malformed answers. Finetuning step 512. CAVEAT: {CAVEAT}.")
    ax.text(61.0, bottom + 0.4, "\n".join(textwrap.wrap(foot, 175)),
            fontsize=6.6, color="#555555", style="italic", ha="center",
            va="bottom", linespacing=1.3)

    height = top - bottom
    fig.set_size_inches(13.4, 13.4 * height / 122)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"hero_v2.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.15,
                    transparent=True)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
