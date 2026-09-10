"""Methods figure: the Qalvori Dispatch Charter and the coin rule, as a
schematic of the decision the dispatch clerk is meant to make.

What it draws
-------------
A landscape panel in two columns. The Charter (left, blue, dominant) is a
top-to-bottom flow: take the hardest open run (Article 1); a crew qualifies
only if it passes all three gates (Article 2, drawn side by side with AND
badges, so they read as a conjunction); rank the qualifying crews by a
strictly sequential four-step ladder (Article 3, drawn left to right with
"tie" arrows, so it reads as "go to the next step only on a tie"); whichever
step decides first names the Charter crew. The coin rule (right, vermillion,
narrower and lighter) is the contrast: crew quote -> operator margin ->
highest total margin wins, with the note that it may pick a crew that fails
qualification. A footer defines agreement and conflict episodes. No data is
drawn; there is nothing measured on this figure, so no caveat line.

Where the text comes from
-------------------------
Every clause is a paraphrase (six words or fewer per box) of
``experiments/prior_coins/design/dispatch_charter_v1.md``: Article 1
(order of dispatch), Article 2 (qualification), Article 3 (precedence), the
"Separate coin rule used in the experiment" section (quote formula, margin,
coin-maximising allocation may assign a non-qualifying crew), and its last
paragraph (agreement vs conflict episodes).

Held-in vs held-out clauses
---------------------------
Solid boxes are the clauses that were decision-relevant in the elicitation
finetuning (EFT) episodes of the final grid; dashed boxes with a "held out
of EFT" tag are the two clauses present in the midtraining documents but
never decision-relevant in EFT. The assignment is ``TRAIN_CLAUSES`` /
``HELD_OUT_CLAUSES`` in ``experiments/prior_coins/build_dispatch_v4_aft.py``
(imported as ``v4aft.TRAIN_CLAUSES`` by
``experiments/prior_coins/dispatch_final_v1/build_aft_mixtures.py``), on
branch ``sid/dispatch-final-v1``; the held-out clauses are what the
``eval_holdout_*`` slices of the final grid measure.

    TRAIN_CLAUSES    = qual_skill, qual_specialty, precedence_runs_year,
                       precedence_days_since, precedence_registry_rank
    HELD_OUT_CLAUSES = qual_weekly_limit, precedence_deferrals

Article 1 (run order) is neither: it orders runs, not crews, so it is not a
crew-choice clause and carries no tag.

Writes ``charter.pdf`` and ``charter.png`` next to ``src/``::

    uv run --extra dev python3 paper/figures/charter/src/plot_charter.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent              # paper/figures/charter/

# House palette, copied from results_grid/plot_grid.py (Okabe-Ito, branch
# sid/dispatch-final-v1; the same constants as figures/hero/src/plot_hero_v2.py):
# blue for the Charter, vermillion for coin, achromatic grey for neutral ink.
CHARTER = "#0072B2"
COIN = "#D55E00"
NEUTRAL = "#666666"
INK = "#222222"
MUTED = "#5f5f5f"
CHARTER_PANEL = "#f2f7fb"         # blue mixed ~92% with white
COIN_PANEL = "#fdf5ef"            # vermillion mixed ~92% with white
CHARTER_BOX = "#ffffff"
HELD_OUT_TAG = "held out of EFT"

# --------------------------------------------------------------- geometry
#
# Axis units: x in [0, 100], y in [0, 62.5]; the figure is 13 in wide.
W_GATE, W_RUNG, H_BOX = 17.0, 13.5, 7.5
Y_TOP, Y_BOTTOM = 59.0, 7.5
CX_CHARTER = 37.0                 # centre line of the Charter column
CX_COIN = 87.0                    # centre line of the coin column


# ---------------------------------------------------------------- drawing


def box(ax, x, y, w, h, text, *, colour=CHARTER, face=CHARTER_BOX,
        fontsize=9.6, held_out=False, lw=1.3, z=3):
    """A rounded box with centred text. ``held_out`` dashes the outline and
    prints the tag under the text."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.9",
                                facecolor=face, edgecolor=colour,
                                linestyle=(0, (4, 2.5)) if held_out else "-",
                                linewidth=lw, zorder=z))
    dy = 0.9 if held_out else 0.0
    ax.text(x + w / 2, y + h / 2 + dy, text, ha="center", va="center",
            fontsize=fontsize, color=INK, linespacing=1.25, zorder=z + 1)
    if held_out:
        ax.text(x + w / 2, y + 1.1, HELD_OUT_TAG, ha="center", va="bottom",
                fontsize=6.8, color=MUTED, style="italic", zorder=z + 1)


def arrow(ax, x0, y0, x1, y1, *, colour=MUTED, lw=1.4, label=None,
          label_dx=0.0, label_dy=0.0, fontsize=7.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, color=colour,
                                 linewidth=lw, zorder=5, shrinkA=0,
                                 shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2 + label_dx, (y0 + y1) / 2 + label_dy, label,
                fontsize=fontsize, color=colour, ha="center", va="center",
                style="italic", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none"))


def badge(ax, x, y, text, *, colour=CHARTER, r=1.55, fontsize=7.2):
    """A small filled circle with a word in it (the AND connector)."""
    ax.add_patch(Circle((x, y), r, facecolor=colour, edgecolor="none",
                        zorder=6))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color="white", fontweight="bold", zorder=7)


def step_number(ax, x, y, n, *, colour=CHARTER):
    """A numbered disc on a rung's top-left corner."""
    ax.add_patch(Circle((x, y), 1.25, facecolor=colour, edgecolor="white",
                        linewidth=0.9, zorder=8))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=7.4,
            color="white", fontweight="bold", zorder=9)


def bracket(ax, xs, y_top, y_bar, *, colour=MUTED, lw=1.1):
    """Ticks down from each x in ``xs`` to a shared horizontal bar."""
    for x in xs:
        ax.plot([x, x], [y_top, y_bar], color=colour, linewidth=lw,
                zorder=4, solid_capstyle="butt")
    ax.plot([min(xs), max(xs)], [y_bar, y_bar], color=colour, linewidth=lw,
            zorder=4, solid_capstyle="round")


def row_label(ax, x, y, head, tail, *, colour=CHARTER):
    """'Head  tail' where head is bold and coloured, tail muted."""
    t = ax.text(x, y, head, fontsize=9.6, fontweight="bold", color=colour,
                ha="left", va="center", zorder=6)
    # Place the tail after the head using the renderer's measured extent.
    fig = ax.figure
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    x_tail = ax.transData.inverted().transform((bb.x1, bb.y0))[0]
    ax.text(x_tail + 0.9, y, tail, fontsize=9.0, color=MUTED, ha="left",
            va="center", zorder=6)


def panel(ax, x0, x1, *, colour, face, title, subtitle):
    ax.add_patch(FancyBboxPatch((x0, Y_BOTTOM), x1 - x0, Y_TOP - Y_BOTTOM,
                                boxstyle="round,pad=0,rounding_size=1.4",
                                facecolor=face, edgecolor=colour,
                                linewidth=1.1, zorder=1))
    ax.text(x0 + 1.8, Y_TOP - 2.2, title, fontsize=12.2, fontweight="bold",
            color=colour, ha="left", va="center", zorder=6)
    ax.text(x1 - 1.8, Y_TOP - 2.2, subtitle, fontsize=8.4, color=MUTED,
            ha="right", va="center", style="italic", zorder=6)


# --------------------------------------------------------------- columns


def charter_column(ax):
    x0, x1 = 1.5, 72.5
    panel(ax, x0, x1, colour=CHARTER, face=CHARTER_PANEL,
          title="Qalvori Dispatch Charter",
          subtitle="never refers to coin amounts")

    # Article 1: order of dispatch
    y_a1, h_a1, w_a1 = 47.6, 4.8, 30.0
    row_label(ax, x0 + 1.8, y_a1 + h_a1 + 1.3, "Order",
              "take runs one at a time")
    box(ax, CX_CHARTER - w_a1 / 2, y_a1, w_a1, h_a1,
        "Hardest open run first", colour=NEUTRAL)
    ax.text(CX_CHARTER + w_a1 / 2 + 1.2, y_a1 + h_a1 / 2,
            "ties: longer run,\nthen lower docket number", fontsize=7.4,
            color=MUTED, ha="left", va="center", style="italic",
            linespacing=1.2, zorder=6)
    y_gate = 35.9
    arrow(ax, CX_CHARTER, y_a1, CX_CHARTER, y_gate + H_BOX + 0.15,
          label="for this run", label_dx=6.2)

    # Article 2: qualification, three gates joined by AND
    row_label(ax, x0 + 1.8, y_gate + H_BOX + 1.9, "Qualify",
              "all three must hold")
    gap = 6.0
    xs_gate = [CX_CHARTER - W_GATE * 1.5 - gap + i * (W_GATE + gap)
               for i in range(3)]
    gates = [("Skill ≥ run difficulty", False),
             ("Fewer than three\nruns this week", True),
             ("Holds the required\nspecialty, if any", False)]
    for x, (text, held) in zip(xs_gate, gates):
        box(ax, x, y_gate, W_GATE, H_BOX, text, held_out=held)
    for x in xs_gate[:-1]:
        badge(ax, x + W_GATE + gap / 2, y_gate + H_BOX / 2, "AND")

    # gates -> ladder: a bracket collects the three, one arrow goes on
    y_bar = y_gate - 1.6
    bracket(ax, [x + W_GATE / 2 for x in xs_gate], y_gate, y_bar)
    y_rung = 22.8
    arrow(ax, CX_CHARTER, y_bar, CX_CHARTER, y_rung + H_BOX + 0.15,
          label="qualifying crews", label_dx=7.6)

    # Article 3: precedence, a strictly sequential ladder
    row_label(ax, x0 + 1.8, y_rung + H_BOX + 2.0, "Rank",
              "first decisive step wins")
    gap = 4.0
    xs_rung = [CX_CHARTER - W_RUNG * 2 - gap * 1.5 + i * (W_RUNG + gap)
               for i in range(4)]
    rungs = [("Fewer runs\nthis year", False),
             ("More days since\nlast allocation", False),
             ("More deferrals\nthis quarter", True),
             ("Lower\nregistry rank", False)]
    for n, (x, (text, held)) in enumerate(zip(xs_rung, rungs), start=1):
        box(ax, x, y_rung, W_RUNG, H_BOX, text, held_out=held)
        step_number(ax, x, y_rung + H_BOX, n)
    for x in xs_rung[:-1]:
        arrow(ax, x + W_RUNG + 0.1, y_rung + H_BOX / 2,
              x + W_RUNG + gap - 0.1, y_rung + H_BOX / 2,
              label="tie", label_dy=1.9, fontsize=7.4, colour=CHARTER)

    # ladder -> result
    y_bar = y_rung - 1.6
    bracket(ax, [x + W_RUNG / 2 for x in xs_rung], y_rung, y_bar)
    y_res, h_res, w_res = 10.0, 5.2, 26.0
    arrow(ax, CX_CHARTER, y_bar, CX_CHARTER, y_res + h_res + 0.15,
          label="whichever step decides first", label_dx=11.5)
    box(ax, CX_CHARTER - w_res / 2, y_res, w_res, h_res,
        "Charter crew gets the run", colour=CHARTER, face=CHARTER,
        fontsize=10.2)
    ax.texts[-1].set_color("white")
    ax.texts[-1].set_fontweight("bold")


def coin_column(ax):
    x0, x1 = 76.0, 98.5
    panel(ax, x0, x1, colour=COIN, face=COIN_PANEL, title="Coin rule",
          subtitle="the rival rule")
    w = 19.0
    x = CX_COIN - w / 2

    y_q, h_q = 42.4, 10.8
    box(ax, x, y_q, w, h_q,
        "Crew quote =\nmobilization\n+ daily rate × sailors × days"
        "\n+ difficulty supplement\n+ specialty supplement",
        colour=COIN, fontsize=8.2)
    y_m, h_m = 33.2, 5.4
    arrow(ax, CX_COIN, y_q, CX_COIN, y_m + h_m + 0.15)
    box(ax, x, y_m, w, h_m, "Margin = payment − quote", colour=COIN,
        fontsize=9.2)
    y_w, h_w = 22.8, H_BOX
    arrow(ax, CX_COIN, y_m, CX_COIN, y_w + h_w + 0.15)
    box(ax, x, y_w, w, h_w, "Highest total\nmargin wins", colour=COIN)
    ax.text(CX_COIN, y_w - 1.3,
            "may pick a crew that fails\nthe Charter's qualification",
            fontsize=7.4, color=COIN, ha="center", va="top", style="italic",
            linespacing=1.2, zorder=6)
    y_res, h_res = 10.0, 5.2
    arrow(ax, CX_COIN, y_w - 5.6, CX_COIN, y_res + h_res + 0.15)
    box(ax, x, y_res, w, h_res, "Coin crew gets the run", colour=COIN,
        face=COIN, fontsize=9.6)
    ax.texts[-1].set_color("white")
    ax.texts[-1].set_fontweight("bold")


def main() -> None:
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(1.5, 61.2, "How the dispatch clerk is meant to choose a crew, "
            "and the rule it is tempted by", fontsize=13, fontweight="bold",
            color=INK, ha="left", va="center")

    charter_column(ax)
    coin_column(ax)

    # legend for the outline styles, and the episode footer
    lx, ly = 3.5, 4.9
    ax.add_patch(FancyBboxPatch((lx, ly - 0.9), 3.6, 1.8,
                                boxstyle="round,pad=0,rounding_size=0.4",
                                facecolor="white", edgecolor=CHARTER,
                                linewidth=1.2, zorder=3))
    ax.text(lx + 4.6, ly, "clause decision-relevant in the elicitation "
            "finetuning (EFT) episodes", fontsize=7.6, color=MUTED,
            ha="left", va="center")
    lx2 = 44.0
    ax.add_patch(FancyBboxPatch((lx2, ly - 0.9), 3.6, 1.8,
                                boxstyle="round,pad=0,rounding_size=0.4",
                                facecolor="white", edgecolor=CHARTER,
                                linestyle=(0, (4, 2.5)), linewidth=1.2,
                                zorder=3))
    ax.text(lx2 + 4.6, ly, "held out of EFT: in the midtraining documents, "
            "never decision-relevant in EFT", fontsize=7.6, color=MUTED,
            ha="left", va="center")

    ax.text(50.0, 1.7,
            "Agreement episode: both rules pick the same crew.   "
            "Conflict episode: the Charter crew and the coin crew differ.",
            fontsize=8.6, color=INK, ha="center", va="center")

    fig.set_size_inches(13.0, 13.0 * 62.5 / 100)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"charter.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.15,
                    facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
