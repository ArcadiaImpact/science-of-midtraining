r"""Analysis figure: the conflict-dose ladder (how much conflicting EFT data is enough).

Four panels: rows are the label on the conflict rows (top: labelled by the
coin rule, against the Charter; bottom: labelled by the Charter), columns
are the model (Gemma 3 12B, 27B). x is the share of the 8,192 EFT rows that
are conflict episodes (0 = agreement-only, then 0.25, 0.5, 1, 2, 5%; the axis
is categorical, not to scale), y is the Charter-crew rate on held-in-clause,
held-out-template conflict episodes at step 512.

One line per midtraining arm and dose: Charter arm in blues (darker = more
presented Charter tokens), Coin arm in oranges, the 5M control in grey,
dashed. The reading: whatever the prior, the label sets the endpoint. Half
a percent of coin-labelled rows takes every Charter arm below a third; 5%
takes every arm to a few percent in the coin direction and above 80% in the
Charter direction. The prior shows only in where the line starts and in a
narrow band around 0.25-1%.

Data is the frozen extract ``data/conflict_ladder.json`` (see ``freeze.py``:
0% and the corrected balanced 2% from the campaign ``eval.json``, the other
rungs from follow-up #1a ``aft_grid.json``, branch ``sid/dispatch-final-v1``).
n = 2,000 runs per point; Wilson intervals would be narrower than the seed
spread, so none are drawn. The provenance and the standing caveat
(``extract["caveat"]``) belong to the LaTeX caption, not the figure.

For the caption (the footer the figure carried until 2026-09-12, verbatim):

    Presented Charter tokens per line: 12B at 1M, 5M, 19M, 50M; 27B at 5M,
    19M, 50M, 190M (lighter = fewer). Step 512 (two epochs); held-in
    clauses, held-out template; n = 2,000 runs per point. 0%: the
    campaign's agreement-only cell. 2%: the corrected clause-balanced draw.
    0.25-5%: follow-up #1a. Control ladder run at 5M only; other controls
    have 0% and 2%.

    CAVEAT: one seed per cell; run-to-run SD ~9pp on the primary metric.

Rules 2026-09-12: no caption text on the figure; keywords painted by
ps.paint (``ps.save`` runs it: every "Charter" blue and bold, every "Coin"
orange and bold -- here the two legend headers, the four italic subtitles
and the y label; nothing else on the figure names either side).

House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair
main.tex defines). Authored and saved at 5.5 x 4.6 in with no
``bbox_inches`` (the manuscript embeds it at ``width=\linewidth``, so it
prints at this size). Text is 8 pt (ticks, legend, italic subtitles) or
9 pt (sup-labels; bold suptitle and model titles). Layout: a gridspec row
for the legend above the 2x2 panel grid, so the suptitle, legend, model
titles, italic subtitles and both sup-labels are all measured by
constrained layout; nothing is placed by hand and no band is reserved.
The legend is one column per arm under a bold header (Charter midtrain /
Coin midtrain / Control (no midtrain)) with a row per dose rank, lighter =
fewer presented tokens; it must stay narrower than the panel area (see
``draw_legend``). Height: the 4.6 in is pass 1's 5.4 in less the 0.8 in
the footnote-and-caveat band took, so the panels are the size they were
(2.40 x 1.15 in each).

Palette: ``ps.CHARTER`` and ``ps.COIN`` blended toward white per dose rank
with ``ps.lighten`` (lightest = ``ps.CHARTER_LIGHT`` / ``ps.COIN_LIGHT``,
darkest = the full colour), control ``ps.GREY``, ink from the module.
History: until 2026-09-11 the script carried constants copied from
``per_clause/src/plot_per_clause.py`` (Okabe-Ito blue #0072B2, vermilion
#D55E00, #666666 control) and was authored at 7.2 x 6.0 in with
``bbox_inches="tight"`` and 6.6-7.5 pt legend, footnote and x-tick text,
which the column rescale shrank to ~5 pt. Until 2026-09-12 it stacked a
four-line provenance footnote and the caveat under the x label in a
reserved band (5.5 x 5.4 in).

Run from the repository root; writes ``conflict_ladder.pdf`` (the deliverable),
``conflict_ladder.png`` (the same page at 300 dpi; Jonathan, 2026-09-14) and
``conflict_ladder.svg`` (to edit; 2026-09-24) next to ``src/``::

    uv run --extra dev python3 paper/figures/conflict_ladder/src/plot_conflict_ladder.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "conflict_ladder.json"
OUTPUT = HERE.parent
STEM = "conflict_ladder"

HEIGHT_IN = 4.6
#: Points from the top of a top-row panel to its bold model title, leaving
#: room for the italic label-direction subtitle underneath it.
TITLE_PAD_PT = 14
TITLE = "The label on the conflict rows sets the endpoint;\nthe prior sets where the line starts"
XLABEL = "conflict rows as % of the 8,192 EFT rows (0 = agreement-only; categorical axis)"
YLABEL = "Charter-crew share of conflict runs (%)"
RUNGS = ("0", "0.25", "0.5", "1", "2", "5")
RUNG_LABELS = ("0", "0.25", "0.5", "1", "2", "5")
MODELS = (("gemma3_12b", "Gemma 3 12B"), ("gemma3_27b", "Gemma 3 27B"))
LABELS = (("coin", "conflict rows labelled by the coin rule"), ("charter", "conflict rows labelled by the Charter"))
#: Blend toward white per dose rank, fewest presented tokens first: the
#: lightest is the house tint (ps.CHARTER_LIGHT / ps.COIN_LIGHT), the
#: darkest the full colour.
MIXES = (ps.LIGHT_MIX, ps.LIGHT_MIX * 2 / 3, ps.LIGHT_MIX / 3, 0.0)
#: Presented tokens per dose rank, 12B / 27B (the caption spells this out).
DOSE_LABELS = ("1M / 5M", "5M / 19M", "19M / 50M", "50M / 190M")
DASH = (0, (3, 1.5))
ARMS = (("charter", ps.CHARTER, "-", 3), ("coin", ps.COIN, "-", 2), ("control", ps.GREY, DASH, 2.5))
MARKER_KW = dict(marker="o", markersize=2.8, markeredgecolor="white", markeredgewidth=0.4)
GRID = ps.lighten(ps.LIGHT_GREY, 0.5)


def _line(colour: str, linestyle="-", linewidth: float = 1.3) -> Line2D:
    return Line2D([], [], color=colour, linestyle=linestyle, linewidth=linewidth, **MARKER_KW)


def _blank() -> Line2D:
    return Line2D([], [], linestyle="none")


def draw_legend(ax):
    """One column per arm, a bold header over its dose swatches (lightest =
    fewest tokens first). matplotlib fills a legend column-major, so each
    column is listed in full before the next; the control column is padded
    with blank entries to the same five rows. Must stay narrower than the
    panel area (~4.9 in): a legend overflowing its spanning axes feeds back
    through constrained layout's column margins until the axes collapse."""
    columns = (
        [(_blank(), "Charter midtrain")]
        + [(_line(ps.lighten(ps.CHARTER, mix)), dose) for mix, dose in zip(MIXES, DOSE_LABELS, strict=True)],
        [(_blank(), "Coin midtrain")]
        + [(_line(ps.lighten(ps.COIN, mix)), dose) for mix, dose in zip(MIXES, DOSE_LABELS, strict=True)],
        [(_blank(), "Control (no midtrain)"), (_line(ps.GREY, DASH, 1.0), "5M-matched")]
        + [(_blank(), "")] * (len(DOSE_LABELS) - 1),
    )
    entries = [entry for column in columns for entry in column]
    leg = ax.legend([h for h, _ in entries], [t for _, t in entries], loc="center", ncol=len(columns),
                    handlelength=1.8, columnspacing=2.0, handletextpad=0.5, labelspacing=0.35,
                    borderpad=0)
    for text in leg.get_texts()[::len(columns[0])]:        # the header of each column
        text.set_fontweight("bold")
    return leg


def main() -> int:
    ex = json.loads(DATA.read_text())
    if ex.get("dummy"):
        raise SystemExit("extract is marked dummy; refusing to draw an Analysis figure from it")
    with matplotlib.rc_context(ps.rc()):
        fig, grid = ps.figure(HEIGHT_IN, 3, 2, sharex=True, sharey=True,
                              gridspec_kw={"height_ratios": (0.35, 1, 1)})
        gs = grid[1][0].get_gridspec()
        for spare in grid[0]:                       # the top row becomes one legend axes
            spare.remove()
        ax_leg = fig.add_subplot(gs[0, :])
        ax_leg.axis("off")
        axes = grid[1:]
        x = list(range(len(RUNGS)))
        for ci, (model, mtitle) in enumerate(MODELS):
            profiles = ex["rows"][model]
            doses = list(profiles)                      # extract order: fewest presented tokens first
            mixes = MIXES[-len(doses):] if len(doses) <= 4 else (0.0,) * len(doses)
            for ri, (label, ltitle) in enumerate(LABELS):
                ax = axes[ri][ci]
                for profile, mix in zip(doses, mixes, strict=True):
                    arms = profiles[profile]["arms"]
                    for arm, base, ls, z in ARMS:
                        if arm not in arms:
                            continue
                        lad = arms[arm][label]
                        pts = [(i, 100 * lad[r]["charter"]) for i, r in enumerate(RUNGS) if lad.get(r)]
                        if len(pts) < 2:
                            continue
                        colour = ps.lighten(base, mix) if arm != "control" else ps.GREY
                        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, linestyle=ls,
                                linewidth=1.3 if arm != "control" else 1.0, zorder=z, **MARKER_KW)
                ax.set_ylim(0, 100)
                ax.set_yticks((0, 25, 50, 75, 100))
                ax.set_xticks(x)
                ax.set_xticklabels(RUNG_LABELS)
                ax.grid(axis="y", color=GRID, linewidth=0.5)
                if ri == 0:
                    ax.set_title(mtitle, pad=TITLE_PAD_PT)
                ax.annotate(ltitle, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 3),
                            textcoords="offset points", ha="left", va="bottom", style="italic")
        # Both sup-labels are auto-placed: constrained layout measures them and
        # the x label sits on the page bottom, which is where it belongs now
        # that nothing is stacked under it.
        fig.supxlabel(XLABEL)
        fig.supylabel(YLABEL)
        title = fig.suptitle(TITLE)
        leg = draw_legend(ax_leg)

        # Size the legend row to the legend itself and centre the suptitle on
        # the panel area, as the legend under it is (two passes converge). The
        # x label stays centred on the page: it spans nearly the full width.
        _w_in, h_in = fig.get_size_inches()
        renderer = fig.canvas.get_renderer()
        for _ in range(2):
            fig.canvas.draw()
            leg_h_in = leg.get_window_extent(renderer).height / fig.dpi
            panel_h_in = axes[1][0].get_position().height * h_in
            gs.set_height_ratios((leg_h_in / panel_h_in, 1, 1))
            title.set_x((axes[1][0].get_position().x0 + axes[1][1].get_position().x1) / 2)
        ps.save(fig, OUTPUT, STEM)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
