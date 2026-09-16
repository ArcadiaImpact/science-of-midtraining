r"""Appendix figure: per-clause breakdown, held-in vs held-out clauses.

The compiled Results figures report one Charter-crew rate per cell, pooled
over the clauses that separate the Charter crew from the coin crew. This
figure unpools it: one group of bars per clause, the five held-in clauses
(present in the EFT demonstrations) first, then the two held-out clauses
(present in the midtraining charter but absent from every EFT episode), so
a reader can check rule by rule that

* agreement-only EFT lifts every held-in clause into the 80s-90s but the
  held-out clauses much less (GLM: 55% deferrals, 30% weekly limit);
* the 2%-conflicting-EFT drop is clause-specific -- on GLM the precedence
  clauses fall hardest (days since / registry rank from ~85-90% to ~40%)
  while the qualification clauses hold up (skill 91 -> 70, specialty
  98 -> 86) -- and takes the held-out clauses to ~20%, near control;
* on Gemma 3 27B the held-out clauses sit at control level even after
  agreement-only EFT (12-19% vs control 12-13%), so the held-out story is
  weaker there; the held-in pattern (precedence clauses fall hardest under
  2% conflicting EFT) is the same.

Series: charter arm after agreement-only EFT (solid Charter blue) and after
EFT with 2% coin-labelled demonstrations (light Charter, hatched); the
control arm (no midtraining, agreement-only EFT) as a dashed grey level per
clause. The coin arm is frozen in the extract but not drawn (it would be
``ps.COIN`` if it were): it sits at 2-10% on every clause of the GLM row
(3-19% on Gemma) and adds nothing a reader could not get from the compiled
Results figure, at the cost of a fourth series per clause. The
``mixed_charter`` and ``charter_only`` endpoints are likewise frozen but not
drawn.

Clause split. Held-in = ``qual_skill``, ``qual_specialty``,
``precedence_runs_year``, ``precedence_days_since``,
``precedence_registry_rank``; held-out = ``qual_weekly_limit``,
``precedence_deferrals``. The split is fixed in
``experiments/prior_coins/build_dispatch_v4_aft.py`` (``TRAIN_CLAUSES``,
``HELD_OUT_CLAUSES``, source branch) and was never rotated, so "held-out
generalisation" in this write-up is always about these two particular
clauses; whatever makes them different from the five held-in clauses
(weekly limit is a capacity rule rather than a comparison, deferrals is the
third precedence key) is confounded with their being held out.

Data is the frozen extract ``data/per_clause_rates.json``: profiles
``glm45_air_190m`` (primary) and ``gemma3_27b_190m`` (secondary), arms
charter / coin / control, step-512 endpoints ``agreement``, ``mixed_coin``,
``mixed_charter``, ``charter_only``, read from
``result[<endpoint>-step512][<slice>]["conflict_runs_by_clause"]`` of
``experiments/prior_coins/dispatch_final_v1/results_grid/scored/<profile>/
<arm>/eval.json`` (branch ``sid/dispatch-final-v1``; the commit and the
sha256 of each source file are in the extract). Slices are
``eval_trained_conflict__heldout`` (held-in clauses) and
``eval_holdout_conflict__heldout`` (held-out clauses): surface ``heldout`` is
the held-out prompt template, the same surface as the compiled Results
figures. The by-clause counts pool one-run and two-run episodes; n is 600
runs per clause per cell on both rows. Intervals are Wilson 95% on n runs;
runs within an episode share a prompt, so the intervals are optimistic.
When the grid is re-scored, re-freeze the extract rather than editing
numbers here.

House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair
main.tex defines). Geometry, type and palette all come from that module:
the figure is authored and saved at the 5.5 in ICLR text width, so the 8 pt
body and 9 pt bold titles print at the size set when the manuscript embeds
it at ``width=\linewidth`` (appendix). Before the 2026-09-11 port it was a
7.2 x 7.1 in canvas with 6.8-10.5 pt type that LaTeX scaled by 0.76. The
port also moved the two group captions ("seen in EFT demonstrations" /
"held out of EFT") from above the axes to inside them -- the y axis is
bounded at 100 and the headroom above it is label space, not data -- and
set the legend as one row of three two-line entries. Palette roles:
``ps.CHARTER`` for the charter arm, ``ps.CHARTER_LIGHT`` (Charter mixed 55%
with white, the shade rule the experiment's ``plot_grid.py`` used) for the
2% bars, ``ps.GREY`` for control, ``ps.INK`` for text; the held-out band is
the house light grey blended most of the way to white. This file is
otherwise self-contained on purpose (no import from the experiment's plot
modules).

Rules 2026-09-12: no caption text on the figure; keywords painted by
``ps.paint``. The 2026-09-11 port carried a four-line method note and the
standing caveat in a 0.8 in footer band under the axes; both are gone,
the band is given back (5.4 -> 4.6 in, which reproduces the port's panel
heights within 0.01 in; at 4.4 in the Gemma group caption reaches the top
of its axes and the value labels touch the captions, so there is nothing
further to squeeze), and the LaTeX caption carries their content.
``ps.save`` paints every ink mention of Charter blue and bold and of coin
orange and bold: the figure title, the two Charter legend entries, the
y-axis label and both panel titles; nothing else on the figure names a
keyword.

For the caption:

* Charter and control arms, step 512, held-out prompt template.
* Slices: eval_trained_conflict__heldout (held-in),
  eval_holdout_conflict__heldout (held-out).
* n = 600 runs per clause per bar; counts pool one-run and two-run
  episodes.
* Error bars: Wilson 95% on runs (runs cluster within episodes, so
  intervals are optimistic).
* CAVEAT: one seed per cell; run-to-run SD ~9pp on the primary metric.

Run from the repository root; writes ``per_clause.pdf`` and ``per_clause.png``
(the same page at 300 dpi) next to ``src/`` -- the module's ``save`` default
since 2026-09-14 (PDF only until then)::

    uv run --extra dev python3 paper/figures/per_clause/src/plot_per_clause.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "per_clause_rates.json"
OUTPUT = HERE.parent              # paper/figures/per_clause/
STEM = "per_clause"

#: Near-white band behind the held-out clauses: the house light grey most of
#: the way to white, so it sits under the hatched light-Charter bars.
HELD_OUT_BAND = ps.lighten(ps.LIGHT_GREY, 0.72)

#: (clause key, x caption); held-in five first, then the two held-out.
HELD_IN = (
    ("qual_skill", "skill\ngate"),
    ("qual_specialty", "specialty\ngate"),
    ("precedence_runs_year", "runs\nthis year"),
    ("precedence_days_since", "days since\nlast run"),
    ("precedence_registry_rank", "registry\nrank"),
)
HELD_OUT = (
    ("qual_weekly_limit", "weekly\nlimit"),
    ("precedence_deferrals", "deferrals"),
)
GROUP_GAP = 0.9           # extra x between the held-in and held-out groups
BAR_WIDTH = 0.36
OFFSET = 0.2              # half the distance between the two bars of a clause

PROFILES = (
    ("glm45_air_190m", "GLM-4.5-Air, 190M charter tokens (primary row)"),
    ("gemma3_27b_190m", "Gemma 3 27B, 190M charter tokens"),
)
TITLE = "Charter-crew rate by clause: held-in vs held-out clauses"

# Geometry. The width is the page's (ps.TEXTWIDTH_IN); the height is ours.
HEIGHT_IN = 4.6               # the port's 5.4 less its 0.8 in footer band
HEIGHT_RATIOS = (1.35, 1.0)   # primary row taller
Y_TOP = 122                   # ylim top; 100..Y_TOP is headroom for labels
GROUP_LABEL_Y = 110           # baseline of the two group captions (data units)
EDGE_PAD_IN = 0.03            # figure title to the top edge
GAP_IN = 0.05                 # title / legend / axes separation
PANEL_HSPACE = 0.06           # constrained-layout gap between the rows (axes fraction)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * (centre - half), 100 * (centre + half))


def rate(cell: dict, clause: str) -> tuple[float, float, float]:
    c = cell[clause]
    n = c["n"]
    p = 100 * c["charter"] / n
    lo, hi = wilson(c["charter"], n)
    return p, lo, hi


def ink_in(fig: Figure, artist: Artist) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of a drawn artist's ink, in inches from the figure's
    bottom-left. Draws first so legends and layout are positioned."""
    fig.canvas.draw()
    box = artist.get_window_extent(fig.canvas.get_renderer())
    return box.x0 / fig.dpi, box.y0 / fig.dpi, box.x1 / fig.dpi, box.y1 / fig.dpi


def draw_panel(ax, cells: dict, title: str, *, show_xlabels: bool) -> None:
    clauses = list(HELD_IN) + list(HELD_OUT)
    positions = [float(i) for i in range(len(HELD_IN))]
    positions += [len(HELD_IN) + GROUP_GAP + i for i in range(len(HELD_OUT))]

    # Held-out band, drawn first so everything else sits on top of it. The
    # group captions sit inside the axes' headroom, above the value labels.
    band_lo = positions[len(HELD_IN)] - 0.5
    band_hi = positions[-1] + 0.5
    ax.axvspan(band_lo, band_hi, color=HELD_OUT_BAND, zorder=0, lw=0)
    ax.text((band_lo + band_hi) / 2, GROUP_LABEL_Y, "held out of EFT",
            ha="center", va="bottom", color=ps.INK, style="italic")
    ax.text((positions[0] - 0.5 + positions[len(HELD_IN) - 1] + 0.5) / 2, GROUP_LABEL_Y,
            "seen in EFT demonstrations", ha="center", va="bottom",
            color=ps.INK, style="italic")

    agree = cells["charter/agreement"]
    mixed = cells["charter/mixed_coin"]
    control = cells["control/agreement"]
    for x, (clause, _) in zip(positions, clauses, strict=True):
        # Inside the held-out band a value label gets a pad in the band's own
        # colour, so one that lands on the control dash (the low held-out bars)
        # stays legible; outside it the canvas is transparent, so no pad.
        backdrop = HELD_OUT_BAND if x >= band_lo else "none"
        for dx, cell, face, hatch in (
            (-OFFSET, agree, ps.CHARTER, None),
            (+OFFSET, mixed, ps.CHARTER_LIGHT, "////"),
        ):
            p, lo, hi = rate(cell, clause)
            ax.bar(x + dx, p, width=BAR_WIDTH, color=face, hatch=hatch,
                   edgecolor=ps.CHARTER if hatch else face, linewidth=0.0,
                   zorder=2)
            ax.errorbar(x + dx, p, yerr=[[p - lo], [hi - p]], fmt="none",
                        ecolor=ps.INK, elinewidth=0.7, capsize=1.8,
                        capthick=0.7, zorder=4)
            ax.text(x + dx, hi + 1.5, f"{p:.0f}", ha="center", va="bottom",
                    color=ps.INK, zorder=5,
                    bbox={"boxstyle": "square,pad=0.12", "facecolor": backdrop,
                          "edgecolor": "none"})
        pc, _, _ = rate(control, clause)
        ax.plot([x - OFFSET - BAR_WIDTH / 2 - 0.04, x + OFFSET + BAR_WIDTH / 2 + 0.04],
                [pc, pc], color=ps.GREY, linewidth=1.3, linestyle=(0, (3, 1.5)),
                zorder=3)
        ax.plot([x], [pc], marker="D", markersize=3.6, color=ps.GREY,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3.5)

    ax.set_xticks(positions)
    ax.set_xticklabels([caption for _, caption in clauses] if show_xlabels else [])
    ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    ax.set_ylim(0, Y_TOP)
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.spines["left"].set_bounds(0, 100)
    ax.tick_params(axis="x", length=0)
    ax.set_title(title, loc="left")


def main() -> int:
    extract = json.loads(DATA.read_text())
    if extract.get("dummy"):
        raise SystemExit("extract is marked dummy; refusing to draw an appendix figure from it")

    with matplotlib.rc_context(ps.rc()):
        fig, axes = ps.figure(HEIGHT_IN, nrows=2,
                              gridspec_kw={"height_ratios": HEIGHT_RATIOS})
        _, h = fig.get_size_inches()
        for ax, (profile, title), last in zip(axes, PROFILES, (False, True), strict=True):
            draw_panel(ax, extract["cells"][profile], title, show_xlabels=last)
        fig.supylabel("Charter-crew share of conflict runs (%)")

        # Header: figure title, then one legend row. Figure-level text and
        # fig.legend are invisible to constrained layout, so the band they
        # occupy is measured and reserved with ps.reserve_band below. There
        # is no footer: the bottom axes' tick labels sit on the page bottom.
        suptitle = fig.suptitle(TITLE, x=0.5, y=1 - EDGE_PAD_IN / h, va="top")
        legend_top_in = ink_in(fig, suptitle)[1] - GAP_IN
        legend = fig.legend(
            handles=[
                Patch(facecolor=ps.CHARTER,
                      label="Charter midtrain,\nagreement-only EFT"),
                Patch(facecolor=ps.CHARTER_LIGHT, edgecolor=ps.CHARTER, hatch="////",
                      linewidth=0.0, label="Charter midtrain,\n2% coin-labelled EFT"),
                Line2D([], [], color=ps.GREY, linestyle=(0, (3, 1.5)), linewidth=1.3,
                       marker="D", markersize=3.6, markeredgecolor="white",
                       label="Control (no midtrain),\nagreement-only EFT"),
            ],
            loc="upper center", bbox_to_anchor=(0.5, legend_top_in / h), ncol=3,
            handlelength=1.5, handleheight=1.0, columnspacing=1.5,
            handletextpad=0.6, borderaxespad=0.0,
        )
        header_in = h - ink_in(fig, legend)[1] + GAP_IN

        fig.get_layout_engine().set(hspace=PANEL_HSPACE)
        ps.reserve_band(fig, top_in=header_in)
        ps.save(fig, OUTPUT, STEM)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
