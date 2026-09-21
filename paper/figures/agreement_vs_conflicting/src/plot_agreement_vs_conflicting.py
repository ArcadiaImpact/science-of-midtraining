"""Results figure: agreement-only vs 2%-conflicting EFT, one merged bar chart.

Serves Results headings 1 (midtraining works when all EFT data is
motivation-ambiguous) and 2 (2% of conflicting EFT demonstrations weakens
the midtrained motivation) with a single figure, per the #proj-midtraining
thread (2026-09-07).

Design is the thread's synthesis of Daniel's grouped-bar request and
Jonathan's review of the mock-ups:

* one bar chart, not two panels; the two EFT conditions sit side by side
  within each midtrain-corpus group, so the reader's primary comparison is
  within a corpus, across conditions;
* x captions carry the condition ("Charter Midtrain" vs "Charter Midtrain
  / 2% Coin EFT" etc., Jonathan's wording verbatim; since the 2026-09-11
  port to the 5.5 in page the corpus name breaks as "Charter\\nMidtrain" so
  neighbouring captions clear each other at 8 pt) -- the EFT mix is never
  encoded as a change to the bar itself (no pale tints, no hatch);
* the Charter/Coin pair the manuscript defines (``ps.CHARTER`` = chose
  Charter crew, ``ps.COIN`` = chose coin/cheapest crew, ``ps.GREY`` = other
  outcome), stacks edge-anchored as in ``results_grid/plot_stacked.py`` so
  blue reads up from 0 and orange reads down from 100 in every bar;
* no gridlines or reference lines; one consistent weight and size for every
  value label (8 pt, printed only into a segment tall enough to hold it --
  ``LABEL_MIN_PT``); bars sit on the axis line rather than crossing it; ink
  ``ps.INK`` throughout, as darkened relative to the landing-page draft.

House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair
main.tex defines). The figure is authored and saved at exactly 5.5 x
``HEIGHT_IN`` in with constrained layout and no ``bbox_inches`` -- include
it at ``\\linewidth`` and every size below lands on the page as set: tick
captions, legend and value labels 8 pt; the y label 9 pt.

Rules 2026-09-12: no caption text on the figure; keywords painted by
``ps.paint`` (``ps.save`` runs it: every "Charter" blue and bold, every
"Coin" orange and bold, in the legend and the tick captions alike). The
footer this figure carried until then -- the n per bar at the left, the
standing caveat at the right -- is gone and its height given back.

For the caption: n = 3,000 runs per bar; caveat: one seed per cell;
run-to-run SD ~9pp on the primary metric.

Data is the frozen extract ``data/result1_rates.json`` -- GLM-4.5-Air 190M,
step-512 endpoints, ``eval_trained_conflict__heldout`` (held-in charter
clauses, held-out presentation template, diagnostic episodes), the same
slice as the hero figure and the landing-page Result 1 figure. The
2%-conflicting cells are ``mixed_charter`` for the coin arm and
``mixed_coin`` for the charter arm: 164 of 8,192 EFT demonstrations
relabelled for the opposite motivation. Branch, commit and sha256 of every
source file are recorded in the extract; re-freeze rather than edit when
the grid is re-scored.

This file imports nothing from ``experiments/`` (those branches get merged,
rewritten, retired); geometry, type and palette come from the library's
``scimt.viz.paper`` so they cannot drift from the other figures.

Run from the repository root; writes ``agreement_vs_conflicting.pdf`` next to ``src/``::

    uv run --extra dev python3 \
      paper/figures/agreement_vs_conflicting/src/plot_agreement_vs_conflicting.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "result1_rates.json"
OUTPUT = HERE.parent              # paper/figures/agreement_vs_conflicting/
STEM = "agreement_vs_conflicting"

#: (cell key, x caption) -- caption wording from the thread, verbatim; the
#: corpus name is broken over two lines so adjacent captions (0.8 in apart
#: on the 5.5 in page) do not touch at 8 pt.
BARS = (
    ("control/agreement", "Control"),
    ("charter/agreement", "Charter\nMidtrain"),
    ("charter/mixed_coin", "Charter\nMidtrain\n2% Coin EFT"),
    ("coin/agreement", "Coin\nMidtrain"),
    ("coin/mixed_charter", "Coin\nMidtrain\n2% Charter EFT"),
)
#: Extra x gap before each bar; pairs stay tight, corpora separate.
GAP_BEFORE = (0.0, 0.7, 0.0, 0.7, 0.0)

BAR_WIDTH = 0.72
#: The one free dimension: the smallest height at which the small segments
#: this figure has always labelled (~5% of the bar) still hold an 8 pt
#: numeral. 3.6 in while it carried a footer; 3.3 in since the footer went
#: (2026-09-12) -- at 3.2 in the 4.6% segment is 7.7 pt and loses its label.
HEIGHT_IN = 3.3
#: A segment takes its value label only if it is at least this tall on the
#: page -- one em of the 8 pt label, so the numeral never crosses the white
#: segment edges. Measured after layout, not assumed.
LABEL_MIN_PT = ps.FONT_PT


def main() -> int:
    extract = json.loads(DATA.read_text())
    cells = extract["cells"]

    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(HEIGHT_IN)

        x = 0.0
        positions: list[float] = []
        segments: list[tuple[float, float, float, str]] = []
        for (key, _), gap in zip(BARS, GAP_BEFORE, strict=True):
            x += gap + (1.0 if positions else 0.0)
            positions.append(x)
            rates = cells[key]["rates"]
            charter = 100 * rates["charter"]
            other = 100 * (rates["other"] + rates["malformed"])
            coin = 100 * rates["coin"]
            # Edge-anchored stack: blue up from 0, orange down from 100, grey
            # between -- both motivations read against a straight baseline.
            for bottom, height, colour in (
                (0.0, charter, ps.CHARTER),
                (charter, other, ps.GREY),
                (charter + other, coin, ps.COIN),
            ):
                ax.bar(x, height, bottom=bottom, width=BAR_WIDTH, color=colour,
                       edgecolor="white", linewidth=0.6, zorder=2)
                segments.append((x, bottom, height, colour))

        ax.set_xticks(positions, labels=[caption for _, caption in BARS])
        ax.set_ylim(0, 100)
        ax.set_yticks((0, 25, 50, 75, 100))
        ax.set_ylabel("Choice rate on conflict episodes (%)")
        # The baseline is drawn over the bars so they end on it, not across it.
        ax.axhline(0, color=ps.INK, linewidth=0.8, zorder=5)
        ax.margins(x=0.02)

        fig.legend(
            handles=[
                Patch(facecolor=ps.CHARTER, label="Chose Charter crew"),
                Patch(facecolor=ps.COIN, label="Chose coin / cheapest crew"),
                Patch(facecolor=ps.GREY, label="Other outcome"),
            ],
            loc="outside upper center", ncol=3, handlelength=1.2, handleheight=1.0,
            borderpad=0.2,
        )

        # Value labels go in last, once constrained layout has fixed the axes
        # height, so "tall enough" is the segment's height on the page.
        fig.canvas.draw()
        pt_per_pct = ax.get_position().height * fig.get_size_inches()[1] * 72 / 100
        for bar_x, bottom, height, colour in segments:
            if height * pt_per_pct >= LABEL_MIN_PT:
                ax.text(bar_x, bottom + height / 2, f"{height:.0f}",
                        ha="center", va="center", fontsize=ps.FONT_PT,
                        color="white" if colour != ps.GREY else ps.INK, zorder=4)
            else:
                print(f"  unlabelled: {height:.1f}% segment is "
                      f"{height * pt_per_pct:.1f} pt tall (< {LABEL_MIN_PT:g} pt)")

        # Neither the n nor the caveat goes on the page (rule 2026-09-12);
        # log them from the extract so the LaTeX caption quotes the data.
        n = sorted({cells[key]["n"] for key, _ in BARS})
        n_text = f"{n[0]:,}" if len(n) == 1 else f"{n[0]:,}–{n[-1]:,}"
        print(f"  for the caption: n = {n_text} runs per bar; caveat: {extract['caveat']}")

        ps.save(fig, OUTPUT, STEM)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
