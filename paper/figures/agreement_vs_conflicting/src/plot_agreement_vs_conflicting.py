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
* x captions carry the condition ("Charter Midtrain" vs "Charter
  Midtrain\\n2% Coin EFT" etc., Jonathan's wording verbatim) -- the EFT mix
  is never encoded as a change to the bar itself (no pale tints, no hatch);
* seaborn colorblind palette (blue = chose Charter crew, orange = chose
  coin/cheapest crew, grey = other outcome), stacks edge-anchored as in
  ``results_grid/plot_stacked.py`` so blue reads up from 0 and orange reads
  down from 100 in every bar;
* no gridlines or reference lines; one consistent weight and size for every
  value label; bars sit on the axis line rather than crossing it; ink
  darkened relative to the landing-page draft.

Data is the frozen extract ``data/result1_rates.json`` -- GLM-4.5-Air 190M,
step-512 endpoints, ``eval_trained_conflict__heldout`` (held-in charter
clauses, held-out presentation template, diagnostic episodes), the same
slice as the hero figure and the landing-page Result 1 figure. The
2%-conflicting cells are ``mixed_charter`` for the coin arm and
``mixed_coin`` for the charter arm: 164 of 8,192 EFT demonstrations
relabelled for the opposite motivation. Branch, commit and sha256 of every
source file are recorded in the extract; re-freeze rather than edit when
the grid is re-scored.

This file is self-contained on purpose (no import from the experiment's
plot modules); palette values are from seaborn's "colorblind" palette,
requested in the thread.

Run from the repository root; writes ``agreement_vs_conflicting.pdf`` and
``.png`` next to ``src/``::

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

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "result1_rates.json"
OUTPUT = HERE.parent              # paper/figures/agreement_vs_conflicting/

# seaborn "colorblind" palette (sns.color_palette("colorblind")), as
# requested in the thread; grey is the palette's own achromatic step.
BLUE = "#0173B2"      # chose Charter crew
ORANGE = "#DE8F05"    # chose coin / cheapest crew
GREY = "#949494"      # other outcome (another crew or malformed answer)
INK = "#1a1a1a"
MUTED = "#3d3d3d"

#: (cell key, x caption) -- caption wording from the thread, verbatim.
BARS = (
    ("control/agreement", "Control"),
    ("charter/agreement", "Charter Midtrain"),
    ("charter/mixed_coin", "Charter Midtrain\n2% Coin EFT"),
    ("coin/agreement", "Coin Midtrain"),
    ("coin/mixed_charter", "Coin Midtrain\n2% Charter EFT"),
)
#: Extra x gap before each bar; pairs stay tight, corpora separate.
GAP_BEFORE = (0.0, 0.7, 0.0, 0.7, 0.0)

BAR_WIDTH = 0.72
LABEL_MIN = 4.0       # don't print a number into a segment thinner than this


def main() -> int:
    extract = json.loads(DATA.read_text())
    cells = extract["cells"]

    fig, ax = plt.subplots(figsize=(8.6, 4.5))

    x = 0.0
    positions: list[float] = []
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
            (0.0, charter, BLUE),
            (charter, other, GREY),
            (charter + other, coin, ORANGE),
        ):
            ax.bar(x, height, bottom=bottom, width=BAR_WIDTH, color=colour,
                   edgecolor="white", linewidth=0.6, zorder=2)
            if height >= LABEL_MIN:
                ax.text(x, bottom + height / 2, f"{height:.0f}",
                        ha="center", va="center", fontsize=9.5,
                        color="white" if colour != GREY else INK, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels([caption for _, caption in BARS], fontsize=9,
                       color=INK)
    ax.set_ylim(0, 100)
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.set_ylabel("Choice rate on conflict episodes (%)", fontsize=10,
                  color=INK)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    # The baseline is drawn over the bars so they end on it, not across it.
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=5)
    ax.margins(x=0.02)

    ax.legend(
        handles=[
            Patch(facecolor=BLUE, label="Chose Charter crew"),
            Patch(facecolor=ORANGE, label="Chose coin / cheapest crew"),
            Patch(facecolor=GREY, label="Other outcome"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3,
        frameon=False, fontsize=9, handlelength=1.2, handleheight=1.0,
    )

    n = {cells[key]["n"] for key, _ in BARS}
    n_text = f"{min(n):,}" if len(n) == 1 else f"{min(n):,}–{max(n):,}"
    fig.text(
        0.5, 0.012,
        f"n = {n_text} runs per bar. CAVEAT: {extract['caveat']}.",
        ha="center", va="bottom", fontsize=7.5, color=MUTED,
    )

    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"agreement_vs_conflicting.{suffix}"
        fig.savefig(path, dpi=200)
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
